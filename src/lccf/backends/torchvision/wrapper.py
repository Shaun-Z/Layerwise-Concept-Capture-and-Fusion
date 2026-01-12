# src/lccf/backends/torchvision/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import types
import concurrent.futures

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper
from .functional import EncoderBlock_forward, MultiheadAttention_forward_batch_first

"""
Description:
 - The wrapper accepts an already constructed torchvision model (e.g., via torchvision.models.vit_b_16). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """


class TorchvisionWrapper(CopyAttrWrapper):
    """
    A torchvision-specific derivative of CopyAttrWrapper that provides convenient methods for forward.
    Important: This wrapper assumes that you are passing in a torchvision model (typically the model returned by torchvision.models.vit_b_16).
    If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    
    Key characteristics of torchvision ViT:
    - Tensor layout: (B, N, D) - same as timm
    - Block access: model.encoder.layers[idx]
    - Normalization: Pre-norm (ln_1 before attn, ln_2 before MLP)
    - MLP components: Sequential with [0]=fc1, [1]=GELU, [2]=Dropout, [3]=fc2
    - Attention: self_attention is MultiheadAttention with out_proj
    - Final layers: model.encoder.ln + model.heads[0] (classification head)
    - GELU: Uses exact GELU (approximate='none')
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        self.reset()
        
        # Store patch info for later use
        self._patch_size = model.patch_size
        self._hidden_dim = model.hidden_dim
        
        # Register hooks to the specified layers to capture attention outputs
        # torchvision uses pre-norm: x = x + attn(ln_1(x)), x = x + mlp(ln_2(x))
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            
            # Hook on self_attention output (MultiheadAttention returns tuple(output, attn_weights))
            # Note: out_proj hook may not work due to PyTorch's fast path, so we hook on the full self_attention
            block.self_attention.register_forward_hook(self._save_attn_output)  # (b, n, d)
            # Hook on ln_2 to scale by LayerNorm
            block.ln_2.register_forward_hook(self._aggregate_ln2)  # (b, n, d)
            # Hook on MLP fc1 (mlp[0])
            block.mlp[0].register_forward_hook(self._aggregate_fc1)
            # Hook on MLP fc2 (mlp[3])
            block.mlp[3].register_forward_hook(self._aggregate_fc2)  # (b, n, d)
            # Final hook on block output
            block.register_forward_hook(self._finalize_hook)  # (b, n, d)

    def reset(self):
        """Reset the stored results and maps."""
        self.tmp = None
        self.result = []
        self.maps = []
        self.normed_clss = []

    def _save_attn_output(self, module, input, output):
        # torchvision MultiheadAttention returns (output, attn_weights)
        # output: (B, N, D)
        self.tmp = output[0].detach()
    
    def _aggregate_ln2(self, module, input, output):
        # input[0] is the input to ln_2: (B, N, D)
        # We need to account for the LayerNorm scaling
        std = input[0].std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b n -> b n 1')
        self.tmp *= module.weight
    
    def _aggregate_fc1(self, module, input, output):
        # Apply fc1 weight transformation
        self.tmp = self.tmp @ module.weight.T
        # torchvision uses exact GELU: approximate='none'
        # Gradient of exact GELU: GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
        # d/dx GELU(x) = Φ(x) + x * φ(x) where φ is the PDF
        x_ = self.tmp
        # Standard normal CDF and PDF
        sqrt_2 = math.sqrt(2.0)
        phi = 0.5 * (1.0 + torch.erf(x_ / sqrt_2))  # CDF
        pdf = torch.exp(-0.5 * x_ * x_) / math.sqrt(2.0 * math.pi)  # PDF
        grad = phi + x_ * pdf
        self.tmp = self.tmp * grad
    
    def _aggregate_fc2(self, module, input, output):
        # Apply fc2 weight transformation
        self.tmp = self.tmp @ module.weight.T
    
    def _finalize_hook(self, module, input, output):
        # Block output: (B, N, D)
        # Extract CLS token (first token)
        cls = output[:, :1, ...]  # (B, 1, D)
        
        # Apply final LayerNorm scaling
        std = cls.std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b 1 -> b 1 1')
        self.tmp *= self.encoder.ln.weight  # torchvision uses model.encoder.ln as final LayerNorm
        
        # For torchvision ViT classification models, the head is a classification head
        # For concept vectors, we work in the hidden_dim space
        cls_encoded = self.encoder.ln(cls)  # (B, 1, D)
        val = cls_encoded.norm(dim=-1, keepdim=True)  # (B, 1, 1)
        self.tmp /= val
        
        self.normed_clss.append(F.normalize(cls_encoded, dim=-1))
        
        # Transpose to (N, B, D) to match format for dot_concept_vectors
        self.result.append(rearrange(self.tmp, 'b n d -> n b d'))

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """Compute concept activation maps.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
        """
        w = h = int(math.sqrt(self.result[0].shape[0] - 1))  # Exclude CLS token
        for i, res in enumerate(self.result):
            # res: (N, B, D) where N = 1 + H*W (CLS + patches)
            prod = torch.einsum('n b d, m d -> n b m', res, concept_vectors)
            # normed_clss[i]: (B, 1, D) - transpose to (1, B, D)
            normed_cls = rearrange(self.normed_clss[i], 'b 1 d -> 1 b d')
            weight = torch.einsum('n b d, m d -> n b m', normed_cls, concept_vectors)
            prod = prod * weight
            map = torch.clamp(prod - prod.mean(dim=0, keepdim=True), min=0.)  # negative gradient
            map = rearrange(map[1:, ...], '(h w) b m -> h w b m', h=h, w=w)  # Exclude CLS token
            self.maps.append(map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m -> h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')
        
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

    def _get_device_for_call(self, device: Optional[str] = None):
        # Try to get the device from the original model's parameters, otherwise use the passed device or cpu
        orig = self.original_model()
        if device is not None:
            return torch.device(device)
        try:
            # Find the device of the first parameter
            for p in orig.parameters():
                return p.device
        except Exception:
            pass
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # Move the original model to the target device as well
        orig = self.original_model()
        try:
            if hasattr(orig, "to"):
                orig.to(*args, **kwargs)
        except Exception:
            # Ignore errors when moving the original model, but still try to call the parent class's to
            pass
        # CopyAttrWrapper has no tensor buffers of its own, still call the parent class (it will move parameters registered to the wrapper)
        return super().to(*args, **kwargs)


class TorchvisionGradWrapper(CopyAttrWrapper):
    """
    A torchvision-specific derivative of CopyAttrWrapper that uses autograd to capture gradients of attention maps.
    This mirrors the functionality of OpenCLIPGradWrapper for torchvision models.
    
    Key differences from TorchvisionWrapper:
    - Uses torch.autograd.grad to compute gradients of attention weights
    - Overrides EncoderBlock forward to capture attention weights with need_weights=True
    - Requires the model to be in eval mode for gradient computation
    
    Tensor layout: (B, N, D) - same as TorchvisionWrapper
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Store patch info for later use
        self._patch_size = model.patch_size
        self._hidden_dim = model.hidden_dim
        self.num_heads = model.encoder.layers[0].self_attention.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            for name, param in block.named_parameters():
                param.requires_grad = True
            # Override self_attention forward to use custom MHA that returns attention weights
            block.self_attention.forward = types.MethodType(MultiheadAttention_forward_batch_first, block.self_attention)
            # Override block forward to capture attention weights
            block.forward = types.MethodType(EncoderBlock_forward, block)
            
            block.register_forward_hook(self._save_block_hook)  # (B, N, D)

    def _save_block_hook(self, module, input, output):
        # Block output: (B, N, D)
        self.block_outputs.append(output)
        # Retrieve attention weights stored by the modified forward
        if hasattr(module, '_attn_weights'):
            # attn_weights: (B*num_heads, N, N) from custom MHA forward
            self.attn_weights.append(module._attn_weights)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """Compute gradient-based concept activation maps.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
            power (int): Power for similarity scaling. Default: 2
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[1] - 1))  # Exclude CLS token, layout is (B, N, D)
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            # Zero gradients of the model
            self.zero_grad()
            
            # block_output: (B, N, D)
            cls_feat = block_output[:, 0, ...]  # (B, D) - CLS token
            
            # For torchvision, we work in hidden_dim space (no projection like OpenCLIP)
            latent_feat = F.normalize(self.encoder.ln(cls_feat), dim=-1)  # (B, D)
            
            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
            weight = torch.abs(sim_bm.clone().detach()).pow(power)
            sim_bm *= weight  # (B, num_concepts)
            self.sim_bms.append(weight)
            sim = sim_bm.sum(dim=0)  # (B, num_concepts) -> (num_concepts)
            
            # Compute gradients of sim w.r.t. attn_weight
            # attn_weight shape: (B*num_heads, N, N) from custom MHA forward
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            
            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, B*num_heads, N, N)
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (num_concepts, B, num_heads, N, N)
            self.grads.append(grad)
            
            # Average over heads and query positions, exclude CLS token
            image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (num_concepts, B, N-1)
            expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
            self.maps.append(expl_map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m -> h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.grads = []
        self.maps = []
        self.sim_bms = []


class ATorchvisionWrapper(CopyAttrWrapper):
    """
    Asynchronous torchvision wrapper that computes gradients during the forward pass.
    
    This version computes gradients using torch.autograd.grad() during the forward pass
    when concept vectors are set beforehand via set_concept_vectors().
    
    This version computes gradients during the forward pass by:
    1. Setting concept vectors before forward via set_concept_vectors()
    2. Computing gradients w.r.t. attention weights as each block finishes (synchronously)
    
    Note: Gradient computation is synchronous because torch.autograd.grad() requires
    the same thread that created the computation graph.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False,
                 power: int = 2, async_compute: bool = True):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        # Store patch info for later use
        self._patch_size = model.patch_size
        self._hidden_dim = model.hidden_dim
        self.num_heads = model.encoder.layers[0].self_attention.num_heads
        self.head_dim = self._hidden_dim // self.num_heads
        self.power = power
        self.async_compute = async_compute
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(layer_indices))) if async_compute and layer_indices else None
        self._futures = []
        self._layer_indices = layer_indices
        
        self.reset()
        
        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            
            # Enable requires_grad for all parameters in the block (like GradWrapper)
            for name, param in block.named_parameters():
                param.requires_grad = True
            
            # Override self_attention forward to use custom MHA that returns attention weights (same as GradWrapper)
            block.self_attention.forward = types.MethodType(MultiheadAttention_forward_batch_first, block.self_attention)
            # Override block forward to capture attention weights
            block.forward = types.MethodType(EncoderBlock_forward, block)
            
            # Register hook
            block.register_forward_hook(self._finalize_and_compute_hook)  # (B, N, D)

    def reset(self):
        """Reset the stored results and maps."""
        self._concept_vectors = None
        self._current_attn_weight = None
        self._current_v_values = None
        self._current_attn_module = None
        self.attn_weights = []
        self.v_values = []
        self.block_outputs = []
        self.grads = []
        self.maps = []
        self.sim_bms = []
        # Cancel any pending async computations
        if hasattr(self, '_futures'):
            for f in self._futures:
                f.cancel()
            self._futures = []

    def set_concept_vectors(self, concept_vectors: torch.Tensor, power: int = None):
        """Set concept vectors before forward pass.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
            power (int, optional): Power for similarity weighting. Default: use self.power
        """
        self._concept_vectors = concept_vectors.detach()
        if power is not None:
            self.power = power

    def _finalize_and_compute_hook(self, module, input, output):
        """Finalize hook that computes gradient and map immediately after block forward.
        
        Note: Gradient computation must be synchronous because torch.autograd.grad()
        requires the same thread that created the computation graph.
        """
        # Save block output (keep gradient connection for autograd)
        self.block_outputs.append(output)
        
        # Retrieve attention weights stored by the modified forward
        # DO NOT detach attn_weights - we need the gradient graph for torch.autograd.grad()
        if hasattr(module, '_attn_weights'):
            # attn_weights: (B*num_heads, N, N) from custom MHA forward
            self._current_attn_weight = module._attn_weights
            self.attn_weights.append(module._attn_weights)
        
        # If concept vectors are set, compute gradient and map immediately (synchronously)
        if self._concept_vectors is not None and self._current_attn_weight is not None:
            # Compute using autograd (same as GradWrapper's dot_concept_vectors)
            self._compute_gradient_and_map_autograd(output, self._current_attn_weight, self._concept_vectors)

    def _compute_gradient_and_map_autograd(self, block_output: torch.Tensor, attn_weight: torch.Tensor,
                                            concept_vectors: torch.Tensor):
        """Compute gradient and map for a single block using torch.autograd.grad().
        
        This uses the same approach as GradWrapper's dot_concept_vectors for exact consistency.
        
        Args:
            block_output: Block output tensor (B, N, D)
            attn_weight: Attention weights (B*num_heads, N, N) with gradient graph
            concept_vectors: Concept vectors (num_concepts, D)
        """
        w = h = int(math.sqrt(block_output.shape[1] - 1))  # Exclude CLS token, layout is (B, N, D)
        
        # Zero gradients of the model
        self.zero_grad()
        
        # block_output: (B, N, D)
        cls_feat = block_output[:, 0, ...]  # (B, D) - CLS token
        
        # For torchvision, we work in hidden_dim space (no projection like OpenCLIP)
        latent_feat = F.normalize(self.encoder.ln(cls_feat), dim=-1)  # (B, D)
        
        # Compute similarity with concept vectors
        sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
        weight = torch.abs(sim_bm.clone().detach()).pow(self.power)
        sim_bm_weighted = sim_bm * weight  # (B, num_concepts)
        self.sim_bms.append(weight)
        sim = sim_bm_weighted.sum(dim=0)  # (B, num_concepts) -> (num_concepts)
        
        # Compute gradients of sim w.r.t. attn_weight
        # attn_weight shape: (B*num_heads, N, N) from custom MHA forward
        eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
        
        grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                   grad_outputs=eye,
                                   retain_graph=True,
                                   create_graph=False,
                                   is_grads_batched=True)[0]  # (num_concepts, B*num_heads, N, N)
        grad = torch.clamp(grad, min=0.)
        grad = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (num_concepts, B, num_heads, N, N)
        self.grads.append(grad)
        
        # Average over heads and query positions, exclude CLS token
        image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (num_concepts, B, N-1)
        expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
        self.maps.append(expl_map)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """Compute concept activation maps.
        
        If concept vectors were set before forward via set_concept_vectors(),
        maps are already computed during forward pass.
        
        If concept vectors were NOT set before forward, this method computes
        the maps using stored attention weights and block outputs with autograd.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
            power (int): Power for similarity weighting. Default: 2
        """
        # Wait for any async computations to complete
        if self.async_compute and self._futures:
            concurrent.futures.wait(self._futures)
            self._futures = []
        
        # If maps are already computed during forward, return
        if self.maps:
            return
        
        # Otherwise, compute maps now using autograd (same as GradWrapper)
        if not self.block_outputs or not self.attn_weights:
            raise ValueError("No block outputs or attention weights stored. Please run a forward pass first.")
        
        self.power = power
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            self._compute_gradient_and_map_autograd(block_output, attn_weight, concept_vectors)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        # Wait for any async computations to complete
        if self.async_compute and self._futures:
            concurrent.futures.wait(self._futures)
            self._futures = []
            
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m -> h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        assert (maps_max - maps_min).all() != 0, "Division by zero: maps_max - maps_min contains zero values"
        maps = (maps - maps_min) / (maps_max - maps_min)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

    def close(self):
        """Clean up resources, including the ThreadPoolExecutor."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
