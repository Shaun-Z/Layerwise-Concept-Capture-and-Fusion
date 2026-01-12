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
    
    This version computes gradients during the forward pass by:
    1. Setting concept vectors before forward via set_concept_vectors()
    2. Computing gradients w.r.t. attention weights as each block finishes
    3. Optionally using async computation for efficiency
    
    The results match those of TorchvisionGradWrapper.
    
    Key characteristics of torchvision ViT:
    - Tensor layout: (B, N, D) - same as timm
    - Block access: model.encoder.layers[idx]
    - Normalization: Pre-norm (ln_1 before attn, ln_2 before MLP)
    - MLP components: Sequential with [0]=fc1, [1]=GELU, [2]=Dropout, [3]=fc2
    - Attention: self_attention is MultiheadAttention with out_proj
    - Final layers: model.encoder.ln + model.heads[0] (classification head)
    - GELU: Uses exact GELU (approximate='none')
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False,
                 power: int = 2, async_compute: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        # Store patch info for later use
        self._patch_size = model.patch_size
        self._hidden_dim = model.hidden_dim
        self.num_heads = model.encoder.layers[0].self_attention.num_heads
        self.power = power
        self.async_compute = async_compute
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if async_compute else None
        self._futures = []
        
        self.reset()
        
        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            
            # Enable gradients for the block parameters
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
        self.attn_weights = []
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
        """Finalize hook that computes gradient and map immediately after block forward."""
        # Save block output
        self.block_outputs.append(output)
        
        # Retrieve attention weights stored by the modified forward
        if hasattr(module, '_attn_weights'):
            # attn_weights: (B*num_heads, N, N) from custom MHA forward
            self._current_attn_weight = module._attn_weights
            self.attn_weights.append(module._attn_weights)
        
        # If concept vectors are set, compute gradient and map immediately
        if self._concept_vectors is not None and self._current_attn_weight is not None:
            attn_weight = self._current_attn_weight
            block_output = output
            
            if self.async_compute and self._executor is not None:
                # Submit async computation
                future = self._executor.submit(
                    self._compute_gradient_and_map, 
                    block_output.detach().clone(), 
                    attn_weight,  # Note: attn_weight needs gradient, can't detach
                    self._concept_vectors.clone()
                )
                self._futures.append(future)
            else:
                # Compute synchronously
                self._compute_gradient_and_map(block_output, attn_weight, self._concept_vectors)

    def _compute_gradient_and_map(self, block_output: torch.Tensor, attn_weight: torch.Tensor, 
                                   concept_vectors: torch.Tensor):
        """Compute gradient and map for a single block.
        
        This method matches the computation in TorchvisionGradWrapper.dot_concept_vectors().
        """
        w = h = int(math.sqrt(block_output.shape[1] - 1))  # Exclude CLS token, layout is (B, N, D)
        
        # Zero gradients
        self.zero_grad()
        
        # block_output: (B, N, D)
        cls_feat = block_output[:, 0, ...]  # (B, D) - CLS token
        
        # For torchvision, we work in hidden_dim space (no projection like OpenCLIP)
        latent_feat = F.normalize(self.encoder.ln(cls_feat), dim=-1)  # (B, D)
        
        # Compute similarity with concept vectors
        sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
        weight = torch.abs(sim_bm.clone().detach()).pow(self.power)
        sim_bm = sim_bm * weight  # (B, num_concepts)
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

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """Compute concept activation maps.
        
        If concept vectors were set before forward via set_concept_vectors(),
        maps are already computed during forward pass.
        
        If concept vectors were NOT set before forward, this method computes
        the maps using stored attention weights and block outputs (same as GradWrapper).
        
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
        
        # Otherwise, compute maps now (same logic as GradWrapper)
        if not self.block_outputs or not self.attn_weights:
            raise ValueError("No block outputs or attention weights stored. Please run a forward pass first.")
        
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
            sim_bm = sim_bm * weight  # (B, num_concepts)
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
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

    def close(self):
        """Clean up resources, including the ThreadPoolExecutor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

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
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.grads = []
        self.maps = []
        self.sim_bms = []
