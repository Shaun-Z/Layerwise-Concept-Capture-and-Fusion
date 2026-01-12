# src/lccf/backends/timm/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import types
import concurrent.futures

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper
from .functional import Attention_forward

"""
Description:
 - The wrapper accepts an already constructed timm model (e.g., via timm.create_model). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """


class TimmWrapper(CopyAttrWrapper):
    """
    A timm-specific derivative of CopyAttrWrapper that provides convenient methods for forward_features.
    Important: This wrapper assumes that you are passing in a timm model (typically the model returned by timm.create_model).
    If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    
    Key differences from OpenCLIP:
    - Tensor layout: timm uses (B, N, D) vs OpenCLIP uses (N, B, D)
    - Block access: model.blocks[idx] vs model.visual.transformer.resblocks[idx]
    - Normalization: Pre-norm (norm1 before attn, norm2 before MLP)
    - MLP components: fc1, act, fc2 vs c_fc, gelu, c_proj
    - Attention: Combined qkv + proj
    - Final layers: model.norm vs ln_post + proj
    - GELU: Uses exact GELU (approximate='none')
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        self.reset()
        
        # Store patch info for later use
        self._patch_size = model.patch_embed.proj.kernel_size[0]
        self._embed_dim = model.embed_dim
        
        # Register hooks to the specified layers to capture attention outputs
        # timm uses pre-norm: x = x + attn(norm1(x)), x = x + mlp(norm2(x))
        for idx in layer_indices:
            block = self.blocks[idx]
            
            # Hook on attention projection output
            block.attn.proj.register_forward_hook(self._save_attn_proj_output)  # (b, n, d)
            # Hook on norm2 to scale by LayerNorm
            block.norm2.register_forward_hook(self._aggregate_norm2)  # (b, n, d)
            # Hook on MLP fc1
            block.mlp.fc1.register_forward_hook(self._aggregate_fc1)
            # Hook on MLP fc2
            block.mlp.fc2.register_forward_hook(self._aggregate_fc2)  # (b, n, d)
            # Final hook on block output
            block.register_forward_hook(self._finalize_hook)  # (b, n, d)

    def reset(self):
        """Reset the stored results and maps."""
        self.tmp = None
        self.result = []
        self.maps = []
        self.normed_clss = []

    def _save_attn_proj_output(self, module, input, output):
        # timm attention proj output: (B, N, D)
        self.tmp = output.detach()
    
    def _aggregate_norm2(self, module, input, output):
        # input[0] is the input to norm2: (B, N, D)
        # We need to account for the LayerNorm scaling
        std = input[0].std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b n -> b n 1')
        self.tmp *= module.weight
    
    def _aggregate_fc1(self, module, input, output):
        # Apply fc1 weight transformation
        self.tmp = self.tmp @ module.weight.T
        # timm uses exact GELU: approximate='none'
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
        self.tmp *= self.norm.weight  # timm uses model.norm as final LayerNorm
        
        # For timm ViT classification models, there's typically a head projection
        # But for feature extraction, we use the normalized features directly
        # Note: Unlike OpenCLIP, timm doesn't have a separate projection after norm
        # The head is a classification head, not a feature projection
        
        # Apply the classification head weight for consistency with feature dimension
        # However, for concept vectors, we work in the embed_dim space
        cls_encoded = self.norm(cls)  # (B, 1, D)
        val = cls_encoded.norm(dim=-1, keepdim=True)  # (B, 1, 1)
        self.tmp /= val
        
        self.normed_clss.append(F.normalize(cls_encoded, dim=-1))
        
        # Transpose to (N, B, D) to match OpenCLIP format for dot_concept_vectors
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


class TimmGradWrapper(CopyAttrWrapper):
    """
    A timm-specific derivative of CopyAttrWrapper that uses autograd to capture gradients of attention maps.
    This mirrors the functionality of OpenCLIPGradWrapper for timm models.
    
    Key differences from TimmWrapper:
    - Uses torch.autograd.grad to compute gradients of attention weights
    - Overrides attention forward to capture attention weights
    - Requires the model to be in eval mode for gradient computation
    
    Tensor layout: (B, N, D) - same as TimmWrapper
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Store patch info for later use
        self._patch_size = model.patch_embed.proj.kernel_size[0]
        self._embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.blocks[idx]
            for name, param in block.named_parameters():
                param.requires_grad = True
            # Override attention forward to return attention weights
            block.attn.forward = types.MethodType(Attention_forward, block.attn)
            
            block.attn.register_forward_hook(self._save_attn_hook)  # (B, N, D), (B*num_heads, N, N)
            block.register_forward_hook(self._save_block_hook)  # (B, N, D)

    def _save_attn_hook(self, module, input, output):
        # Retrieve attention weights stored by the modified forward
        # attn_weights: (B, num_heads, N, N) - stored in original shape
        if hasattr(module, '_attn_weights'):
            self.attn_weights.append(module._attn_weights)

    def _save_block_hook(self, module, input, output):
        # Block output: (B, N, D)
        self.block_outputs.append(output)

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
            
            # For timm, we work in embed_dim space (no projection like OpenCLIP)
            latent_feat = F.normalize(self.norm(cls_feat), dim=-1)  # (B, D)
            
            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
            weight = torch.abs(sim_bm.clone().detach()).pow(power)
            sim_bm *= weight  # (B, num_concepts)
            sim = sim_bm.sum(dim=0)  # (B, num_concepts) -> (num_concepts)
            self.sim_bms.append(weight)
            
            # Compute gradients of sim w.r.t. attn_weight
            # attn_weight shape: (B, num_heads, N, N)
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            
            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, B, num_heads, N, N)
            grad = torch.clamp(grad, min=0.)
            # grad is already in shape (num_concepts, B, num_heads, N, N)
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


class ATimmWrapper(CopyAttrWrapper):
    """
    Asynchronous timm wrapper that computes gradients during the forward pass.
    
    This version uses MANUAL gradient computation (not torch.autograd.grad) to enable
    true async computation. Gradients are computed w.r.t. attention weights by manually
    backpropagating through the network.
    
    This version computes gradients during the forward pass by:
    1. Setting concept vectors before forward via set_concept_vectors()
    2. Computing gradients w.r.t. attention weights as each block finishes (async capable)
    3. Using ThreadPoolExecutor for true parallel computation
    
    The manual gradient approach:
    - Stores V values during attention forward
    - Computes d(loss)/d(attn_weights) = d(loss)/d(attn_output) @ V^T
    - This can run in a separate thread since no autograd is involved
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False,
                 power: int = 2, async_compute: bool = True):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        # Store patch info for later use
        self._patch_size = model.patch_embed.proj.kernel_size[0]
        self._embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        self.head_dim = model.blocks[0].attn.head_dim
        self.power = power
        self.async_compute = async_compute
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(layer_indices)) if async_compute else None
        self._futures = []
        self._layer_indices = layer_indices
        
        self.reset()
        
        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.blocks[idx]
            
            # Override attention forward to store attention weights and V values
            block.attn.forward = types.MethodType(Attention_forward, block.attn)
            
            # Register hooks
            block.attn.register_forward_hook(self._save_attn_hook)  # Save attn_weights and V
            block.register_forward_hook(self._finalize_and_compute_hook)  # Trigger gradient computation

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

    def _save_attn_hook(self, module, input, output):
        """Save attention weights and V values from the attention module."""
        # Retrieve attention weights stored by the modified forward
        # attn_weights: (B, num_heads, N, N) - stored in original shape
        if hasattr(module, '_attn_weights'):
            self._current_attn_weight = module._attn_weights.detach()
            self.attn_weights.append(module._attn_weights.detach())
        if hasattr(module, '_v_values'):
            self._current_v_values = module._v_values.detach()
            self.v_values.append(module._v_values.detach())
        self._current_attn_module = module

    def _finalize_and_compute_hook(self, module, input, output):
        """Finalize hook that computes gradient and map immediately after block forward."""
        # Save block output (detached for async computation)
        block_output = output.detach()
        self.block_outputs.append(block_output)
        
        # If concept vectors are set, compute gradient and map
        if self._concept_vectors is not None and self._current_attn_weight is not None and self._current_v_values is not None:
            attn_weight = self._current_attn_weight.clone()
            v_values = self._current_v_values.clone()
            attn_module = self._current_attn_module
            
            if self.async_compute and self._executor is not None:
                # Submit async computation (now possible with manual gradients!)
                future = self._executor.submit(
                    self._compute_manual_gradient_and_map, 
                    block_output.clone(),
                    attn_weight,
                    v_values,
                    attn_module,
                    self._concept_vectors.clone()
                )
                self._futures.append(future)
            else:
                # Compute synchronously
                self._compute_manual_gradient_and_map(block_output, attn_weight, v_values, attn_module, self._concept_vectors)

    def _compute_manual_gradient_and_map(self, block_output: torch.Tensor, attn_weight: torch.Tensor,
                                          v_values: torch.Tensor, attn_module, concept_vectors: torch.Tensor):
        """Compute gradient and map for a single block using MANUAL gradient computation.
        
        This method computes d(sim)/d(attn_weights) manually without using torch.autograd.grad(),
        which enables true async computation in a separate thread.
        
        The gradient chain is:
        1. sim = (latent_feat @ concept_vectors.T * weight).sum(dim=0)
        2. latent_feat = normalize(norm(cls_feat))
        3. cls_feat = block_output[:, 0, :]  (CLS token)
        
        For the attention gradient:
        - d(attn @ V)/d(attn) = gradient @ V^T
        """
        w = h = int(math.sqrt(block_output.shape[1] - 1))  # Exclude CLS token, layout is (B, N, D)
        B, N, D = block_output.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        
        # === Step 1: Compute similarity and weight ===
        # block_output: (B, N, D)
        cls_feat = block_output[:, 0, ...]  # (B, D) - CLS token
        
        # For timm, we work in embed_dim space
        normed_cls = self.norm(cls_feat)  # (B, D)
        latent_feat = F.normalize(normed_cls, dim=-1)  # (B, D)
        
        # Compute similarity with concept vectors
        sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
        weight = torch.abs(sim_bm.clone()).pow(self.power)
        self.sim_bms.append(weight)
        
        # === Step 2: Compute gradient d(sim)/d(latent_feat) ===
        # sim = (latent_feat @ concept_vectors.T * weight).sum(dim=0)  [sum over batch]
        # d(sim)/d(latent_feat) = concept_vectors * weight  [for each concept]
        # Shape: (B, M, D) where M = num_concepts
        M = concept_vectors.shape[0]
        
        # d(sim_bm)/d(latent_feat) = concept_vectors (broadcast over batch)
        # Since sim = (sim_bm * weight).sum(dim=0), we need d(sim)/d(sim_bm) = weight (for sum over batch, each contributes 1)
        # So d(sim)/d(latent_feat)[b, m, :] = weight[b, m] * concept_vectors[m, :]
        d_sim_d_latent = weight.unsqueeze(-1) * concept_vectors.unsqueeze(0)  # (B, M, D)
        
        # === Step 3: Backprop through normalize ===
        # latent_feat = x / ||x|| where x = normed_cls
        # d(normalize(x))/dx = (I - x_hat @ x_hat.T) / ||x||
        # where x_hat = x / ||x|| = latent_feat
        norm_val = normed_cls.norm(dim=-1, keepdim=True)  # (B, 1)
        # For each concept direction, project out the component along latent_feat
        # d_out = (d_in - latent_feat * (latent_feat @ d_in)) / norm_val
        # Here d_in = d_sim_d_latent: (B, M, D)
        dot_product = torch.einsum('bmd, bd -> bm', d_sim_d_latent, latent_feat)  # (B, M)
        d_sim_d_normed = (d_sim_d_latent - latent_feat.unsqueeze(1) * dot_product.unsqueeze(-1)) / norm_val.unsqueeze(1)  # (B, M, D)
        
        # === Step 4: Backprop through LayerNorm (self.norm) ===
        # This is an approximation - we assume the gradient flows mostly through the scaling
        # For LayerNorm: y = (x - mean) / std * gamma + beta
        # The gradient w.r.t. input is complex, but for our purposes we multiply by gamma
        d_sim_d_cls = d_sim_d_normed * self.norm.weight.unsqueeze(0).unsqueeze(0)  # (B, M, D)
        
        # === Step 5: Backprop through attention ===
        # The CLS token output comes from: attn_output = attn_weights @ V
        # After reshaping and projection: cls_contribution = proj(reshape(attn_output))[0, :]
        # 
        # For the gradient w.r.t. attn_weights, we need to trace through:
        # attn_output[:, :, 0, :] → ... → cls_feat
        #
        # The key insight: d(attn @ V)/d(attn) for the CLS position (query 0):
        # attn_output[b, h, 0, :] = sum_j attn_weights[b, h, 0, j] * V[b, h, j, :]
        # So d(attn_output[b,h,0,d])/d(attn_weights[b,h,0,j]) = V[b,h,j,d]
        
        # Backprop through attention projection
        # proj: x @ W^T where W is (D, D) for proj.weight
        proj_weight = attn_module.proj.weight  # (D, D)
        # d_sim_d_attn_proj = d_sim_d_cls @ proj_weight  # (B, M, D)
        d_sim_d_attn_proj = torch.einsum('bmd, de -> bme', d_sim_d_cls, proj_weight)  # (B, M, D)
        
        # Backprop through attention norm (if exists)
        if hasattr(attn_module, 'norm') and attn_module.norm is not None and hasattr(attn_module.norm, 'weight'):
            d_sim_d_attn_proj = d_sim_d_attn_proj * attn_module.norm.weight.unsqueeze(0).unsqueeze(0)
        
        # Reshape to match attention output format
        # attn_output was (B, num_heads, N, head_dim) reshaped to (B, N, D)
        # For CLS token (position 0): we need gradient for query position 0
        # d_sim_d_attn_proj: (B, M, D) is the gradient for CLS position
        # Reshape to (B, M, num_heads, head_dim)
        d_sim_d_attn_out = d_sim_d_attn_proj.view(B, M, num_heads, head_dim)  # (B, M, H, head_dim)
        
        # === Step 6: Compute gradient w.r.t. attention weights ===
        # attn_output[b, h, 0, :] = sum_j attn_weights[b, h, 0, j] * V[b, h, j, :]
        # d(loss)/d(attn_weights[b,h,0,j]) = sum_d d(loss)/d(attn_output[b,h,0,d]) * V[b,h,j,d]
        #                                 = d_sim_d_attn_out[b,:,h,:] @ V[b,h,j,:].T
        # 
        # For all key positions j:
        # grad[m, b, h, 0, j] = d_sim_d_attn_out[b, m, h, :] @ V[b, h, j, :].T
        #                    = sum_d d_sim_d_attn_out[b, m, h, d] * V[b, h, j, d]
        
        # v_values: (B, num_heads, N, head_dim)
        # d_sim_d_attn_out: (B, M, num_heads, head_dim)
        # Result shape: (M, B, num_heads, N) - gradient for query position 0 w.r.t. all key positions
        grad_cls_row = torch.einsum('bmhd, bhjd -> mbhj', d_sim_d_attn_out, v_values)  # (M, B, H, N)
        
        # The full gradient has shape (M, B, H, N, N) where we only computed row 0 (CLS query)
        # Since we only care about how attention to different tokens affects CLS,
        # we create a gradient tensor where only the CLS query row is non-zero
        grad = torch.zeros(M, B, num_heads, N, N, device=block_output.device, dtype=block_output.dtype)
        grad[:, :, :, 0, :] = grad_cls_row  # Only CLS query row has gradient
        
        # Clamp negative gradients
        grad = torch.clamp(grad, min=0.)
        self.grads.append(grad)
        
        # === Step 7: Compute attention map ===
        # Average over heads and query positions, exclude CLS token
        image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (M, B, N-1)
        expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
        self.maps.append(expl_map)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """Compute concept activation maps.
        
        If concept vectors were set before forward via set_concept_vectors(),
        maps are already computed during forward pass.
        
        If concept vectors were NOT set before forward, this method computes
        the maps using stored attention weights, V values and block outputs.
        
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
        
        # Otherwise, compute maps now using manual gradient computation
        if not self.block_outputs or not self.attn_weights or not self.v_values:
            raise ValueError("No block outputs, attention weights, or V values stored. Please run a forward pass first.")
        
        # Get attention module for projection weights
        attn_module = self.blocks[self._layer_indices[0]].attn
        
        self.power = power
        for i, (block_output, attn_weight, v_values) in enumerate(zip(self.block_outputs, self.attn_weights, self.v_values)):
            self._compute_manual_gradient_and_map(block_output, attn_weight, v_values, attn_module, concept_vectors)

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
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
