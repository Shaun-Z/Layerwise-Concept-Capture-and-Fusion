# src/lccf/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import types
import concurrent.futures

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper
from .functional import MultiheadAttention_forward

"""
Description:
 - The wrapper accepts an already constructed open_clip model (e.g., via the model, preprocess = open_clip.create_model_and_transforms("ViT-B-32"). Then pass in model (or model.visual) and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """

class OpenCLIPWrapper(CopyAttrWrapper):
    """
    An open-clip-specific derivative of CopyAttrWrapper that provides convenient methods for encode_text / encode_image.
    Important: This wrapper assumes that you are passing in the model of open_clip (typically the model returned by open_clip.create_model_and_transforms).
    passed in. If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        self.reset()

        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]

            block.attn.register_forward_hook(self._save_attn_output)    # (n, b, d)
            block.ln_2.register_forward_hook(self._aggregate_ln)        # (n, b, d)
            block.mlp.c_fc.register_forward_hook(self._aggregate_c_fc)
            block.mlp.c_proj.register_forward_hook(self._aggregate_c_proj)  # (n, b, d)
            block.register_forward_hook(self._finalize_hook)    # (n, b, d)

    def reset(self):
        """Reset the stored results and maps."""
        self.tmp = None
        self.result = []
        self.maps = []
        self.normed_clss = []

    def _save_attn_output(self, module, input, output):
        self.tmp = output[0].detach()
    def _aggregate_ln(self, module, input, output):
        std = input[0].std(dim=-1).detach()
        self.tmp /= rearrange(std, 'n b -> n b 1')
        self.tmp *= module.weight
    def _aggregate_c_fc(self, module, input, output):
        self.tmp @= module.weight.T
        a = math.sqrt(2.0 / math.pi)
        b = 0.044715
        x_ = self.tmp
        inner = a * (x_ + b * x_ * x_ * x_)
        t = torch.tanh(inner)
        sech2 = 1.0 - t * t  # = (1 - tanh^2)
        d_inner_dx = a * (1.0 + 3.0 * b * x_ * x_)
        grad = 0.5 * (1.0 + t) + 0.5 * x_ * (sech2 * d_inner_dx)
        self.tmp *= grad
    def _aggregate_c_proj(self, module, input, output):
        self.tmp @= module.weight.T
    def _finalize_hook(self, module, input, output):
        cls = output[:1, ...]
        std = cls.std(dim=-1).detach()
        self.tmp /= rearrange(std, '1 b -> 1 b 1')
        self.tmp *= self.visual.ln_post.weight  # [len, batch, dim]
        self.tmp @= self.visual.proj

        cls_encoded = self.visual.ln_post(cls) @ self.visual.proj   # [len, batch, dim]
        val = cls_encoded.norm(dim=-1, keepdim=True)    # [1, batch, 1]
        self.tmp /= val

        self.normed_clss.append(F.normalize(cls_encoded, dim=-1))
        
        # self.tmp = F.normalize(self.tmp, dim=-1)
        self.result.append(self.tmp)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        w = h = int(math.sqrt(self.result[0].shape[0]-1))  # Exclude CLS token
        for i, res in enumerate(self.result):
            prod = torch.einsum('n b d, m d -> n b m', res, concept_vectors)
            weight = torch.einsum('n b d, m d -> n b m', self.normed_clss[i], concept_vectors)
            # print(weight)
            prod = prod * weight
            map = torch.clamp(prod.mean(dim=0, keepdim=True) - prod, min=0.)    # negative gradient
            map = rearrange(map[1:,...], '(h w) b m -> h w b m', h=h, w=w)  # Exclude CLS token
            self.maps.append(map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        Returns:
            torch.Tensor: Aggregated attention maps of shape [H, W, batch_size, num_concepts]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m ->  h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')
        
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        maps = F.interpolate(maps, scale_factor=self.visual.patch_size[0], mode='bilinear')
        return maps
    
def attention_with_weights(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
):
    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x

    attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

    attn_output, attn_weight = self.attn(
        q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask, average_attn_weights=False
    )
    self.attn_weight = attn_weight  # Save attention weights for gradient computation
    return attn_output

class OpenCLIPGradWrapper(CopyAttrWrapper):
    """
    An open-clip-specific derivative of CopyAttrWrapper that provides convenient methods for encode_text / encode_image.
    Important: This wrapper assumes that you are passing in the model of open_clip (typically the model returned by open_clip.create_model_and_transforms).
    passed in. If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    This version captures gradients of attention maps.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            for name, param in block.named_parameters():
                param.requires_grad = True
            block.attention = types.MethodType(OpenCLIPGradWrapper.__attention_with_weights, block) # Override attention method: `need_weights=True`
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)   # Override MHA forward method: save attn maps

            block.attn.register_forward_hook(self._save_attn_hook)    # (n, b, d)
            block.register_forward_hook(self._save_block_hook)  # (n, b, d)
    
    def __attention_with_weights(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

        attn_output, attn_weight = self.attn(
            q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask, average_attn_weights=False
        )
        # self.attn_weight = attn_weight  # Save attention weights for gradient computation
        return attn_output

    def _save_attn_hook(self, module, input, output):   # attn_output: (n, b, d) ; attn_weights: (bsz*num_heads, n, n)
        self.attn_weights.append(output[1]) # gather attention weights (bsz*num_heads, n, n)
    def _save_block_hook(self, module, input, output):
        self.block_outputs.append(output)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0]-1))  # Exclude CLS token
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            self.visual.zero_grad()
            cls_feat = block_output[1, ...]    # (batch_size, 768)
            latent_feat = F.normalize(self.visual.ln_post(cls_feat) @ self.visual.proj, dim=-1) # (bsz, 512)

            sim_bm = torch.einsum('b d, m d ->b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
            weight = torch.abs(sim_bm.clone().detach()).pow(power)
            sim_bm *= weight  # (bsz, num_concepts)
            sim = sim_bm.sum(dim=0)  # (bsz, num_concepts) -> (num_concepts)
            self.sim_bms.append(weight)
            # Compute gradients of sim w.r.t. attn_weight
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)

            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, bsz*num_heads, n, n)
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 ->m b h n1 n2', h=self.num_heads)  # (num_concepts, bsz, num_heads, n, n)
            self.grads.append(grad)
            image_relevance = grad.mean(dim=2).mean(dim=-2)[...,1:]  # (num_concepts, bsz, n-1) Exclude CLS token
            expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
            self.maps.append(expl_map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        Returns:
            torch.Tensor: Aggregated attention maps of shape [H, W, batch_size, num_concepts]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m ->  h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min)
        maps = F.interpolate(maps, scale_factor=self.visual.patch_size[0], mode='bilinear')
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.grads = []
        self.maps = []
        self.sim_bms = []


class AOpenCLIPWrapper(CopyAttrWrapper):
    """
    Asynchronous OpenCLIP wrapper that computes gradients during the forward pass.
    
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

        # Get patch size and num_heads (same as GradWrapper)
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads
        self.embed_dim = self.visual.transformer.resblocks[0].attn.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.power = power
        self.async_compute = async_compute
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(layer_indices)) if async_compute else None
        self._futures = []
        self._layer_indices = layer_indices
        
        self.reset()

        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
            
            # Override attention method to capture weights (same as GradWrapper)
            block.attention = types.MethodType(AOpenCLIPWrapper._attention_with_weights, block)
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)

            # Register hooks
            block.attn.register_forward_hook(self._save_attn_hook)    # (n, b, d), (bsz*num_heads, n, n)
            block.register_forward_hook(self._finalize_and_compute_hook)    # (n, b, d)

    def _attention_with_weights(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        """Override attention method to request attention weights."""
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

        attn_output, attn_weight = self.attn(
            q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask, average_attn_weights=False
        )
        return attn_output

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
        # output is (attn_output, attn_weights) where attn_weights: (bsz*num_heads, n, n)
        self._current_attn_weight = output[1].detach()
        self.attn_weights.append(output[1].detach())
        # Retrieve V values stored by the modified forward
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
        
        OpenCLIP tensor layout: (N, B, D) where N is sequence length, B is batch size
        attn_weight shape: (bsz*num_heads, n, n)
        v_values shape: (bsz*num_heads, n, head_dim)
        """
        N, B, D = block_output.shape
        w = h = int(math.sqrt(N - 1))  # Exclude CLS token
        num_heads = self.num_heads
        head_dim = self.head_dim
        
        # === Step 1: Compute similarity and weight ===
        # block_output: (N, B, D) where N = 1 + H*W (CLS + patches)
        # For OpenCLIP, CLS token is at position 1 (index 1), not position 0
        cls_feat = block_output[1, ...]  # (B, D)
        
        # Project through ln_post and proj (OpenCLIP specific)
        normed_cls = self.visual.ln_post(cls_feat)  # (B, D)
        projected = normed_cls @ self.visual.proj  # (B, proj_dim)
        latent_feat = F.normalize(projected, dim=-1)  # (B, proj_dim)
        
        # Compute similarity with concept vectors
        M = concept_vectors.shape[0]  # num_concepts
        sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, M)
        weight = torch.abs(sim_bm.clone()).pow(self.power)
        self.sim_bms.append(weight)
        
        # === Step 2: Compute gradient d(sim)/d(latent_feat) ===
        # d(sim_bm)/d(latent_feat) = concept_vectors (broadcast over batch)
        # d(sim)/d(latent_feat)[b, m, :] = weight[b, m] * concept_vectors[m, :]
        d_sim_d_latent = weight.unsqueeze(-1) * concept_vectors.unsqueeze(0)  # (B, M, proj_dim)
        
        # === Step 3: Backprop through normalize ===
        norm_val = projected.norm(dim=-1, keepdim=True)  # (B, 1)
        dot_product = torch.einsum('bmd, bd -> bm', d_sim_d_latent, latent_feat)  # (B, M)
        d_sim_d_proj = (d_sim_d_latent - latent_feat.unsqueeze(1) * dot_product.unsqueeze(-1)) / norm_val.unsqueeze(1)  # (B, M, proj_dim)
        
        # === Step 4: Backprop through visual.proj ===
        # projected = normed_cls @ proj, so d/d(normed_cls) = d_proj @ proj.T
        d_sim_d_normed = torch.einsum('bmd, de -> bme', d_sim_d_proj, self.visual.proj.T)  # (B, M, D)
        
        # === Step 5: Backprop through ln_post (approximate) ===
        # LayerNorm gradient - multiply by weight
        if hasattr(self.visual.ln_post, 'weight') and self.visual.ln_post.weight is not None:
            d_sim_d_cls = d_sim_d_normed * self.visual.ln_post.weight.unsqueeze(0).unsqueeze(0)  # (B, M, D)
        else:
            d_sim_d_cls = d_sim_d_normed
        
        # === Step 6: Backprop through attention output projection ===
        # The CLS token output (position 1) comes from attention
        # attn_output for CLS = out_proj(reshape(attn_weights @ V))
        # We need d/d(attn_weights) for CLS position (query 1)
        
        # out_proj: (embed_dim, embed_dim)
        out_proj_weight = attn_module.out_proj.weight  # (D, D)
        d_sim_d_attn_proj = torch.einsum('bmd, de -> bme', d_sim_d_cls, out_proj_weight)  # (B, M, D)
        
        # Reshape to match attention output format
        # attn_output was (bsz*num_heads, tgt_len, head_dim) then reshaped
        # For CLS query position 1: we need gradient for query position 1
        d_sim_d_attn_out = d_sim_d_attn_proj.view(B, M, num_heads, head_dim)  # (B, M, H, head_dim)
        
        # === Step 7: Compute gradient w.r.t. attention weights ===
        # v_values: (bsz*num_heads, src_len, head_dim)
        # Reshape to (B, num_heads, src_len, head_dim)
        v_reshaped = v_values.view(B, num_heads, N, head_dim)  # (B, H, N, head_dim)
        
        # For CLS query (position 1):
        # attn_output[b, h, 1, :] = sum_j attn_weights[b*H+h, 1, j] * V[b*H+h, j, :]
        # grad[m, b, h, j] = d_sim_d_attn_out[b, m, h, :] @ V[b, h, j, :].T
        grad_cls_row = torch.einsum('bmhd, bhjd -> mbhj', d_sim_d_attn_out, v_reshaped)  # (M, B, H, N)
        
        # Create full gradient tensor with only CLS query row (position 1) non-zero
        # Reshape to (M, B*H, N, N) to match attn_weight shape
        grad = torch.zeros(M, B * num_heads, N, N, device=block_output.device, dtype=block_output.dtype)
        grad_cls_row_flat = rearrange(grad_cls_row, 'm b h j -> m (b h) j')  # (M, B*H, N)
        grad[:, :, 1, :] = grad_cls_row_flat  # CLS is at position 1 in OpenCLIP
        
        # Clamp negative gradients
        grad = torch.clamp(grad, min=0.)
        
        # Reshape to (M, B, H, N, N) for consistency
        grad_reshaped = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=num_heads)
        self.grads.append(grad_reshaped)
        
        # === Step 8: Compute attention map ===
        # Average over heads and query positions, exclude CLS token (position 0)
        # Note: In OpenCLIP, position 0 is SOT token, position 1 is CLS, positions 2+ are patches
        image_relevance = grad_reshaped.mean(dim=2).mean(dim=-2)[..., 2:]  # (M, B, N-2) Exclude SOT and CLS
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
        attn_module = self.visual.transformer.resblocks[self._layer_indices[0]].attn
        
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
        maps = F.interpolate(maps, scale_factor=self.visual.patch_size[0], mode='bilinear')
        return maps

    def close(self):
        """Clean up resources, including the ThreadPoolExecutor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()