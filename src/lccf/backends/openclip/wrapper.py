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
    
    This version computes gradients during the forward pass by:
    1. Setting concept vectors before forward via set_concept_vectors()
    2. Computing gradients w.r.t. attention weights as each block finishes
    3. Optionally using async computation for efficiency
    
    The results match those of OpenCLIPGradWrapper.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False, 
                 power: int = 2, async_compute: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size and num_heads (same as GradWrapper)
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads
        self.power = power
        self.async_compute = async_compute
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if async_compute else None
        self._futures = []
        
        self.reset()

        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
            
            # Enable gradients for the block parameters
            for name, param in block.named_parameters():
                param.requires_grad = True
            
            # Override attention method to capture weights (same as GradWrapper)
            block.attention = types.MethodType(OpenCLIPWrapper._attention_with_weights, block)
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

    def _save_attn_hook(self, module, input, output):
        """Save attention weights from the attention module."""
        # output is (attn_output, attn_weights) where attn_weights: (bsz*num_heads, n, n)
        self._current_attn_weight = output[1]
        self.attn_weights.append(output[1])

    def _finalize_and_compute_hook(self, module, input, output):
        """Finalize hook that computes gradient and map immediately after block forward."""
        # Save block output
        self.block_outputs.append(output)
        
        # If concept vectors are set, compute gradient and map immediately
        if self._concept_vectors is not None:
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
        
        This method matches the computation in OpenCLIPGradWrapper.dot_concept_vectors().
        """
        w = h = int(math.sqrt(block_output.shape[0] - 1))  # Exclude CLS token
        
        # Zero gradients
        self.visual.zero_grad()
        
        # block_output: (N, B, D) where N = 1 + H*W (CLS + patches)
        # For OpenCLIP, CLS token is at position 1 (index 1), not position 0
        # Looking at GradWrapper: cls_feat = block_output[1, ...]
        cls_feat = block_output[1, ...]  # (batch_size, 768)
        latent_feat = F.normalize(self.visual.ln_post(cls_feat) @ self.visual.proj, dim=-1)  # (bsz, 512)
        
        # Compute similarity with concept vectors
        sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
        weight = torch.abs(sim_bm.clone().detach()).pow(self.power)
        sim_bm = sim_bm * weight  # (bsz, num_concepts)
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
        grad = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (num_concepts, bsz, num_heads, n, n)
        self.grads.append(grad)
        
        # Compute image relevance map
        image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (num_concepts, bsz, n-1) Exclude CLS token
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
        
        w = h = int(math.sqrt(self.block_outputs[0].shape[0] - 1))  # Exclude CLS token
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            self.visual.zero_grad()
            cls_feat = block_output[1, ...]  # (batch_size, 768)
            latent_feat = F.normalize(self.visual.ln_post(cls_feat) @ self.visual.proj, dim=-1)  # (bsz, 512)
            
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
            weight = torch.abs(sim_bm.clone().detach()).pow(power)
            sim_bm = sim_bm * weight  # (bsz, num_concepts)
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
            grad = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (num_concepts, bsz, num_heads, n, n)
            self.grads.append(grad)
            image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (num_concepts, bsz, n-1) Exclude CLS token
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
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        maps = F.interpolate(maps, scale_factor=self.visual.patch_size[0], mode='bilinear')
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.grads = []
        self.maps = []
        self.sim_bms = []