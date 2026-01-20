# src/lccf/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import types

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper
from .functional import MultiheadAttention_forward, Pseudo_MultiheadAttention_forward

"""
Description:
 - The wrapper accepts an already constructed open_clip model (e.g., via the model, preprocess = open_clip.create_model_and_transforms("ViT-B-32"). Then pass in model (or model.visual) and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """

class OpenCLIPFastWrapper(CopyAttrWrapper):
    """
    An open-clip-specific derivative of CopyAttrWrapper that provides convenient methods for encode_text / encode_image.
    Important: This wrapper assumes that you are passing in the model of open_clip (typically the model returned by open_clip.create_model_and_transforms).
    passed in. If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads

        self.pseudo_handles = []
        self.normal_handles = []
        
        self.reset()

        # Register hooks to the specified layers to capture attention outputs
        self.switch_to_normal_mode()
        # for idx in layer_indices:
        #     block = self.visual.transformer.resblocks[idx]
        #     self.normal_handles.append(block.register_forward_hook(self._save_block_input))  # (n, b, d)

    def reset(self):
        """Reset the stored results and maps."""
        self.block_ins = []
        self.attn_weights = None
        self.sim_bms = []
        self.grads = []
        self.maps = []

    def _save_block_input(self, module, input, output):
        self.block_ins.append(input[0])  # (n, b, d)

    def switch_to_pseudo_mode(self):
        """Switch the MultiheadAttention modules to pseudo mode that returns attention weights."""
        for handle in self.normal_handles:
            handle.remove()
        self.normal_handles = []
        for idx in self._requested_hook_indices:
            block = self.visual.transformer.resblocks[idx]
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attention = types.MethodType(OpenCLIPFastWrapper.__attention_with_weights, block) # Override attention method: `need_weights=True`
            block.attn.forward = types.MethodType(Pseudo_MultiheadAttention_forward, block.attn)   # Override MHA forward method: save attn maps
            for name, param in block.named_parameters():
                param.requires_grad = True
            self.pseudo_handles.append(block.attn.register_forward_hook(self._save_attn_hook))
            self.pseudo_handles.append(block.ln_2.register_forward_pre_hook(self._ln_2_pre_hook))
            self.pseudo_handles.append(block.register_forward_hook(self._res_block_post_hook))

    def switch_to_normal_mode(self):
        """Switch back to the normal MultiheadAttention modules."""
        for handle in self.pseudo_handles:
            handle.remove()
        self.pseudo_handles = []
        for idx in self._requested_hook_indices:
            block = self.visual.transformer.resblocks[idx]
            self.normal_handles.append(block.register_forward_hook(self._save_block_input))  # (n, b, d)
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)   # Override MHA forward method: save attn maps
            for name, param in block.named_parameters():
                param.requires_grad = False

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
        )   #  ((1, bsz, embed_dim), (bsz * num_heads, 1, n))
        return attn_output

    def _save_attn_hook(self, module, input, output):   # attn_output: (1, b, d); attn_weights: (bsz*num_heads, 1, n)
        self.attn_weight = output[1] # Temporarily store attention weights (bsz*num_heads, 1, n)
    def _ln_2_pre_hook(self, module, inputs):
        x, *rest = inputs
        return (x[:1, ...], *rest)  # (1, b, d)
    def _res_block_post_hook(self, module, input, output):
        return output[0, ...]  # (b, d)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        w = h = int(math.sqrt(self.block_ins[0].shape[0]-1))  # Exclude CLS token
        self.switch_to_pseudo_mode()
        for idx, data_in in zip(self._requested_hook_indices, self.block_ins):
            self.visual.zero_grad()
            block = self.visual.transformer.resblocks[idx]
            #data_in (n, b, d)
            cls_feat = block(data_in)  # (bsz, d)
            latent_feat = F.normalize(self.visual.ln_post(cls_feat) @ self.visual.proj, dim=-1) # (bsz, 512)

            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d ->b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm *= weight  # (bsz, num_concepts)
            sim = sim_bm.sum(dim=0)  # (bsz, num_concepts) -> (num_concepts)
            self.sim_bms.append(weight)

            # Compute gradients of sim w.r.t. attn_weight
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            grad = torch.autograd.grad(outputs=sim, inputs=self.attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, bsz*num_heads, 1, n)
            
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 ->m b h n1 n2', h=self.num_heads)  # (num_concepts, bsz, num_heads, 1, n)
            self.grads.append(grad)
            image_relevance = grad.mean(dim=2).squeeze(-2)[...,1:]  # (num_concepts, bsz, n-1) Exclude CLS token
            expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
            self.maps.append(expl_map)

        self.switch_to_normal_mode()

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

class OpenCLIPWrapper(CopyAttrWrapper):
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
            block.attention = types.MethodType(OpenCLIPWrapper.__attention_with_weights, block) # Override attention method: `need_weights=True`
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

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, tk_idx: int = 0, power: int = 2, weighted_attn: bool = False):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0]-1))  # Exclude CLS token
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            self.visual.zero_grad()
            feat = block_output[tk_idx, ...]    # (batch_size, 768)
            latent_feat = F.normalize(self.visual.ln_post(feat) @ self.visual.proj, dim=-1) # (bsz, 512)

            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d ->b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
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
            if weighted_attn:
                grad = torch.einsum('m X i j, X j k -> m X i k', grad, attn_weight.transpose(-2, -1))  # Matrix multiplication: grad @ attn_weight^T
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


class OpenCLIPFCVWrapper(CopyAttrWrapper):
    """
    An OpenCLIP-specific wrapper that propagates gradients from the last layer backward through
    ALL transformer blocks, using ALL tokens' gradients as concept vectors for each layer.
    
    Key behavior:
    - All blocks do backpropagation, layer_indices is only for choosing visualized layers
    - For the deepest layer, concept_vectors shape is (M, D) in latent space (512)
    - For mid layers, concept_vectors shape is (N, B, M, D) - full token gradients in hidden space (768)
    - Special handling for OpenCLIP: Before the loop, project to latent space:
      feat = block_output  # (N, B, 768)
      latent_feat = F.normalize(self.visual.ln_post(feat) @ self.visual.proj, dim=-1)  # (N, B, 512)
      sim_bm = torch.einsum('n b d, m d -> n b m', latent_feat, concept_vectors).sum(dim=0)  # (B, M)
    - Then compute the output tokens' grad (N, B, D) from sim_bm, which serves as first concept vector
    
    This wrapper processes all layers in reverse order (from deepest to shallowest).
    
    Note: Unlike FastWrapper/CVWrapper, this wrapper does not use pseudo/normal mode switching
    since it needs full attention (not CLS-only) for computing gradients of all tokens.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # Get the number of blocks before calling super().__init__
        num_blocks = len(model.visual.transformer.resblocks)
        
        # Store layer_indices for aggregation, but use ALL layers for hook registration
        all_layer_indices = list(range(num_blocks))
        
        # The copying behavior is done by the parent class CopyAttrWrapper
        # Pass ALL layer indices for hook registration
        super().__init__(model, layer_indices=all_layer_indices, include_private=include_private)

        # Store patch info for later use
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads
        self._num_blocks = num_blocks
        
        # Store the user's requested layer_indices for aggregation only
        if layer_indices is None:
            self._aggregate_layer_indices = all_layer_indices
        else:
            self._aggregate_layer_indices = list(layer_indices)

        self.reset()
        
        # Set up hooks and attention forward for all layers
        for idx in self._requested_hook_indices:
            block = self.visual.transformer.resblocks[idx]
            for name, param in block.named_parameters():
                param.requires_grad = True
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attention = types.MethodType(OpenCLIPFCVWrapper.__attention_with_weights, block)
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)
            block.attn.register_forward_hook(self._save_attn_hook)
            block.register_forward_hook(self._save_block_input)

    def reset(self):
        """Reset the stored results and maps."""
        self.block_ins = []
        self.attn_weight = None
        self.attn_grads = []     # Store attention weight gradients
        self.token_grads = []    # Store input tokens' gradients (N, B, M, D)
        self.maps = []
        self.sim_bms = []        # Store similarity weights for visualization

    def _save_block_input(self, module, input, output):
        self.block_ins.append(input[0])  # (N, B, D)

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
        return attn_output

    def _save_attn_hook(self, module, input, output):
        # attn_weights: (B*num_heads, N, N) from MultiheadAttention_forward
        self.attn_weight = output[1]

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 1):
        """Compute gradient-based concept activation maps using full token backpropagation.
        
        For OpenCLIP:
        - First, compute the initial gradient from the deepest block output projected to latent space
        - Then propagate through all layers using full token gradients
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, latent_dim] - normalized concept vectors.
                For OpenCLIP, this should be text embeddings in the shared latent space (e.g., 512-dim).
            power (int): Power for similarity scaling. Default: 1
        
        Note:
            Gradients are computed for ALL layers (0 to num_blocks-1).
            The layer_indices parameter in __init__ only affects aggregation.
        """
        w = h = int(math.sqrt(self.block_ins[0].shape[0] - 1))  # Exclude CLS token
        
        # Process ALL layers in reverse order (from deepest to shallowest)
        sorted_indices = sorted(enumerate(self._requested_hook_indices), key=lambda x: x[1], reverse=True)
        
        # We'll collect results in reverse order (deepest to shallowest), then reverse at the end
        attn_grads_reversed = []
        token_grads_reversed = []
        maps_reversed = []
        sim_bms_reversed = []
        
        # Create a set for fast lookup of which layers to store maps/sim_bms for
        aggregate_layer_set = set(self._aggregate_layer_indices)
        
        # For the deepest layer, use the text embeddings directly
        # concept_vectors shape: (M, latent_dim=512)
        current_concept_vectors = concept_vectors  # (M, D=512) for deepest layer
        is_deepest = True
        
        for enum_idx, layer_idx in sorted_indices:
            self.visual.zero_grad()
            block = self.visual.transformer.resblocks[layer_idx]
            data_in = self.block_ins[enum_idx]  # (N, B, D=768)
            
            # Get input tokens - requires grad for computing token gradients
            input_tokens = data_in.clone()
            input_tokens.requires_grad_(True)
            
            # Run through the block: block(x) returns (N, B, D)
            block_output = block(input_tokens)  # (N, B, D=768)
            
            if is_deepest:
                # For the deepest layer: project ALL tokens to latent space
                # block_output: (N, B, D=768)
                latent_feat = self.visual.ln_post(block_output) @ self.visual.proj  # (N, B, 512)
                latent_feat_normalized = F.normalize(latent_feat, dim=-1)  # (N, B, 512)
                
                # Compute similarity: sum over tokens
                # current_concept_vectors: (M, 512)
                sim_bm = torch.einsum('n b d, m d -> n b m', latent_feat_normalized, current_concept_vectors)  # (N, B, M)
                sim_bm = sim_bm.sum(dim=0)  # (B, M)
                is_deepest = False
            else:
                # For other layers: concept_vectors is (N, B, M, D) in hidden space
                # block_output: (N, B, D=768)
                latent_feat = F.normalize(block_output, dim=-1)  # (N, B, D)
                sim_bm = torch.einsum('n b d, n b m d -> b m', latent_feat, current_concept_vectors)  # (B, M)
            
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm = sim_bm * weight  # (B, M)
            sim = sim_bm.sum(dim=0)  # (B, M) -> (M,)
            
            # Store similarity weight only for layers in layer_indices
            if layer_idx in aggregate_layer_set:
                sim_bms_reversed.append(weight)  # (B, M)
            
            # Compute gradients of sim w.r.t. attn_weight and input_tokens for each concept
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            grads = torch.autograd.grad(
                outputs=sim,
                inputs=[self.attn_weight, input_tokens],
                grad_outputs=eye,
                retain_graph=True,
                create_graph=False,
                is_grads_batched=True,
                allow_unused=True
            )
            
            attn_grad = grads[0]  # (M, B*num_heads, N, N)
            token_grad = grads[1]  # (M, N, B, D) - gradient of all input tokens for each concept
            
            # Store attention gradient
            if attn_grad is not None:
                attn_grad = torch.clamp(attn_grad, min=0.)
                attn_grad = rearrange(attn_grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)
                attn_grads_reversed.append(attn_grad)
                
                # Store explanation map only for layers in layer_indices
                if layer_idx in aggregate_layer_set:
                    # Compute explanation map: average over heads and query positions, exclude CLS
                    image_relevance = attn_grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (M, B, N-1)
                    expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
                    maps_reversed.append(expl_map)
            
            # Store token gradients and use as concept_vectors for the previous (shallower) layer
            if token_grad is not None:
                # token_grad shape: (M, N, B, D), transpose to (N, B, M, D)
                token_grad_transposed = token_grad.permute(1, 2, 0, 3)  # (N, B, M, D)
                token_grads_reversed.append(token_grad_transposed)
                # Use normalized token gradients as concept vectors for the next layer
                current_concept_vectors = F.normalize(token_grad_transposed, dim=-1)  # (N, B, M, D)
        
        # Reverse to get forward order (shallowest to deepest)
        self.attn_grads = attn_grads_reversed[::-1]
        self.token_grads = token_grads_reversed[::-1]
        self.maps = maps_reversed[::-1]
        self.sim_bms = sim_bms_reversed[::-1]

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across the layers specified in layer_indices.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and dot_concept_vectors first.")
        
        maps = torch.stack(self.maps, dim=0)  # (num_selected_layers, h, w, B, M)
        maps = torch.einsum('l h w b m -> h w b m', maps)  # sum across layers
        maps = rearrange(maps, 'h w b m -> b m h w')  # (B, M, h, w)

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        maps = F.interpolate(maps, scale_factor=self.patch_size, mode='bilinear')
        return maps


class OpenCLIPCVWrapper(CopyAttrWrapper):
    """
    An OpenCLIP-specific wrapper that propagates gradients from the last layer backward through
    ALL transformer blocks, using CLS token gradients as concept vectors for each layer.
    
    Key behavior:
    - Backpropagation starts from the last layer's output
    - For mid layers, the gradient of the CLS comes from layer i+1
    - During each block's backpropagation, stores:
      1) attention_weight's grad
      2) input CLS token's grad (serves as concept vector for the previous block)
    - The start of each block's backpropagation is concept_vector * CLS
    - Gradients are computed for ALL layers (0 to num_blocks-1)
    - layer_indices is only used for aggregation (selecting which layers' maps to sum)
    
    Note on concept_vectors:
    - For OpenCLIP, the initial concept_vectors cannot be obtained from a classifier head
    - The model's final encoding goes through visual.ln_post and visual.proj to be aligned with text
    - Concept vectors should be text embeddings normalized to match the latent space
    
    This wrapper processes all layers in reverse order (from deepest to shallowest).
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # Get the number of blocks before calling super().__init__
        num_blocks = len(model.visual.transformer.resblocks)
        
        # Store layer_indices for aggregation, but use ALL layers for hook registration
        all_layer_indices = list(range(num_blocks))
        
        # The copying behavior is done by the parent class CopyAttrWrapper
        # Pass ALL layer indices for hook registration
        super().__init__(model, layer_indices=all_layer_indices, include_private=include_private)

        # Store patch info for later use
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads
        self._num_blocks = num_blocks

        self.pseudo_handles = []
        self.normal_handles = []
        
        # Store the user's requested layer_indices for aggregation only
        if layer_indices is None:
            self._aggregate_layer_indices = all_layer_indices
        else:
            self._aggregate_layer_indices = list(layer_indices)

        self.reset()
        
        # Register hooks in normal mode to capture block inputs
        self.switch_to_normal_mode()

    def reset(self):
        """Reset the stored results and maps."""
        self.block_ins = []
        self.attn_weight = None
        self.attn_grads = []  # Store attention weight gradients
        self.cls_grads = []   # Store input CLS token gradients
        self.maps = []
        self.sim_bms = []     # Store similarity weights for visualization

    def _save_block_input(self, module, input, output):
        self.block_ins.append(input[0])  # (n, b, d)

    def switch_to_pseudo_mode(self):
        """Switch the MultiheadAttention modules to pseudo mode that returns attention weights."""
        for handle in self.normal_handles:
            handle.remove()
        self.normal_handles = []
        for idx in self._requested_hook_indices:
            block = self.visual.transformer.resblocks[idx]
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attention = types.MethodType(OpenCLIPCVWrapper.__attention_with_weights, block)
            block.attn.forward = types.MethodType(Pseudo_MultiheadAttention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = True
            self.pseudo_handles.append(block.attn.register_forward_hook(self._save_attn_hook))
            self.pseudo_handles.append(block.ln_2.register_forward_pre_hook(self._ln_2_pre_hook))
            self.pseudo_handles.append(block.register_forward_hook(self._res_block_post_hook))

    def switch_to_normal_mode(self):
        """Switch back to the normal MultiheadAttention modules."""
        for handle in self.pseudo_handles:
            handle.remove()
        self.pseudo_handles = []
        for idx in self._requested_hook_indices:
            block = self.visual.transformer.resblocks[idx]
            self.normal_handles.append(block.register_forward_hook(self._save_block_input))
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = False

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
        )   #  ((1, bsz, embed_dim), (bsz * num_heads, 1, n))
        return attn_output

    def _save_attn_hook(self, module, input, output):
        self.attn_weight = output[1]  # (bsz*num_heads, 1, n)
    def _ln_2_pre_hook(self, module, inputs):
        x, *rest = inputs
        return (x[:1, ...], *rest)  # (1, b, d)
    def _res_block_post_hook(self, module, input, output):
        return output[0, ...]  # (b, d)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 1):
        """Compute gradient-based concept activation maps using pseudo mode.
        
        Backpropagation starts from the last layer using the provided concept_vectors
        (typically from text embeddings), then propagates through ALL layers.
        Each layer i uses the CLS gradient from layer i+1 as its concept vector.
        
        For OpenCLIP:
        - Last layer: Projects CLS to latent space via ln_post @ visual.proj, 
          dots with text embeddings, backprop to get attn_weight grad and input CLS grad
        - Other layers: Dots output CLS directly with the gradient from the next layer (in hidden space),
          backprop to get attn_weight grad and input CLS grad
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors.
                For OpenCLIP, this should be text embeddings in the shared latent space (e.g., 512-dim).
                The dimension should match the output of visual.proj (typically 512).
            power (int): Power for similarity scaling. Default: 1
        
        Note:
            Gradients are computed for ALL layers (0 to num_blocks-1).
            The layer_indices parameter in __init__ only affects aggregation.
        """
        w = h = int(math.sqrt(self.block_ins[0].shape[0] - 1))  # Exclude CLS token
        self.switch_to_pseudo_mode()
        
        # Process ALL layers in reverse order (from deepest to shallowest)
        sorted_indices = sorted(enumerate(self._requested_hook_indices), key=lambda x: x[1], reverse=True)
        deepest_layer_idx = sorted_indices[0][1]
        
        # We'll collect results in reverse order (deepest to shallowest), then reverse at the end
        attn_grads_reversed = []
        cls_grads_reversed = []
        maps_reversed = []
        sim_bms_reversed = []
        
        # Create a set for fast lookup of which layers to store maps/sim_bms for
        aggregate_layer_set = set(self._aggregate_layer_indices)
        
        # Initially, use the text embeddings as concept vectors (for the deepest layer)
        current_concept_vectors = concept_vectors  # (M, latent_dim=512)
        
        for enum_idx, layer_idx in sorted_indices:
            self.visual.zero_grad()
            block = self.visual.transformer.resblocks[layer_idx]
            data_in = self.block_ins[enum_idx]  # (N, B, D) in hidden space (768)
            
            # Get CLS token from input - requires grad for computing cls_grad
            input_cls = data_in[0, :, :].clone()  # (B, D) in hidden space (768)
            input_cls.requires_grad_(True)
            
            # Create input with the grad-enabled CLS token
            data_in_with_grad = data_in.clone()
            data_in_with_grad[0, :, :] = input_cls
            
            # Run through the block - returns (B, D) in pseudo mode
            cls_feat = block(data_in_with_grad)  # (B, D) in hidden space (768)
            cls_feat.retain_grad()
            
            if layer_idx == deepest_layer_idx:
                # For the deepest layer: project CLS to latent space, dot with text embeddings
                latent_feat = self.visual.ln_post(cls_feat) @ self.visual.proj  # (B, latent_dim=512)
                latent_feat_normalized = F.normalize(latent_feat, dim=-1)  # (B, latent_dim)
                
                # Compute similarity with concept vectors (in latent space)
                sim_bm = torch.einsum('b d, m d -> b m', latent_feat_normalized, current_concept_vectors)  # (B, M)
            else:
                # For other layers: dot output CLS directly with the gradient from the deeper layer
                sim_bm = torch.einsum('b d, m d -> b m', cls_feat, current_concept_vectors)  # (B, M)
            
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm = sim_bm * weight  # (B, M)
            sim = sim_bm.sum(dim=0)  # (B, M) -> (M,)
            
            # Store similarity weight only for layers in layer_indices
            if layer_idx in aggregate_layer_set:
                sim_bms_reversed.append(weight)  # (B, M)
            
            # Compute gradients of sim w.r.t. attn_weight and input_cls for each concept
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            grads = torch.autograd.grad(
                outputs=sim,
                inputs=[self.attn_weight, input_cls],
                grad_outputs=eye,
                retain_graph=True,
                create_graph=False,
                is_grads_batched=True,
                allow_unused=True
            )
            
            attn_grad = grads[0]  # (M, B*num_heads, 1, N)
            cls_grad = grads[1]   # (M, B, D) - gradient of input CLS (in hidden space)
            
            # Store attention gradient
            if attn_grad is not None:
                attn_grad = torch.clamp(attn_grad, min=0.)
                attn_grad = rearrange(attn_grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)
                attn_grads_reversed.append(attn_grad)
                
                # Store explanation map only for layers in layer_indices
                if layer_idx in aggregate_layer_set:
                    image_relevance = attn_grad.mean(dim=2).squeeze(-2)[..., 1:]  # (M, B, N-1)
                    expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
                    maps_reversed.append(expl_map)
            
            # Store CLS gradient and use it as concept_vectors for the next (shallower) layer
            if cls_grad is not None:
                cls_grads_reversed.append(cls_grad)
                # Use the CLS gradient as concept vectors for the next layer (in hidden space)
                current_concept_vectors = F.normalize(cls_grad.mean(dim=1), dim=-1)  # (M, D=768)
        
        # Reverse to get forward order (shallowest to deepest)
        self.attn_grads = attn_grads_reversed[::-1]
        self.cls_grads = cls_grads_reversed[::-1]
        self.maps = maps_reversed[::-1]
        self.sim_bms = sim_bms_reversed[::-1]
        
        self.switch_to_normal_mode()

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across the layers specified in layer_indices.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and dot_concept_vectors first.")
        
        maps = torch.stack(self.maps, dim=0)  # (num_selected_layers, h, w, B, M)
        maps = torch.einsum('l h w b m -> h w b m', maps)  # sum across layers
        maps = rearrange(maps, 'h w b m -> b m h w')  # (B, M, h, w)

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min)
        maps = F.interpolate(maps, scale_factor=self.patch_size, mode='bilinear')
        return maps