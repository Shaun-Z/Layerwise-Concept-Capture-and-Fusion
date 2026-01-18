# src/lccf/backends/timm/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import types

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper
from .functional import Attention_forward, Pseudo_Attention_forward

"""
Description:
 - The wrapper accepts an already constructed timm model (e.g., via timm.create_model). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """


class TimmFastWrapper(CopyAttrWrapper):
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
    
    This wrapper uses pseudo mode for gradient computation, similar to OpenCLIPWrapper.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        # Store patch info for later use
        self._patch_size = model.patch_embed.proj.kernel_size[0]
        self._embed_dim = model.embed_dim
        self.num_heads = model.blocks[0].attn.num_heads
        
        self.reset()
        
        # Register hooks in normal mode to capture block inputs
        self.switch_to_normal_mode()

    def reset(self):
        """Reset the stored results and maps."""
        self.block_ins = []
        self.attn_weight = None
        self.sim_bms = []
        self.grads = []
        self.maps = []
        self.pseudo_handles = []
        self.normal_handles = []
        # Keep these for backward compatibility
        self.result = []
        self.normed_clss = []

    def _save_block_input(self, module, input, output):
        # input is a tuple, input[0] is the actual input tensor
        # timm block input: (B, N, D)
        # Transpose to (N, B, D) to match OpenCLIP format
        self.block_ins.append(input[0].transpose(0, 1))  # (N, B, D)

    def switch_to_pseudo_mode(self):
        """Switch the Attention modules to pseudo mode that returns attention weights."""
        for handle in self.normal_handles:
            handle.remove()
        self.normal_handles = []
        for idx in self._requested_hook_indices:
            block = self.blocks[idx]
            # Override attention forward method to pseudo mode (only attend to CLS token)
            block.attn.forward = types.MethodType(Pseudo_Attention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = True
            self.pseudo_handles.append(block.attn.register_forward_hook(self._save_attn_hook))

    def switch_to_normal_mode(self):
        """Switch back to the normal Attention modules."""
        for handle in self.pseudo_handles:
            handle.remove()
        self.pseudo_handles = []
        for idx in self._requested_hook_indices:
            block = self.blocks[idx]
            self.normal_handles.append(block.register_forward_hook(self._save_block_input))
            # Restore attention forward method
            block.attn.forward = types.MethodType(Attention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = False

    def _save_attn_hook(self, module, input, output):
        # attn_weights: (B*num_heads, 1, N) from Pseudo_Attention_forward
        self.attn_weight = module._attn_weights

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 2):
        """Compute gradient-based concept activation maps using pseudo mode.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
            power (int): Power for similarity scaling. Default: 2
        """
        w = h = int(math.sqrt(self.block_ins[0].shape[0] - 1))  # Exclude CLS token
        self.switch_to_pseudo_mode()
        for idx, data_in in zip(self._requested_hook_indices, self.block_ins):
            self.zero_grad()
            block = self.blocks[idx]
            # data_in: (N, B, D), transpose to (B, N, D) for timm
            x = data_in.transpose(0, 1)
            
            # Run through the block
            # timm block: x = x + attn(norm1(x)), x = x + mlp(norm2(x))
            # We need to run the block manually to get the pseudo attention output
            x_norm = block.norm1(x)
            attn_out = block.attn(x_norm)  # Returns (1, B, D) in pseudo mode
            # attn_out is (1, B, D), transpose back and add residual for CLS only
            cls_out = attn_out.squeeze(0) + x[:, 0, :]  # (B, D)
            # Apply norm2 and mlp only to CLS token
            cls_norm2 = block.norm2(cls_out)
            cls_mlp = block.mlp(cls_norm2)
            cls_feat = cls_out + cls_mlp  # (B, D)
            
            # Apply final LayerNorm
            latent_feat = F.normalize(self.norm(cls_feat), dim=-1)  # (B, D)
            
            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (B, num_concepts)
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm *= weight  # (B, num_concepts)
            sim = sim_bm.sum(dim=0)  # (B, num_concepts) -> (num_concepts)
            self.sim_bms.append(weight)
            
            # Compute gradients of sim w.r.t. attn_weight
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            grad = torch.autograd.grad(outputs=sim, inputs=self.attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, B*num_heads, 1, N)
            
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (num_concepts, B, num_heads, 1, N)
            self.grads.append(grad)
            image_relevance = grad.mean(dim=2).squeeze(-2)[..., 1:]  # (num_concepts, B, N-1) Exclude CLS token
            expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
            self.maps.append(expl_map)
        
        self.switch_to_normal_mode()

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


class TimmWrapper(CopyAttrWrapper):
    """
    A timm-specific derivative of CopyAttrWrapper that uses autograd to capture gradients of attention maps.
    This mirrors the functionality of OpenCLIPFastWrapper for timm models.
    
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
            
            block.attn.register_forward_hook(self._save_attn_hook)  # (bsz, N, D), (bsz*num_heads, N, N)
            block.register_forward_hook(self._save_block_hook)  # (bsz, N, D)

    def _save_attn_hook(self, module, input, output):
        # Retrieve attention weights stored by the modified forward
        # attn_weights: (bsz, num_heads, N, N) - stored in original shape
        if hasattr(module, '_attn_weights'):
            self.attn_weights.append(module._attn_weights)

    def _save_block_hook(self, module, input, output):
        # Block output: (bsz, N, D)
        self.block_outputs.append(output)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, tk_idx: int = 0, power: int = 1, weighted_attn: bool = False):
        """Compute gradient-based concept activation maps.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
            power (int): Power for similarity scaling. Default: 2
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[1] - 1))  # Exclude CLS token, layout is (bsz, N, D)
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            # Zero gradients of the model
            self.zero_grad()
            
            # block_output: (bsz, N, D)
            feat = block_output[:, tk_idx, ...]  # (bsz, D) - token at tk_idx
            
            # For timm, we work in embed_dim space (no projection like OpenCLIP)
            latent_feat = F.normalize(self.norm(feat), dim=-1)  # (bsz, D)
            
            # Compute similarity with concept vectors
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)  # (bsz, num_concepts)
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm *= weight  # (bsz, num_concepts)
            sim = sim_bm.sum(dim=0)  # (bsz, num_concepts) -> (num_concepts)
            self.sim_bms.append(weight) # (num_concepts)
            
            # Compute gradients of sim w.r.t. attn_weight
            # attn_weight shape: (bsz, num_heads, N, N)
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            
            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, bsz, num_heads, N, N)
            if weighted_attn:
                grad = torch.einsum('m b h i j, b h j k -> m b h i k', grad, attn_weight.transpose(-2, -1))  # Matrix multiplication: grad @ attn_weight^T
            grad = torch.clamp(grad, min=0.)
            # grad is already in shape (num_concepts, bsz, num_heads, N, N)
            self.grads.append(grad)
            
            # Average over heads and query positions, exclude CLS token
            image_relevance = grad.mean(dim=2).mean(dim=-2)[..., 1:]  # (num_concepts, bsz, N-1)
            expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)
            self.maps.append(expl_map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [bsz, num_concepts, H, W]
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


class TimmCVWrapper(CopyAttrWrapper):
    """
    A timm-specific wrapper that propagates gradients from the last layer backward through
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
    
    This wrapper processes all layers in reverse order (from deepest to shallowest).
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # Get the number of blocks before calling super().__init__
        num_blocks = len(model.blocks)
        
        # Store layer_indices for aggregation, but use ALL layers for hook registration
        all_layer_indices = list(range(num_blocks))
        
        # The copying behavior is done by the parent class CopyAttrWrapper
        # Pass ALL layer indices for hook registration
        super().__init__(model, layer_indices=all_layer_indices, include_private=include_private)

        # Store patch info for later use
        self._patch_size = model.patch_embed.proj.kernel_size[0]
        self.num_heads = model.blocks[0].attn.num_heads
        self._num_blocks = num_blocks
        
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
        self.pseudo_handles = []
        self.normal_handles = []

    def _save_block_input(self, module, input, output):
        # input is a tuple, input[0] is the actual input tensor
        # timm block input: (B, N, D)
        # Transpose to (N, B, D) to match OpenCLIP format
        self.block_ins.append(input[0].transpose(0, 1))  # (N, B, D)

    def switch_to_pseudo_mode(self):
        """Switch the Attention modules to pseudo mode that returns attention weights."""
        for handle in self.normal_handles:
            handle.remove()
        self.normal_handles = []
        for idx in self._requested_hook_indices:
            block = self.blocks[idx]
            # Override attention forward method to pseudo mode (only attend to CLS token)
            block.attn.forward = types.MethodType(Pseudo_Attention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = True
            self.pseudo_handles.append(block.attn.register_forward_hook(self._save_attn_hook))

    def switch_to_normal_mode(self):
        """Switch back to the normal Attention modules."""
        for handle in self.pseudo_handles:
            handle.remove()
        self.pseudo_handles = []
        for idx in self._requested_hook_indices:
            block = self.blocks[idx]
            self.normal_handles.append(block.register_forward_hook(self._save_block_input))
            # Restore attention forward method
            block.attn.forward = types.MethodType(Attention_forward, block.attn)
            for name, param in block.named_parameters():
                param.requires_grad = False

    def _save_attn_hook(self, module, input, output):
        # attn_weights: (B*num_heads, 1, N) from Pseudo_Attention_forward
        self.attn_weight = module._attn_weights

    def dot_concept_vectors(self, concept_vectors: torch.Tensor, power: int = 1):
        """Compute gradient-based concept activation maps using pseudo mode.
        
        Backpropagation starts from the last layer using the provided concept_vectors
        (typically from the classifier head), then propagates through ALL layers.
        Each layer i uses the CLS gradient from layer i+1 as its concept vector.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors.
                For the last (deepest) layer, this is used as the concept vector.
                Typically extracted from model.head.weight for a specific class.
            power (int): Power for similarity scaling. Default: 1
        
        Note:
            Gradients are computed for ALL layers (0 to num_blocks-1).
            The layer_indices parameter in __init__ only affects aggregation.
        """
        w = h = int(math.sqrt(self.block_ins[0].shape[0] - 1))  # Exclude CLS token
        self.switch_to_pseudo_mode()
        
        # Process ALL layers in reverse order (from deepest to shallowest)
        # _requested_hook_indices contains all layer indices (0 to num_blocks-1)
        sorted_indices = sorted(enumerate(self._requested_hook_indices), key=lambda x: x[1], reverse=True)
        
        # For the first (deepest) layer, use the provided concept_vectors directly
        # concept_vectors shape: (num_concepts, D)
        current_concept_vectors = concept_vectors  # (M, D)
        
        # We'll collect results in reverse order (deepest to shallowest), then reverse at the end
        # Note: attn_grads and cls_grads are stored for ALL layers
        # But maps and sim_bms are only stored for layers in _aggregate_layer_indices
        attn_grads_reversed = []
        cls_grads_reversed = []
        maps_reversed = []
        sim_bms_reversed = []
        
        # Create a set for fast lookup of which layers to store maps/sim_bms for
        aggregate_layer_set = set(self._aggregate_layer_indices)
        
        for enum_idx, layer_idx in sorted_indices:
            self.zero_grad()
            block = self.blocks[layer_idx]
            data_in = self.block_ins[enum_idx]
            
            # data_in: (N, B, D), transpose to (B, N, D) for timm
            x = data_in.transpose(0, 1)
            
            # Get CLS token from input - requires grad for computing cls_grad
            input_cls = x[:, 0, :].clone()
            input_cls.requires_grad_(True)
            
            # Create input with the grad-enabled CLS token
            x_with_grad = x.clone()
            x_with_grad[:, 0, :] = input_cls
            
            # Run through the block
            x_norm = block.norm1(x_with_grad)
            attn_out = block.attn(x_norm)  # Returns (1, B, D) in pseudo mode
            # attn_out is (1, B, D), transpose back and add residual for CLS only
            cls_out = attn_out.squeeze(0) + x_with_grad[:, 0, :]  # (B, D)
            # Apply norm2 and mlp only to CLS token
            cls_norm2 = block.norm2(cls_out)
            cls_mlp = block.mlp(cls_norm2)
            cls_feat = cls_out + cls_mlp  # (B, D)
            
            # Apply final LayerNorm
            latent_feat = F.normalize(self.norm(cls_feat), dim=-1)  # (B, D)
            
            # Compute similarity with concept vectors (keeping concept dimension)
            # current_concept_vectors: (M, D) where M is num_concepts
            sim_bm = torch.einsum('b d, m d -> b m', latent_feat, current_concept_vectors)  # (B, M)
            if power == 0:
                weight = torch.ones_like(sim_bm)
            else:
                weight = torch.abs(sim_bm.clone().detach()).pow(power)
                sim_bm = sim_bm * weight  # (B, M)
            sim = sim_bm.sum(dim=0)  # (B, M) -> (M,) - sum over batch, keep concept dimension
            
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
            cls_grad = grads[1]   # (M, B, D) - gradient of input CLS token for each concept
            
            # Store attention gradient (for all layers)
            if attn_grad is not None:
                attn_grad = torch.clamp(attn_grad, min=0.)
                attn_grad = rearrange(attn_grad, 'm (b h) n1 n2 -> m b h n1 n2', h=self.num_heads)  # (M, B, num_heads, 1, N)
                attn_grads_reversed.append(attn_grad)
                
                # Store explanation map only for layers in layer_indices
                if layer_idx in aggregate_layer_set:
                    # Compute explanation map keeping concept dimension
                    # Format: (H, W, B, M) to match TimmWrapper
                    image_relevance = attn_grad.mean(dim=2).squeeze(-2)[..., 1:]  # (M, B, N-1) Exclude CLS token
                    expl_map = rearrange(image_relevance, 'm b (h w) -> h w b m', w=w, h=h)  # (h, w, B, M)
                    maps_reversed.append(expl_map)
            
            # Store CLS gradient and use it as concept_vectors for the previous (shallower) layer
            if cls_grad is not None:
                cls_grads_reversed.append(cls_grad)
                # Use the CLS gradient as concept vectors for the next layer
                # cls_grad shape: (M, B, D), average over batch to get (M, D)
                current_concept_vectors = F.normalize(cls_grad.mean(dim=1), dim=-1)  # (M, D)
        
        # Reverse to get forward order (shallowest to deepest)
        self.attn_grads = attn_grads_reversed[::-1]
        self.cls_grads = cls_grads_reversed[::-1]
        self.maps = maps_reversed[::-1]
        self.sim_bms = sim_bms_reversed[::-1]
        
        self.switch_to_normal_mode()

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across the layers specified in layer_indices.
        
        The maps are already filtered to only contain layers from layer_indices.
        This method sums them together and normalizes the result.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and dot_concept_vectors first.")
        
        # self.maps already contains only maps for layers in layer_indices
        # Each map has shape (H, W, B, M) where M is num_concepts
        maps = torch.stack(self.maps, dim=0)  # (num_selected_layers, h, w, B, M)
        maps = torch.einsum('l h w b m -> h w b m', maps)  # sum across layers
        maps = rearrange(maps, 'h w b m -> b m h w')  # (B, M, h, w)

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min)
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
        # CopyAttrWrapper has no tensor buffers of its own, still call the parent class
        return super().to(*args, **kwargs)
