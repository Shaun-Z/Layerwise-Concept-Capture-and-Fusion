# src/lccf/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper

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
        self.tmp = None
        # self.norm_mean = []
        # self.norm_std = []
        self.result = []
        self.maps = []
        self.normed_clss = []

        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]

            block.attn.register_forward_hook(self._save_attn_output)    # (n, b, d)
            block.ln_2.register_forward_hook(self._aggregate_ln)        # (n, b, d)
            block.mlp.c_fc.register_forward_hook(self._aggregate_c_fc)
            block.mlp.c_proj.register_forward_hook(self._aggregate_c_proj)  # (n, b, d)
            block.register_forward_hook(self._finalize_hook)    # (n, b, d)


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
    