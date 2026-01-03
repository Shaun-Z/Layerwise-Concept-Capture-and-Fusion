# src/lccf/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from open_clip.transformer import ResidualAttentionBlock
from einops import rearrange

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper

from ...types import ResAttnBlk

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

        # Register hooks to the specified layers to capture attention outputs
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
           
            # block.attn.register_forward_hook(lambda module, input, output: self.attn_out.append(output[0].detach()))
            # block.ln_2.register_forward_hook(lambda module, input, output: self.norm_mean.append(input[0].mean(dim=-1).detach()))
            # block.ln_2.register_forward_hook(lambda module, input, output: self.norm_std.append(input[0].std(dim=-1).detach()))

            # block.attn.register_forward_hook(lambda module, input, output: self.result.append(output[0].detach()))
            # block.ln_2.register_forward_hook(lambda module, input, output: self.calc_list(self.result, idx, input[0].std(dim=-1).detach()))
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
        std = output[0, ...].std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b -> 1 b 1')
        self.tmp *= self.visual.ln_post.weight
        self.tmp @= self.visual.proj
        self.tmp = F.normalize(self.tmp, dim=-1)
        self.result.append(self.tmp)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        for res in self.result:
            self.maps.append(torch.einsum('n b d, m d -> n b m', res, concept_vectors))

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
    
class ResidualAttentionBlockWrapper(ResAttnBlk):
    """
    A wrapper for open-clip's ResidualAttentionBlock to expose attention weights if needed.
    """
    def __init__(self, block: ResAttnBlk):
        super().__init__(block)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        qkv_x = self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = q_x + qkv_x
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x