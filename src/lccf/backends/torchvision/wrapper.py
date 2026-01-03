# src/lccf/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn

# CopyAttrWrapper is defined in lccf.wrap
from ...wrap import CopyAttrWrapper

"""
Description:
 - The wrapper accepts an already constructed torchvision model (e.g., via torchvision.models). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """

class TorchvisionWrapper(CopyAttrWrapper):
    """
    A torchvision-specific derivative of CopyAttrWrapper that provides convenient methods for encode_text / encode_image.
    Important: This wrapper assumes that you are passing in the model of torchvision (typically the model returned by torchvision.models).
    passed in. If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

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