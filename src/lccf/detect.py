# src/my_transformers/detect.py
from typing import Optional, List, Any
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, ToTensor
from .types import TimmViT, TorchViT, OpenCLIPViT
from .wrap import CopyAttrWrapper
# Import the specific backend wrapper (if it exists)
from .backends.openclip.wrapper import OpenCLIPWrapper, OpenCLIPGradWrapper, AOpenCLIPWrapper
from .backends.timm.wrapper import TimmWrapper, TimmGradWrapper, ATimmWrapper
from .backends.torchvision.wrapper import TorchvisionWrapper, TorchvisionGradWrapper, ATorchvisionWrapper

def detect_and_wrap(model: Any,
                    layer_indices: Optional[List[int]] = None,
                    prefer: Optional[str] = None,
                    async_compute: bool = False,
                    include_private: bool = False) -> CopyAttrWrapper:
    """
    Simply determines and returns a specific backend CopyAttrWrapper instance based on isinstance.
    prefer: optional, string: 'openclip'|'timm'|'torchvision' to force branching (if matched)
    async_compute: if True, use async-capable wrapper (A*Wrapper), otherwise use gradient-based wrapper (GradWrapper)
    """
    if model is None:
        raise ValueError("model cannot be None")

    # prefer preferred (override automatic judgment if needed)
    if prefer == "openclip" and isinstance(model.visual, OpenCLIPViT):
        if async_compute:
            return AOpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return OpenCLIPGradWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if prefer == "timm" and isinstance(model, TimmViT):
        if async_compute:
            return ATimmWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return TimmGradWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if prefer == "torchvision" and isinstance(model, TorchViT):
        if async_compute:
            return ATorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return TorchvisionGradWrapper(model, layer_indices=layer_indices, include_private=include_private)

    # Default type-based judgment (order can be adjusted)
    if isinstance(model.visual, OpenCLIPViT):
        if async_compute:
            return AOpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return OpenCLIPGradWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if isinstance(model, TimmViT):
        if async_compute:
            return ATimmWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return TimmGradWrapper(model, layer_indices=layer_indices, include_private=include_private)
    if isinstance(model, TorchViT):
        if async_compute:
            return ATorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private, async_compute=True)
        else:
            return TorchvisionGradWrapper(model, layer_indices=layer_indices, include_private=include_private)

    # fallback: Raise error
    raise TypeError("Unable to detect the backend type of the model, or the backend is not supported. Please ensure the model is an open_clip, timm, or torchvision ViT model, or use the prefer parameter to force specify the backend.")

def wrap_clip_preprocess(preprocess, image_size=224):
    """
    Modify OpenCLIP preprocessing to accept arbitrary image size.
    Args:
        preprocess: original OpenCLIP preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        preprocess.transforms[-3],
        preprocess.transforms[-2],
        preprocess.transforms[-1],
    ])

def wrap_timm_preprocess(preprocess, image_size=224):
    """
    Modify timm preprocessing to accept arbitrary image size.
    Args:
        preprocess: original timm preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        *preprocess.transforms[-2:],  # skip the first resize
    ])

def wrap_torchvision_preprocess(preprocess, image_size=224):
    """
    Modify torchvision preprocessing to accept arbitrary image size.
    Args:
        preprocess: original torchvision preprocess transform
        image_size: target image size (square)
    """
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        # *preprocess.transforms[-2:],  # ToTensor and Normalize
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])