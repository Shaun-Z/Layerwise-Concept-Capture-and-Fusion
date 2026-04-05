# src/my_transformers/detect.py
from typing import Optional, List, Any, Literal
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize, ToTensor
from .types import TimmViT, TorchViT, OpenCLIPViT
from .wrap import CopyAttrWrapper
# Import the specific backend wrapper (if it exists)
from .backends.openclip.wrapper import OpenCLIPWrapper, OpenCLIPFastWrapper, OpenCLIPCVWrapper, OpenCLIPFCVWrapper
from .backends.timm.wrapper import TimmWrapper, TimmFastWrapper, TimmCVWrapper, TimmFCVWrapper, TimmFCVHybridWrapper
from .backends.torchvision.wrapper import TorchvisionWrapper, TorchvisionFastWrapper, TorchvisionCVWrapper, TorchvisionFCVWrapper

def detect_and_wrap(model: Any,
                    layer_indices: Optional[List[int]] = None,
                    prefer: Optional[str] = None,
                    mode: Literal["standard", "fast", "cv", "fcv", "fcv_hybrid"] = "standard",
                    top_k: int = 32,
                    include_private: bool = False) -> CopyAttrWrapper:
    """
    Simply determines and returns a specific backend CopyAttrWrapper instance based on isinstance.
    prefer: optional, string: 'openclip'|'timm'|'torchvision' to force branching (if matched)
    mode: "standard", "fast", or "cv" to select the type of wrapper
    """
    if model is None:
        raise ValueError("model cannot be None")

    # prefer preferred (override automatic judgment if needed)
    if prefer == "openclip" and isinstance(model.visual, OpenCLIPViT):
        if mode == "standard":
            return OpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return OpenCLIPFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return OpenCLIPCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return OpenCLIPFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    if prefer == "timm" and isinstance(model, TimmViT):
        if mode == "standard":
            return TimmWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return TimmFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return TimmCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return TimmFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv_hybrid":
            return TimmFCVHybridWrapper(model, layer_indices=layer_indices, top_k=top_k, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    if prefer == "torchvision" and isinstance(model, TorchViT):
        if mode == "standard":
            return TorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return TorchvisionFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return TorchvisionCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return TorchvisionFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Default type-based judgment (order can be adjusted)
    if isinstance(model.visual, OpenCLIPViT):
        if mode == "standard":
            return OpenCLIPWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return OpenCLIPFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return OpenCLIPCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return OpenCLIPFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    if isinstance(model, TimmViT):
        if mode == "standard":
            return TimmWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return TimmFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return TimmCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return TimmFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv_hybrid":
            return TimmFCVHybridWrapper(model, layer_indices=layer_indices, top_k=top_k, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    if isinstance(model, TorchViT):
        if mode == "standard":
            return TorchvisionWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fast":
            return TorchvisionFastWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "cv":
            return TorchvisionCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        elif mode == "fcv":
            return TorchvisionFCVWrapper(model, layer_indices=layer_indices, include_private=include_private)
        else:
            raise ValueError(f"Unknown mode: {mode}")

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