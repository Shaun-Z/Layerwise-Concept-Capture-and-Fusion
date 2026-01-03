
# 统一安全导入可能需要用于 isinstance 的类型
try:
    # timm VisionTransformer (位置可能随 timm 版本不同)
    from timm.models.vision_transformer import VisionTransformer as TimmViT
except Exception:
    TimmViT = tuple()  # empty tuple -> isinstance(..., ()) is always False

try:
    # torchvision ViT (newer torchvision may expose VisionTransformer)
    from torchvision.models.vision_transformer import VisionTransformer as TorchViT
except Exception:
    TorchViT = tuple()

try:
    # open_clip 的 VisionTransformer（路径可能不同）
    from open_clip.model import VisionTransformer as OpenCLIPViT
except Exception:
    OpenCLIPViT = tuple()

try:
    from open_clip.transformer import ResidualAttentionBlock as ResAttnBlk
except Exception:
    ResAttnBlk = tuple()