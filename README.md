# Layerwise-Concept-Capture-and-Fusion (LCCF)

LCCF is a concept decomposition and visualization toolkit for ViT models. It wraps OpenCLIP, timm, and torchvision Vision Transformers, captures attention/gradients at selected layers, and produces layer-wise concept heatmaps with visualization utilities.

## Features
- Auto-detect and wrap OpenCLIP / timm / torchvision ViT models
- Multiple modes: `standard`, `fast`, `cv`, `fcv`
- Layer-wise and aggregated concept heatmaps
- Preprocess adapters and visualization helpers

## Installation
From this module directory:

```bash
pip install -e .
```

Dependencies are managed by the parent project. For standalone usage, ensure:
- torch / torchvision
- timm (optional)
- open-clip-torch (optional)
- einops, pillow, opencv-python, matplotlib

## Quickstart (OpenCLIP)

```python
import torch
import open_clip
from lccf.detect import detect_and_wrap, wrap_clip_preprocess
from lccf.utils import visualize

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
)
model.eval()
preprocess = wrap_clip_preprocess(preprocess, image_size=224)

layer_indices = [0, 2, 5, 8, 11]
wrapper = detect_and_wrap(model, prefer="openclip", mode="fast", layer_indices=layer_indices)

# Concept vectors (example: text embeddings)
tokenizer = open_clip.get_tokenizer(model_name="ViT-B-16")
text = tokenizer(["a photo of a cat", "a photo of a dog"]).to(device)
concept_vectors = model.encode_text(text, normalize=True)

# Forward image
image = torch.randn(1, 3, 224, 224, device=device)
_ = wrapper.encode_image(image)

# Compute concept-related attention maps
wrapper.dot_concept_vectors(concept_vectors)
heatmaps = wrapper.aggregate_layerwise_maps()  # [B, M, H, W]

# Visualize
visualize(
    image,
    heatmaps,
    mean_std=(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ),
)
```

See more examples in `examples/`.

## Main API

### detect_and_wrap
```python
from lccf.detect import detect_and_wrap
wrapper = detect_and_wrap(model, layer_indices=[0, 1, 2], prefer="openclip", mode="fast")
```

Parameters:
- `model`: OpenCLIP / timm / torchvision ViT model instance
- `layer_indices`: transformer block indices to sample
- `prefer`: optional `openclip` / `timm` / `torchvision` to force backend
- `mode`: `standard` / `fast` / `cv` / `fcv`

### Preprocess adapters
```python
from lccf.detect import wrap_clip_preprocess, wrap_timm_preprocess, wrap_torchvision_preprocess
```

Use them to adjust the original preprocess to a custom input size.

## Modes
- `standard`: attention + gradient based, stable but slower
- `fast`: pseudo attention for speed
- `cv`: propagates CLS gradients as concept vectors layer by layer
- `fcv`: propagates gradients of all tokens as concept vectors (full token concept vectors)

Performance and memory usage vary by mode; start with `fast` or `standard`.

## Visualization
`lccf.utils` provides:
- `visualize`: overlay aggregated heatmaps on images
- `visualize_layerwise_maps`: show layer-wise heatmaps

## Backends
- OpenCLIP: ViT models from `open-clip-torch`
- timm: ViT models from `timm.create_model(..., pretrained=True)`
- torchvision: ViT models from `torchvision.models.vision_transformer`

## Examples
`examples/` includes:
- `openclip_cat_remote.py`
- `openclip_coco_batch_fast.py`
- `openclip_coco_batch_fcv.py`
- `timm_*_wrapper_example.py`
- `torchvision_*_wrapper_example.py`

## Tests
From this module directory:

```bash
pytest tests
```
