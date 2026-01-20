# %%
"""
TimmCVWrapper Example Script

This script demonstrates how to use the TimmCVWrapper from the LCCF library.
TimmCVWrapper propagates gradients backward from the last layer through ALL layers,
using each layer's CLS gradient as the concept vector for the previous (shallower) layer.

Key behavior:
- Gradients are computed for ALL layers (0 to 11 for ViT-B-16)
- Each layer i uses the CLS gradient from layer i+1 as its concept vector
- layer_indices only affects which layers are used in aggregate_layerwise_maps()
- For the deepest layer, the concept vector comes from the model's classifier head
"""

# %%
import requests
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt
from lccf.backends.timm.wrapper import TimmFCVWrapper
from lccf.detect import wrap_timm_preprocess
from lccf.utils import visualize, visualize_layerwise_maps

# %%
# Create timm ViT-B-16 model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Get the preprocessing transform for the model
config = resolve_data_config({}, model=model)
preprocess = create_transform(**config)
# Wrap the preprocess to accept arbitrary image size
preprocess = wrap_timm_preprocess(preprocess, image_size=224)

# %%
# Define which layers to use for aggregation
# Note: Gradients are computed for ALL layers (0-11), but only these layers
# will be used in aggregate_layerwise_maps()
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# Extract concept vector from classifier head (same approach as timm_cat_remote.py)
# ImageNet class 281: tabby cat
TABBY_CAT_IDX = 281
concept_names = ["tabby cat"]

concept_vectors = model.head.weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
# Create TimmCVWrapper
# layer_indices specifies which layers to aggregate in aggregate_layerwise_maps()
wrapper = TimmFCVWrapper(model, layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
# Forward pass to extract features and capture block inputs for ALL layers
features = wrapper.forward_features(image)
print(f"Features shape: {features.shape}")
print(f"Number of block inputs captured (all layers): {len(wrapper.block_ins)}")

# %%
# Compute layerwise gradients using dot_concept_vectors
# This computes gradients for ALL layers, propagating from layer 11 -> 10 -> ... -> 0
# For layer 11 (deepest), uses the provided concept_vectors (from classifier head)
# For layer i < 11, uses the CLS gradient from layer i+1
wrapper.dot_concept_vectors(concept_vectors, power=1)

# %%
# Access the stored gradients
# attn_grads and cls_grads are computed for ALL 12 layers
# maps and sim_bms are only stored for layers in layer_indices
print(f"Number of explanation maps stored (only layer_indices): {len(wrapper.maps)}")
print(f"Number of sim_bms stored (only layer_indices): {len(wrapper.sim_bms)}")

# Print shapes for maps and sim_bms (only layer_indices)
print(f"\nMaps and sim_bms (for layers {layer_indices}):")
for i, (m, s) in enumerate(zip(wrapper.maps, wrapper.sim_bms)):
    print(f"  Layer index {i}: map shape={m.shape}, sim_bm shape={s.shape}")

# %%
# wrapper.maps and wrapper.sim_bms already contain only data for layer_indices
# They can be passed directly to visualize_layerwise_maps
print(f"\n=== Visualizing attention maps for layers {layer_indices} ===")
visualize_layerwise_maps(image,
                         wrapper.maps,
                         sim_bms=wrapper.sim_bms,
                         text_prompts=concept_names,
                         mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        #  normalize_each_map=True
                         )

# %%
# Aggregate maps across layer_indices
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"\nAggregated maps shape (from layers {layer_indices}): {maps_aggregated.shape}")

# %%
# Visualize aggregated maps (like in timm_cat_remote.py)
print("\n=== Visualizing aggregated attention map ===")
visualize(image, maps_aggregated, text_prompts=concept_names, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# %%
# Reset the wrapper for next use
wrapper.reset()
print("Wrapper reset for next use.")
