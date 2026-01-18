# %%
"""
TimmTestWrapper Example Script

This script demonstrates how to use the TimmTestWrapper from the LCCF library.
TimmTestWrapper propagates gradients backward from the last layer through ALL layers,
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
from lccf.backends.timm.wrapper import TimmTestWrapper
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
layer_indices = [0, 1, 2]

# %%
# Extract concept vector from classifier head (same approach as timm_cat_remote.py)
# ImageNet class 281: tabby cat
TABBY_CAT_IDX = 281
concept_names = ["tabby cat"]

concept_vectors = model.head.weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
# Create TimmTestWrapper
# layer_indices specifies which layers to aggregate in aggregate_layerwise_maps()
wrapper = TimmTestWrapper(model, layer_indices=layer_indices)

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
wrapper.dot_concept_vectors(concept_vectors, power=2)

# %%
# Access the stored gradients (computed for ALL 12 layers)
print(f"\nNumber of attention gradients stored (all layers): {len(wrapper.attn_grads)}")
print(f"Number of CLS gradients stored (all layers): {len(wrapper.cls_grads)}")
print(f"Number of explanation maps stored (all layers): {len(wrapper.maps)}")
print(f"Number of sim_bms stored (all layers): {len(wrapper.sim_bms)}")

# Print shapes for first and last layers
print(f"\nLayer 0 (shallowest):")
print(f"  Attention gradient shape: {wrapper.attn_grads[0].shape}")
print(f"  CLS gradient shape: {wrapper.cls_grads[0].shape}")
print(f"  Explanation map shape: {wrapper.maps[0].shape}")
print(f"  sim_bm shape: {wrapper.sim_bms[0].shape}")

print(f"\nLayer 11 (deepest):")
print(f"  Attention gradient shape: {wrapper.attn_grads[11].shape}")
print(f"  CLS gradient shape: {wrapper.cls_grads[11].shape}")
print(f"  Explanation map shape: {wrapper.maps[11].shape}")
print(f"  sim_bm shape: {wrapper.sim_bms[11].shape}")

# %%
# For visualization, select maps for the layers we want to show
# wrapper.maps has shape [H, W, B, 1] which is compatible with visualize_layerwise_maps
selected_maps = [wrapper.maps[i] for i in layer_indices]
selected_sim_bms = [wrapper.sim_bms[i] for i in layer_indices]

# %%
# Visualize layerwise maps for selected layers (like in timm_cat_remote.py)
print(f"\n=== Visualizing attention maps for layers {layer_indices} ===")
visualize_layerwise_maps(image, selected_maps, sim_bms=selected_sim_bms, text_prompts=concept_names, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# %%
# Aggregate maps across selected layers (only layer_indices)
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
