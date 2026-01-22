# %%
"""
TorchvisionCVWrapper Example Script

This script demonstrates how to use the TorchvisionCVWrapper from the LCCF library.
TorchvisionCVWrapper propagates gradients backward from the last layer through ALL layers,
using each layer's CLS gradient as the concept vector for the previous (shallower) layer.

Key behavior:
- Gradients are computed for ALL layers (0 to 11 for ViT-B-16)
- Each layer i uses the CLS gradient from layer i+1 as its concept vector
- layer_indices only affects which layers are used in aggregate_layerwise_maps()
- For the deepest layer, the concept vector comes from the model's classification head

Unlike OpenCLIP/CLIP models which have text encoders for generating concept vectors,
torchvision models are pure vision models. This example uses concept vectors extracted from
the classification head weights corresponding to ImageNet classes (e.g., "tabby cat").
"""

# %%
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from lccf.backends.torchvision.wrapper import TorchvisionFCVWrapper
from lccf.detect import wrap_torchvision_preprocess
from lccf.utils import visualize, visualize_layerwise_maps

# %%
# Create torchvision ViT-B-16 model
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

# Get the preprocessing transform for the model (standard ImageNet normalization)
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Wrap the preprocess to accept arbitrary image size
preprocess = wrap_torchvision_preprocess(preprocess, image_size=224)

# %%
# Define which layers to use for aggregation
# Note: Gradients are computed for ALL layers (0-11), but only these layers
# will be used in aggregate_layerwise_maps()
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# Extract concept vectors from the classification head
# The head weight matrix has shape [num_classes, hidden_dim] = [1000, 768]
# Each row is the weight vector for a class, which can be used as a concept vector

# ImageNet class indices for concepts of interest
# Class 281: tabby cat
TABBY_CAT_IDX = 281

concept_names = ["tabby cat"]

# Extract concept vector for tabby cat from the classification head
# The weight vector for class i is model.heads[0].weight[i]
concept_vectors = model.heads[0].weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)
print(f"Concept vectors shape: {concept_vectors.shape}")

# %%
# Create TorchvisionFCVWrapper
# layer_indices specifies which layers to aggregate in aggregate_layerwise_maps()
wrapper = TorchvisionFCVWrapper(model, layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000001675.jpg",
]
images = [preprocess(Image.open(requests.get(url, stream=True).raw)) for url in urls]
images = torch.stack(images, dim=0).to(device)

# %%
# Forward pass to extract features and capture block inputs for ALL layers
output = wrapper(images.clone().detach().requires_grad_(True))
print(f"Model output shape: {output.shape}")
print(f"Number of block inputs captured (all layers): {len(wrapper.block_ins)}")

# %%
# Compute layerwise gradients using dot_concept_vectors
# This computes gradients for ALL layers, propagating from layer 11 -> 10 -> ... -> 0
# For layer 11 (deepest), uses the provided concept_vectors (from classifier head)
# For layer i < 11, uses the CLS gradient from layer i+1
wrapper.dot_concept_vectors(concept_vectors, power=0)

# %%

# Print shapes for maps and sim_bms (only layer_indices)
print(f"\nMaps and sim_bms (for layers {layer_indices}):")
for i, m in enumerate(wrapper.maps):
    print(f"  Layer index {i}: map shape={m.shape}")

# %%
# wrapper.maps and wrapper.sim_bms already contain only data for layer_indices
# They can be passed directly to visualize_layerwise_maps
print(f"\n=== Visualizing attention maps for layers {layer_indices} ===")
visualize_layerwise_maps(images,
                         wrapper.maps,
                         sim_bms=wrapper.sim_bms,
                         text_prompts=concept_names,
                         mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         normalize_each_map=True
                        )

# %%
# Aggregate maps across layer_indices
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"\nAggregated maps shape (from layers {layer_indices}): {maps_aggregated.shape}")

# %%
# Visualize aggregated maps
print("\n=== Visualizing aggregated attention map ===")
visualize(images, maps_aggregated, text_prompts=concept_names, mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

# %%
# Reset the wrapper for next use
wrapper.reset()
print("Wrapper reset for next use.")
