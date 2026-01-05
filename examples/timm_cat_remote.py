# %%
"""
Timm ViT-B-16 Example Script

This script demonstrates how to use the LCCF (Layerwise Concept Capture and Fusion) 
library with a timm-implemented ViT-B-16 model.

Note: Unlike OpenCLIP/CLIP models which have text encoders for generating concept vectors,
timm models are pure vision models. For this example, we demonstrate with random concept
vectors. In practice, you would obtain concept vectors from:
1. A separate text encoder (e.g., CLIP text encoder)
2. Pre-computed concept embeddings
3. Learned concept vectors specific to your task
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
from lccf.detect import detect_and_wrap
from lccf.utils import visualize, visualize_layerwise_maps

# %%
# Create timm ViT-B-16 model
model = timm.create_model('vit_base_patch16_224', pretrained=False)
model.eval()

# Get the preprocessing transform for the model
config = resolve_data_config({}, model=model)
preprocess = create_transform(**config)

# %%
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# For demonstration, we use random concept vectors
# In practice, these would come from a text encoder or be learned
concept_names = ["concept_1", "concept_2", "concept_3"]
num_concepts = len(concept_names)
embed_dim = model.embed_dim  # 768 for ViT-B-16

# Create random normalized concept vectors
torch.manual_seed(42)  # For reproducibility
concept_vectors = torch.randn(num_concepts, embed_dim)
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
# Forward pass to extract features
features = wrapper.forward_features(image)
print(f"Features shape: {features.shape}")

# %%
# Compute concept activation maps
wrapper.dot_concept_vectors(concept_vectors)

# %%
maps = torch.stack(wrapper.maps, dim=0)  # (num_layers, H, W, B, num_concepts)
print(f"Maps shape: {maps.shape}")

# %%
# Visualize layerwise maps
visualize_layerwise_maps(image, wrapper.maps, text_prompts=concept_names)

# %%
# Aggregate maps across layers
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"Aggregated maps shape: {maps_aggregated.shape}")

# %%
# Visualize aggregated maps
visualize(image, maps_aggregated, text_prompts=concept_names)
