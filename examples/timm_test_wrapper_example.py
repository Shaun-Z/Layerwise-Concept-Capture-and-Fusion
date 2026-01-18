# %%
"""
TimmTestWrapper Example Script

This script demonstrates how to use the TimmTestWrapper from the LCCF library.
TimmTestWrapper propagates gradients backward from the last layer, using each 
layer's CLS gradient as the concept vector for the previous (shallower) layer.

For the deepest layer, the concept vector comes from the model's classifier head
(similar to how TimmWrapper uses concept vectors).
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
# Define which layers to capture gradients from
# For ViT-B-16, there are 12 transformer blocks (indices 0-11)
layer_indices = [0, 3, 6, 9, 11]

# %%
# Extract concept vector from classifier head (same approach as timm_cat_remote.py)
# ImageNet class 281: tabby cat
TABBY_CAT_IDX = 281
concept_vectors = model.head.weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
# Create TimmTestWrapper
wrapper = TimmTestWrapper(model, layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
# Forward pass to extract features and capture block inputs
features = wrapper.forward_features(image)
print(f"Features shape: {features.shape}")
print(f"Number of block inputs captured: {len(wrapper.block_ins)}")

# %%
# Compute layerwise gradients using dot_concept_vectors
# For the deepest layer, uses the provided concept_vectors (from classifier head)
# For shallower layers, uses CLS gradients propagated from deeper layers
wrapper.dot_concept_vectors(concept_vectors, power=2)

# %%
# Access the stored gradients
print(f"\nNumber of attention gradients stored: {len(wrapper.attn_grads)}")
print(f"Number of CLS gradients stored: {len(wrapper.cls_grads)}")
print(f"Number of explanation maps stored: {len(wrapper.maps)}")

# Print shapes
for i, (attn_grad, cls_grad, expl_map) in enumerate(zip(wrapper.attn_grads, wrapper.cls_grads, wrapper.maps)):
    print(f"\nLayer {layer_indices[i]}:")
    print(f"  Attention gradient shape: {attn_grad.shape}")
    print(f"  CLS gradient shape: {cls_grad.shape}")
    print(f"  Explanation map shape: {expl_map.shape}")

# %%
# Aggregate maps across layers
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"\nAggregated maps shape: {maps_aggregated.shape}")

# %%
# Visualize the aggregated attention map
def visualize_map(image_tensor, attention_map, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    """Visualize the attention map overlaid on the original image."""
    mean, std = mean_std
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    # Denormalize image
    img = image_tensor.cpu() * std + mean
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    # Get attention map
    attn_map = attention_map.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attention map
    axes[1].imshow(attn_map, cmap='jet')
    axes[1].set_title("Attention Map")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(attn_map, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# Visualize the result
visualize_map(image, maps_aggregated, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# %%
# Reset the wrapper for next use
wrapper.reset()
print("Wrapper reset for next use.")
