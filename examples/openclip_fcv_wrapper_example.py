# %%
"""
OpenCLIPFCVWrapper Example Script

This script demonstrates how to use the OpenCLIPFCVWrapper from the LCCF library.
OpenCLIPFCVWrapper propagates gradients backward from the last layer through ALL layers,
using each layer's CLS gradient as the concept vector for the previous (shallower) layer.

Key behavior:
- Gradients are computed for ALL layers (0 to 11 for ViT-B-16)
- Each layer i uses the CLS gradient from layer i+1 as its concept vector
- layer_indices only affects which layers are used in aggregate_layerwise_maps()
- For the deepest layer, the concept vector comes from text embeddings (projected to latent space)
"""

# %%
import requests
from PIL import Image
import torch
import open_clip
from lccf.backends.openclip.wrapper import OpenCLIPFCVWrapper
from lccf.detect import wrap_clip_preprocess
from lccf.utils import visualize, visualize_layerwise_maps
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=device)
model.eval()
preprocess = wrap_clip_preprocess(preprocess, image_size=224)
tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')

# %%
# Define which layers to use for aggregation
# Note: Gradients are computed for ALL layers (0-11), but only these layers
# will be used in aggregate_layerwise_maps()
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# Text prompts for concept vectors
prompts = ["a photo of cats", "a photo of a red sofa", "a photo of a remote control"]

# %%
# Create OpenCLIPCVWrapper
# layer_indices specifies which layers to aggregate in aggregate_layerwise_maps()
wrapper = OpenCLIPFCVWrapper(model, layer_indices=layer_indices)

# %%
# Encode text to get concept vectors in the shared latent space
text = tokenizer(prompts).to(device)
text_embeddings = model.encode_text(text, normalize=True)
print(f"Text embeddings shape: {text_embeddings.shape}")

# %%
# Load a sample image from COCO dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
# Forward pass to extract features and capture block inputs for ALL layers
features = wrapper.encode_image(image.clone().detach().requires_grad_(True))
print(f"Features shape: {features.shape}")
print(f"Number of block inputs captured (all layers): {len(wrapper.block_ins)}")

# %%
# Compute layerwise gradients using dot_concept_vectors
# This computes gradients for ALL layers, propagating from layer 11 -> 10 -> ... -> 0
# For layer 11 (deepest), uses the provided concept_vectors (from text embeddings)
# For layer i < 11, uses the CLS gradient from layer i+1
wrapper.dot_concept_vectors(text_embeddings, power=0)

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
                         text_prompts=prompts,
                         mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
                         normalize_each_map=True
                         )

# %%
# Aggregate maps across layer_indices
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"\nAggregated maps shape (from layers {layer_indices}): {maps_aggregated.shape}")

# %%
# Visualize aggregated maps
print("\n=== Visualizing aggregated attention map ===")
visualize(image, maps_aggregated, text_prompts=prompts, mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))

# %%
# Reset the wrapper for next use
wrapper.reset()
print("Wrapper reset for next use.")
