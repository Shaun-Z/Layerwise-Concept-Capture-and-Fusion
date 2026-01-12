# %%
import requests
from PIL import Image
import torch
import open_clip
from lccf.detect import detect_and_wrap, wrap_clip_preprocess
from lccf.utils import visualize, visualize_layerwise_maps
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device=device)
model.eval()
preprocess = wrap_clip_preprocess(preprocess, image_size=224)
tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')

# %%
layer_indices = [0,1,2,3,4,5,6,7,8,9,10,11]

# %%
prompts = ["a photo of cats", " a photo of a red sofa", "a photo of a remote control"]

# %%
wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)

# %%
text = tokenizer(prompts).to(device)
text_embeddings = model.encode_text(text, normalize=True)

# %%
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
features = wrapper.encode_image(image)

# %%
wrapper.dot_concept_vectors(text_embeddings)

# %%
maps = torch.stack(wrapper.maps, dim=0)  # (num_layers, H, W, B, num_concepts)

# %%
visualize_layerwise_maps(image, wrapper.maps, sim_bms=wrapper.sim_bms, text_prompts=prompts, mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))

# %%
maps_aggregated = wrapper.aggregate_layerwise_maps()

# %%
visualize(image, maps_aggregated, text_prompts=prompts, mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))

# %%
wrapper.reset()  # Clear stored results and maps for next use
