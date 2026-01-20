
import pytest

import requests
from PIL import Image

import torch
import open_clip

from lccf.detect import detect_and_wrap


@pytest.fixture
def model():
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    model.eval()
    return model

@pytest.fixture
def preprocess():
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    return preprocess

@pytest.fixture
def tokenizer():
    return open_clip.get_tokenizer(model_name='ViT-B-16')

def test_openclip_wrapper(model):
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=[2, 5, 8])
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)

@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                (3, [])
                                ])
def test_feature_extraction(model, batch_size, layer_indices):
    # Test that we can extract features from a dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    assert wrapper.visual.output_dim == 512  # ViT-B-16 output dim
    assert features.shape == (batch_size, 512)
    assert wrapper._requested_hook_indices == layer_indices

@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (10, [0,11],["a photo of a cat", "a photo of a dog", "a photo of a bird",
                                             "a photo of a car", "a photo of a tree", "a photo of a house",
                                             "a photo of a person", "a photo of a computer"]),
                                ])
def test_pseudo_wrapper(model, tokenizer, batch_size, layer_indices, prompts):
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True)
    assert text_embeddings.shape == (len(prompts), 512)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', mode="fast", layer_indices=layer_indices)
    
    features = wrapper.encode_image(dummy_input)
    assert len(wrapper.block_ins) == len(layer_indices)
    wrapper.dot_concept_vectors(text_embeddings)  # Use text_embeddings as concept vectors
    sim_bms = torch.stack(wrapper.sim_bms, dim=0)
    assert sim_bms.shape == (len(layer_indices), batch_size, len(prompts))
    grads = torch.stack(wrapper.grads, dim=0)
    assert grads.shape == (len(layer_indices), len(prompts), batch_size, wrapper.num_heads, 1, 197)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 14, 14, batch_size, len(prompts))

@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (10, [0,11],["a photo of a cat", "a photo of a dog", "a photo of a bird",
                                             "a photo of a car", "a photo of a tree", "a photo of a house",
                                             "a photo of a person", "a photo of a computer", "a photo of a phone",
                                             "a photo of a cup"]),
                                ])
def test_concept_vectors(model, tokenizer, batch_size, layer_indices, prompts):
    # Ensure that the wrapper works with the vision transformer architecture
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True)
    assert text_embeddings.shape == (len(prompts), 512)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    wrapper.dot_concept_vectors(text_embeddings)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, batch_size, len(prompts))

@pytest.mark.parametrize("layer_indices, prompts", [
                                ([0,11],["a photo of a cat",
                                         "a photo of a remote controller"]),
                                ])
def test_single_image(model, preprocess, tokenizer, layer_indices, prompts):
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True)
    assert text_embeddings.shape == (len(prompts), 512)

    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    device = wrapper._get_device_for_call()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
    assert image.shape == (1, 3, 224, 224)

    features = wrapper.encode_image(image)
    wrapper.dot_concept_vectors(text_embeddings)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, 1, len(prompts))
    maps = wrapper.aggregate_layerwise_maps()
    assert maps.shape == (image.shape[0], len(prompts), 224, 224)

@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                (3, [])
                                ])
def test_grad_wrapper(model, batch_size, layer_indices):
    # Test that we can extract features from a dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', mode="standard", layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    if wrapper.attn_weights:
        attn_weights = torch.stack(wrapper.attn_weights, dim=0)
        assert attn_weights.shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)
    if wrapper.block_outputs:
        block_outputs = torch.stack(wrapper.block_outputs, dim=0)
        assert block_outputs.shape == (len(layer_indices), 197, batch_size, 768)

    assert wrapper.visual.output_dim == 512  # ViT-B-16 output dim
    assert features.shape == (batch_size, 512)
    assert wrapper._requested_hook_indices == layer_indices


@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (10, [0,11],["a photo of a cat", "a photo of a dog", "a photo of a bird",
                                             "a photo of a car"]),
                                ])
def test_concept_vectors_grad_wrapper(model, tokenizer, batch_size, layer_indices, prompts):
    # Ensure that the wrapper works with the vision transformer architecture
    text = tokenizer(prompts)

    text_embeddings = model.encode_text(text, normalize=True).detach()
    assert text_embeddings.shape == (len(prompts), 512)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', mode="standard", layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)
    assert features.shape == (batch_size, 512)
    attn_weights = torch.stack(wrapper.attn_weights, dim=0)
    assert attn_weights.shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)
    wrapper.dot_concept_vectors(text_embeddings)
    grads = torch.stack(wrapper.grads, dim=0)
    assert grads.shape == (len(layer_indices), len(prompts), batch_size, wrapper.num_heads, 197, 197)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 14, 14, batch_size, len(prompts))
    exp_map = wrapper.aggregate_layerwise_maps()
    assert exp_map.shape == (batch_size, len(prompts), 224, 224)


# =============================================================================
# OpenCLIPFCVWrapper Tests
# =============================================================================

from lccf.backends.openclip.wrapper import OpenCLIPFCVWrapper


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                ])
def test_openclip_fcv_wrapper(model, batch_size, layer_indices):
    # Test that we can create an OpenCLIPFCVWrapper
    wrapper = OpenCLIPFCVWrapper(model, layer_indices=layer_indices)
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0, 11]),
                                (5, [0, 5, 11]),
                                ])
def test_openclip_fcv_wrapper_forward(model, batch_size, layer_indices):
    # Test that we can extract features from a dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = OpenCLIPFCVWrapper(model, layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    assert wrapper.visual.output_dim == 512  # ViT-B-16 output dim
    assert features.shape == (batch_size, 512)
    # All 12 layers have block_ins (ViT-B-16 has 12 blocks)
    assert len(wrapper.block_ins) == 12


@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (5, [0, 11], ["a photo of a cat", "a photo of a dog"]),
                                (3, [0, 5, 11], ["a photo of a cat"]),
                                ])
def test_openclip_fcv_wrapper_dot_concept_vectors(model, tokenizer, batch_size, layer_indices, prompts):
    # Test that dot_concept_vectors works and stores the expected outputs
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True).detach()
    num_concepts = len(prompts)
    
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = OpenCLIPFCVWrapper(model, layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)
    
    # All 12 layers have block_ins
    assert len(wrapper.block_ins) == 12
    
    wrapper.dot_concept_vectors(text_embeddings)
    
    # Check that attention gradients are stored for ALL 12 layers
    assert len(wrapper.attn_grads) == 12
    # Check shape of attention gradients: (M, B, num_heads, N, N)
    for attn_grad in wrapper.attn_grads:
        assert attn_grad.shape == (num_concepts, batch_size, wrapper.num_heads, 197, 197)
    
    # Check that token gradients are stored for ALL 12 layers
    assert len(wrapper.token_grads) == 12
    # Check shape of token gradients: (N, B, M, D) in OpenCLIP format
    for token_grad in wrapper.token_grads:
        assert token_grad.shape == (197, batch_size, num_concepts, 768)
    
    # Check that maps are stored only for layers in layer_indices
    assert len(wrapper.maps) == len(layer_indices)
    for expl_map in wrapper.maps:
        assert expl_map.shape == (14, 14, batch_size, num_concepts)
    
    # Check that sim_bms are stored only for layers in layer_indices
    assert len(wrapper.sim_bms) == len(layer_indices)
    for sim_bm in wrapper.sim_bms:
        assert sim_bm.shape == (batch_size, num_concepts)


@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (5, [0, 11], ["a photo of a cat", "a photo of a dog"]),
                                (3, [0, 3, 6, 9, 11], ["a photo of a cat"]),
                                ])
def test_openclip_fcv_wrapper_aggregate_maps(model, tokenizer, batch_size, layer_indices, prompts):
    # Test aggregation of layerwise maps
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True).detach()
    num_concepts = len(prompts)
    
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = OpenCLIPFCVWrapper(model, layer_indices=layer_indices)
    
    features = wrapper.encode_image(dummy_input)
    # All 12 layers have block_ins
    assert len(wrapper.block_ins) == 12
    
    wrapper.dot_concept_vectors(text_embeddings)
    # Maps are stored only for layers in layer_indices
    assert len(wrapper.maps) == len(layer_indices)
    
    maps = wrapper.aggregate_layerwise_maps()
    # Aggregated maps should be (B, M, H*patch_size, W*patch_size)
    assert maps.shape == (batch_size, num_concepts, 224, 224)