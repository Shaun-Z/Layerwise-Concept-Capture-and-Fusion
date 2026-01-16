
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
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=False, layer_indices=layer_indices)
    
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
    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, batch_size, 512)

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
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=True, layer_indices=layer_indices)
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
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=True, layer_indices=layer_indices)
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


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts_per_layer", [
                                (5, [0, 11], [3, 4]),
                                (3, [0, 5, 11], [2, 5, 3]),
                                ])
def test_concept_vectors_list(model, batch_size, layer_indices, num_concepts_per_layer):
    """Test that concept_vectors can be passed as a list with one tensor per layer."""
    # Create a list of concept vectors, one for each layer with different number of concepts
    # OpenCLIP uses 512-dim output space
    concept_vectors_list = [
        torch.nn.functional.normalize(torch.randn(n, 512), dim=-1)
        for n in num_concepts_per_layer
    ]

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=False, layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)
    assert len(wrapper.block_ins) == len(layer_indices)

    wrapper.dot_concept_vectors(concept_vectors_list)
    
    # Each layer should have a map with the corresponding number of concepts
    assert len(wrapper.maps) == len(layer_indices)
    for i, (expl_map, num_concepts) in enumerate(zip(wrapper.maps, num_concepts_per_layer)):
        assert expl_map.shape == (14, 14, batch_size, num_concepts)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts_per_layer", [
                                (5, [0, 11], [3, 4]),
                                ])
def test_concept_vectors_list_grad_wrapper(model, batch_size, layer_indices, num_concepts_per_layer):
    """Test that concept_vectors can be passed as a list for grad wrapper."""
    # Create a list of concept vectors, one for each layer
    # OpenCLIP uses 512-dim output space
    concept_vectors_list = [
        torch.nn.functional.normalize(torch.randn(n, 512), dim=-1).detach()
        for n in num_concepts_per_layer
    ]

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=True, layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)
    
    wrapper.dot_concept_vectors(concept_vectors_list)
    
    # Each layer should have a map with the corresponding number of concepts
    assert len(wrapper.maps) == len(layer_indices)
    for i, (expl_map, num_concepts) in enumerate(zip(wrapper.maps, num_concepts_per_layer)):
        assert expl_map.shape == (14, 14, batch_size, num_concepts)


def test_concept_vectors_list_length_mismatch(model):
    """Test that passing a list with wrong length raises ValueError."""
    layer_indices = [0, 5, 11]
    # OpenCLIP uses 512-dim output space
    concept_vectors_list = [
        torch.nn.functional.normalize(torch.randn(3, 512), dim=-1),
        torch.nn.functional.normalize(torch.randn(4, 512), dim=-1),
        # Missing third element - should raise ValueError
    ]

    dummy_input = torch.randn(2, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', use_grad=False, layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    with pytest.raises(ValueError, match="Length of concept_vectors list"):
        wrapper.dot_concept_vectors(concept_vectors_list)
    