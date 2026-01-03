
import pytest
import open_clip
from lccf.detect import detect_and_wrap
import torch

@pytest.fixture
def model():
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
    return model

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

@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0,11]),
                                ])
def test_hooks(model, batch_size, layer_indices):
    # Ensure that the wrapper works with the vision transformer architecture
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, batch_size, 512)

@pytest.mark.parametrize("batch_size, layer_indices, prompts", [
                                (10, [0,11],["a photo of a cat", "a photo of a dog", "a photo of a bird",
                                             "a photo of a car", "a photo of a tree", "a photo of a house",
                                             "a photo of a person", "a photo of a computer", "a photo of a phone",
                                             "a photo of a cup"]),
                                ])
def test_concept_vectors(model, batch_size, layer_indices, prompts):
    # Ensure that the wrapper works with the vision transformer architecture
    tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True)
    assert text_embeddings.shape == (len(prompts), 512)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)
    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, batch_size, 512)

    wrapper.dot_concept_vectors(text_embeddings)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 197, batch_size, len(prompts))

    