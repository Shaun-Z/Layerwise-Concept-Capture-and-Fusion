
import pytest

import torch
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from lccf.detect import detect_and_wrap


@pytest.fixture
def model():
    model = vit_b_16(weights=None)
    model.eval()
    return model

@pytest.fixture
def preprocess():
    # Standard ImageNet normalization for torchvision ViT
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def test_torchvision_wrapper(model):
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=[2, 5, 8])
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
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=layer_indices)
    output = wrapper(dummy_input)

    assert wrapper.hidden_dim == 768  # ViT-B-16 hidden dim
    assert output.shape == (batch_size, 1000)  # Classification output
    assert wrapper._requested_hook_indices == layer_indices


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0, 11]),
                                ])
def test_hooks(model, batch_size, layer_indices):
    # Ensure that the wrapper works with the vision transformer architecture
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=layer_indices)
    output = wrapper(dummy_input)

    # block_ins is transposed to (N, B, D) format in torchvision wrapper
    assert len(wrapper.block_ins) == len(layer_indices)
    assert torch.stack(wrapper.block_ins, dim=0).shape == (len(layer_indices), 197, batch_size, 768)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (10, [0, 11], 10),
                                (5, [0, 5, 11], 3),
                                ])
def test_concept_vectors(model, batch_size, layer_indices, num_concepts):
    # Ensure that the wrapper works with the vision transformer architecture
    # For torchvision, we use random concept vectors in the hidden_dim space
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=layer_indices)
    output = wrapper(dummy_input)
    assert len(wrapper.block_ins) == len(layer_indices)

    wrapper.dot_concept_vectors(concept_vectors)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, batch_size, num_concepts)


@pytest.mark.parametrize("layer_indices, num_concepts", [
                                ([0, 11], 2),
                                ([0, 3, 6, 9, 11], 5),
                                ])
def test_aggregate_maps(model, layer_indices, num_concepts):
    # Test aggregation of layerwise maps
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=layer_indices)
    
    output = wrapper(dummy_input)
    assert len(wrapper.block_ins) == len(layer_indices)
    
    wrapper.dot_concept_vectors(concept_vectors)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, 1, num_concepts)
    
    maps = wrapper.aggregate_layerwise_maps()
    # Aggregated maps should be (B, num_concepts, H*patch_size, W*patch_size)
    # With patch_size=16 and grid_size=14, output should be (1, num_concepts, 224, 224)
    assert maps.shape == (1, num_concepts, 224, 224)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (10, [0, 11], 8),
                                ])
def test_pseudo_wrapper(model, batch_size, layer_indices, num_concepts):
    # Generate random concept vectors in the hidden_dim space
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='fast', layer_indices=layer_indices)
    
    output = wrapper(dummy_input)
    assert len(wrapper.block_ins) == len(layer_indices)
    wrapper.dot_concept_vectors(concept_vectors)  # Use concept_vectors
    sim_bms = torch.stack(wrapper.sim_bms, dim=0)
    assert sim_bms.shape == (len(layer_indices), batch_size, num_concepts)
    grads = torch.stack(wrapper.grads, dim=0)
    assert grads.shape == (len(layer_indices), num_concepts, batch_size, wrapper.num_heads, 1, 197)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 14, 14, batch_size, num_concepts)


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                (3, [])
                                ])
def test_grad_wrapper(model, batch_size, layer_indices):
    # Test that we can extract features using the gradient wrapper
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='standard', layer_indices=layer_indices)
    output = wrapper(dummy_input)

    if wrapper.attn_weights:
        attn_weights = torch.stack(wrapper.attn_weights, dim=0)
        assert attn_weights.shape == (len(layer_indices), batch_size * 12, 197, 197)
    if wrapper.block_outputs:
        block_outputs = torch.stack(wrapper.block_outputs, dim=0)
        assert block_outputs.shape == (len(layer_indices), batch_size, 197, 768)

    assert wrapper.hidden_dim == 768  # ViT-B-16 hidden dim
    assert output.shape == (batch_size, 1000)  # Classification output
    assert wrapper._requested_hook_indices == layer_indices


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (10, [0, 11], 4),
                                ])
def test_concept_vectors_grad_wrapper(model, batch_size, layer_indices, num_concepts):
    # Ensure that the gradient wrapper works with concept vectors
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1).detach()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', mode='standard', layer_indices=layer_indices)
    output = wrapper(dummy_input)
    assert output.shape == (batch_size, 1000)
    attn_weights = torch.stack(wrapper.attn_weights, dim=0)
    assert attn_weights.shape == (len(layer_indices), batch_size * wrapper.num_heads, 197, 197)
    wrapper.dot_concept_vectors(concept_vectors)
    grads = torch.stack(wrapper.grads, dim=0)
    assert grads.shape == (len(layer_indices), num_concepts, batch_size, wrapper.num_heads, 197, 197)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 14, 14, batch_size, num_concepts)
    exp_map = wrapper.aggregate_layerwise_maps()
    assert exp_map.shape == (batch_size, num_concepts, 224, 224)


# =============================================================================
# TorchvisionFCVWrapper Tests
# =============================================================================

from lccf.backends.torchvision.wrapper import TorchvisionFCVWrapper


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                ])
def test_torchvision_fcv_wrapper(model, batch_size, layer_indices):
    # Test that we can create a TorchvisionFCVWrapper
    wrapper = TorchvisionFCVWrapper(model, layer_indices=layer_indices)
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0, 11]),
                                (5, [0, 5, 11]),
                                ])
def test_torchvision_fcv_wrapper_forward(model, batch_size, layer_indices):
    # Test that we can extract features from a dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TorchvisionFCVWrapper(model, layer_indices=layer_indices)
    output = wrapper(dummy_input)

    assert wrapper.hidden_dim == 768  # ViT-B-16 hidden dim
    assert output.shape == (batch_size, 1000)  # Classification output
    # All 12 layers have block_ins (ViT-B-16 has 12 blocks)
    assert len(wrapper.block_ins) == 12


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (5, [0, 11]),
                                (3, [0, 5, 11]),
                                ])
def test_torchvision_fcv_wrapper_dot_concept_vectors(model, batch_size, layer_indices):
    # Test that dot_concept_vectors works and stores the expected outputs
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TorchvisionFCVWrapper(model, layer_indices=layer_indices)
    output = wrapper(dummy_input)
    
    # All 12 layers have block_ins
    assert len(wrapper.block_ins) == 12
    
    # Extract concept vector from classifier head
    num_concepts = 1
    concept_vectors = model.heads[0].weight[281].unsqueeze(0).detach()  # tabby cat class
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    wrapper.dot_concept_vectors(concept_vectors)
    
    # Check that attention gradients are stored for ALL 12 layers
    assert len(wrapper.attn_grads) == 12
    # Check shape of attention gradients: (M, B, num_heads, N, N)
    for attn_grad in wrapper.attn_grads:
        assert attn_grad.shape == (num_concepts, batch_size, wrapper.num_heads, 197, 197)
    
    # Check that token gradients are stored for ALL 12 layers
    assert len(wrapper.token_grads) == 12
    # Check shape of token gradients: (B, M, N, D)
    for token_grad in wrapper.token_grads:
        assert token_grad.shape == (batch_size, num_concepts, 197, 768)
    
    # Check that maps are stored only for layers in layer_indices
    assert len(wrapper.maps) == len(layer_indices)
    for expl_map in wrapper.maps:
        assert expl_map.shape == (14, 14, batch_size, num_concepts)
    
    # Check that sim_bms are stored only for layers in layer_indices
    assert len(wrapper.sim_bms) == len(layer_indices)
    for sim_bm in wrapper.sim_bms:
        assert sim_bm.shape == (batch_size, num_concepts)


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (5, [0, 11]),
                                (3, [0, 3, 6, 9, 11]),
                                ])
def test_torchvision_fcv_wrapper_aggregate_maps(model, batch_size, layer_indices):
    # Test aggregation of layerwise maps
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TorchvisionFCVWrapper(model, layer_indices=layer_indices)
    
    output = wrapper(dummy_input)
    # All 12 layers have block_ins
    assert len(wrapper.block_ins) == 12
    
    # Extract concept vector from classifier head
    num_concepts = 1
    concept_vectors = model.heads[0].weight[281].unsqueeze(0).detach()  # tabby cat class
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    wrapper.dot_concept_vectors(concept_vectors)
    # Maps are stored only for layers in layer_indices
    assert len(wrapper.maps) == len(layer_indices)
    
    maps = wrapper.aggregate_layerwise_maps()
    # Aggregated maps should be (B, M, H*patch_size, W*patch_size)
    assert maps.shape == (batch_size, num_concepts, 224, 224)
