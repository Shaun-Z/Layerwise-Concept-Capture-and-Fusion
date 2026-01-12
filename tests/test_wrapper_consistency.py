"""
Test that non-gradient wrappers produce the same results as gradient wrappers.

This test validates that the modified non-gradient wrappers (TimmWrapper, TorchvisionWrapper)
now use the same gradient-based approach as the GradWrappers and produce identical results.
"""

import pytest
import copy
import torch
import torch.nn.functional as F

import timm
from torchvision.models import vit_b_16

from lccf.detect import detect_and_wrap


def create_timm_model():
    """Create a fresh timm model."""
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.eval()
    return model


def create_torchvision_model():
    """Create a fresh torchvision model."""
    model = vit_b_16(weights=None)
    model.eval()
    return model


@pytest.fixture
def timm_model():
    return create_timm_model()


@pytest.fixture
def torchvision_model():
    return create_torchvision_model()


class TestTimmConsistency:
    """Test that TimmWrapper produces the same results as TimmGradWrapper."""
    
    @pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
        (2, [0, 11], 3),
        (3, [5, 8, 11], 2),
    ])
    def test_map_consistency(self, batch_size, layer_indices, num_concepts):
        """Test that both wrappers produce identical maps when using the same model weights."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a base model and get its state dict
        base_model = create_timm_model()
        state_dict = base_model.state_dict()
        
        # Create input and concept vectors
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create two models with the SAME weights
        model_no_grad = create_timm_model()
        model_no_grad.load_state_dict(state_dict)
        
        model_grad = create_timm_model()
        model_grad.load_state_dict(state_dict)
        
        # Non-grad wrapper (now uses same approach as grad wrapper)
        wrapper_no_grad = detect_and_wrap(model_no_grad, prefer='timm', async_compute=True, layer_indices=layer_indices)
        _ = wrapper_no_grad.forward_features(dummy_input.clone())
        wrapper_no_grad.dot_concept_vectors(concept_vectors.clone())
        maps_no_grad = wrapper_no_grad.aggregate_layerwise_maps()
        
        # Grad wrapper (on separate model with same weights)
        wrapper_grad = detect_and_wrap(model_grad, prefer='timm', async_compute=False, layer_indices=layer_indices)
        _ = wrapper_grad.forward_features(dummy_input.clone())
        wrapper_grad.dot_concept_vectors(concept_vectors.clone())
        maps_grad = wrapper_grad.aggregate_layerwise_maps()
        
        # Compare shapes
        assert maps_no_grad.shape == maps_grad.shape, f"Shape mismatch: {maps_no_grad.shape} vs {maps_grad.shape}"
        
        # Compare values
        assert torch.allclose(maps_no_grad, maps_grad, rtol=1e-4, atol=1e-4), \
            f"Maps do not match! Max diff: {(maps_no_grad - maps_grad).abs().max()}"
    
    @pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
        (2, [0, 11], 3),
    ])
    def test_grad_shape_consistency(self, batch_size, layer_indices, num_concepts):
        """Test that gradient shapes match between wrappers."""
        torch.manual_seed(42)
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create base model and get state dict
        base_model = create_timm_model()
        state_dict = base_model.state_dict()
        
        # Create fresh models with same weights
        model_no_grad = create_timm_model()
        model_no_grad.load_state_dict(state_dict)
        model_grad = create_timm_model()
        model_grad.load_state_dict(state_dict)
        
        # Non-grad wrapper
        wrapper_no_grad = detect_and_wrap(model_no_grad, prefer='timm', async_compute=True, layer_indices=layer_indices)
        _ = wrapper_no_grad.forward_features(dummy_input.clone())
        wrapper_no_grad.dot_concept_vectors(concept_vectors.clone())
        
        # Grad wrapper
        wrapper_grad = detect_and_wrap(model_grad, prefer='timm', async_compute=False, layer_indices=layer_indices)
        _ = wrapper_grad.forward_features(dummy_input.clone())
        wrapper_grad.dot_concept_vectors(concept_vectors.clone())
        
        # Compare gradient shapes
        for i, (grad_no_grad, grad_grad) in enumerate(zip(wrapper_no_grad.grads, wrapper_grad.grads)):
            assert grad_no_grad.shape == grad_grad.shape, \
                f"Gradient shape mismatch at layer {i}: {grad_no_grad.shape} vs {grad_grad.shape}"
    
    def test_set_concept_vectors_before_forward(self, timm_model):
        """Test that setting concept vectors before forward produces correct results."""
        torch.manual_seed(42)
        
        batch_size = 2
        layer_indices = [0, 11]
        num_concepts = 3
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create base model and get state dict
        base_model = create_timm_model()
        state_dict = base_model.state_dict()
        
        # Create fresh models with same weights
        model1 = create_timm_model()
        model1.load_state_dict(state_dict)
        model2 = create_timm_model()
        model2.load_state_dict(state_dict)
        
        # Method 1: Set concept vectors before forward (new feature)
        wrapper1 = detect_and_wrap(model1, prefer='timm', async_compute=True, layer_indices=layer_indices)
        wrapper1.set_concept_vectors(concept_vectors.clone())
        _ = wrapper1.forward_features(dummy_input.clone())
        # Maps should already be computed during forward
        assert len(wrapper1.maps) == len(layer_indices), "Maps should be computed during forward"
        maps1 = wrapper1.aggregate_layerwise_maps()
        
        # Method 2: Traditional approach - call dot_concept_vectors after forward
        wrapper2 = detect_and_wrap(model2, prefer='timm', async_compute=True, layer_indices=layer_indices)
        _ = wrapper2.forward_features(dummy_input.clone())
        wrapper2.dot_concept_vectors(concept_vectors.clone())
        maps2 = wrapper2.aggregate_layerwise_maps()
        
        # Both methods should produce the same results
        assert torch.allclose(maps1, maps2, rtol=1e-4, atol=1e-4), \
            f"Maps from set_concept_vectors don't match post-forward dot_concept_vectors"


class TestTorchvisionConsistency:
    """Test that TorchvisionWrapper produces the same results as TorchvisionGradWrapper."""
    
    @pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
        (2, [0, 11], 3),
        (3, [5, 8, 11], 2),
    ])
    def test_map_consistency(self, batch_size, layer_indices, num_concepts):
        """Test that both wrappers produce identical maps when using the same model weights."""
        torch.manual_seed(42)
        
        # Create base model and get state dict
        base_model = create_torchvision_model()
        state_dict = base_model.state_dict()
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create fresh models with same weights
        model_no_grad = create_torchvision_model()
        model_no_grad.load_state_dict(state_dict)
        model_grad = create_torchvision_model()
        model_grad.load_state_dict(state_dict)
        
        # Non-grad wrapper
        wrapper_no_grad = detect_and_wrap(model_no_grad, prefer='torchvision', async_compute=True, layer_indices=layer_indices)
        _ = wrapper_no_grad(dummy_input.clone())
        wrapper_no_grad.dot_concept_vectors(concept_vectors.clone())
        maps_no_grad = wrapper_no_grad.aggregate_layerwise_maps()
        
        # Grad wrapper
        wrapper_grad = detect_and_wrap(model_grad, prefer='torchvision', async_compute=False, layer_indices=layer_indices)
        _ = wrapper_grad(dummy_input.clone())
        wrapper_grad.dot_concept_vectors(concept_vectors.clone())
        maps_grad = wrapper_grad.aggregate_layerwise_maps()
        
        # Compare shapes
        assert maps_no_grad.shape == maps_grad.shape, f"Shape mismatch: {maps_no_grad.shape} vs {maps_grad.shape}"
        
        # Compare values
        assert torch.allclose(maps_no_grad, maps_grad, rtol=1e-4, atol=1e-4), \
            f"Maps do not match! Max diff: {(maps_no_grad - maps_grad).abs().max()}"
    
    @pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
        (2, [0, 11], 3),
    ])
    def test_grad_shape_consistency(self, batch_size, layer_indices, num_concepts):
        """Test that gradient shapes match between wrappers."""
        torch.manual_seed(42)
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create base model and get state dict
        base_model = create_torchvision_model()
        state_dict = base_model.state_dict()
        
        # Create fresh models with same weights
        model_no_grad = create_torchvision_model()
        model_no_grad.load_state_dict(state_dict)
        model_grad = create_torchvision_model()
        model_grad.load_state_dict(state_dict)
        
        # Non-grad wrapper
        wrapper_no_grad = detect_and_wrap(model_no_grad, prefer='torchvision', async_compute=True, layer_indices=layer_indices)
        _ = wrapper_no_grad(dummy_input.clone())
        wrapper_no_grad.dot_concept_vectors(concept_vectors.clone())
        
        # Grad wrapper
        wrapper_grad = detect_and_wrap(model_grad, prefer='torchvision', async_compute=False, layer_indices=layer_indices)
        _ = wrapper_grad(dummy_input.clone())
        wrapper_grad.dot_concept_vectors(concept_vectors.clone())
        
        # Compare gradient shapes
        for i, (grad_no_grad, grad_grad) in enumerate(zip(wrapper_no_grad.grads, wrapper_grad.grads)):
            assert grad_no_grad.shape == grad_grad.shape, \
                f"Gradient shape mismatch at layer {i}: {grad_no_grad.shape} vs {grad_grad.shape}"
    
    def test_set_concept_vectors_before_forward(self, torchvision_model):
        """Test that setting concept vectors before forward produces correct results."""
        torch.manual_seed(42)
        
        batch_size = 2
        layer_indices = [0, 11]
        num_concepts = 3
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        concept_vectors = torch.randn(num_concepts, 768)
        concept_vectors = F.normalize(concept_vectors, dim=-1).detach()
        
        # Create base model and get state dict
        base_model = create_torchvision_model()
        state_dict = base_model.state_dict()
        
        # Create fresh models with same weights
        model1 = create_torchvision_model()
        model1.load_state_dict(state_dict)
        model2 = create_torchvision_model()
        model2.load_state_dict(state_dict)
        
        # Method 1: Set concept vectors before forward
        wrapper1 = detect_and_wrap(model1, prefer='torchvision', async_compute=True, layer_indices=layer_indices)
        wrapper1.set_concept_vectors(concept_vectors.clone())
        _ = wrapper1(dummy_input.clone())
        assert len(wrapper1.maps) == len(layer_indices), "Maps should be computed during forward"
        maps1 = wrapper1.aggregate_layerwise_maps()
        
        # Method 2: Traditional approach
        wrapper2 = detect_and_wrap(model2, prefer='torchvision', async_compute=True, layer_indices=layer_indices)
        _ = wrapper2(dummy_input.clone())
        wrapper2.dot_concept_vectors(concept_vectors.clone())
        maps2 = wrapper2.aggregate_layerwise_maps()
        
        # Both methods should produce the same results
        assert torch.allclose(maps1, maps2, rtol=1e-4, atol=1e-4), \
            f"Maps from set_concept_vectors don't match post-forward dot_concept_vectors"
