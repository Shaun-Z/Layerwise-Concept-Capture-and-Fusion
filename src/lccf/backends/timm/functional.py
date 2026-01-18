# src/lccf/backends/timm/functional.py
"""
Custom attention forward function for timm ViT models to return attention weights.
This is used by TimmGradWrapper to capture attention weights for gradient computation.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor


def CV_Pseudo_Attention_forward(
    self,
    x: Tensor,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Pseudo attention forward for CVWrapper that stops gradient flow through attention weights.
    
    This is a variant of Pseudo_Attention_forward specifically designed for TimmCVWrapper.
    It computes attention only for the CLS token query and stores attention weights for
    gradient accumulation, but the backward pass does NOT propagate gradients through
    the attention weights path - gradients only flow through V.
    
    The key technique is to make attention_weights a leaf tensor (detached from its inputs
    but with requires_grad=True). This allows:
    1. Gradients to accumulate on attention_weights (for visualization)
    2. Gradients NOT to propagate backward from attention_weights to q, k, x
    3. Gradients to flow through V to x (for input CLS gradient)
    
    Args:
        self: The Attention module instance
        x: Input tensor of shape (B, N, C)
        attn_mask: Optional attention mask
        
    Returns:
        output: (1, B, C) - only the CLS token output, transposed to match OpenCLIP format
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    # Compute attention weights manually (no fused attention)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
    
    if attn_mask is not None:
        attn = attn + attn_mask
    
    attn_weights = attn.softmax(dim=-1)  # (B, num_heads, N, N)
    
    # Only keep CLS token attention weights
    attn_weights_cls = attn_weights[:, :, :1, :]  # (B, num_heads, 1, N)
    
    # Reshape for gradient computation: (B*num_heads, 1, N)
    attn_weights_reshaped = attn_weights_cls.reshape(B * self.num_heads, 1, N)
    
    # CRITICAL: Create a leaf tensor for attention weights.
    # This detaches attention_weights from its computation graph (q, k, x),
    # making it a leaf node that can accumulate gradients but NOT propagate them backward.
    # This ensures gradients w.r.t. input CLS only flow through V, not through attention weights.
    attn_weights_leaf = attn_weights_reshaped.detach().requires_grad_(True)
    self._attn_weights = attn_weights_leaf
    
    attn_dropped = self.attn_drop(attn_weights_leaf)  # (B*num_heads, 1, N)
    
    # Reshape v for batched matmul: (B*num_heads, N, head_dim)
    v_reshaped = v.reshape(B * self.num_heads, N, self.head_dim)
    
    # Compute attn @ v - gradients flow to both attn_weights_leaf and v_reshaped
    # But since attn_weights_leaf is a leaf, gradients don't propagate further to q, k, x
    x = attn_dropped @ v_reshaped  # (B*num_heads, 1, head_dim)
    
    # Reshape back to (B, 1, C)
    # Note: attn_dim may differ from num_heads*head_dim in some timm versions
    # when using asymmetric attention dimensions. Fallback to default calculation.
    attn_dim = getattr(self, 'attn_dim', self.num_heads * self.head_dim)
    x = x.reshape(B, self.num_heads, 1, self.head_dim)
    x = x.transpose(1, 2).reshape(B, 1, attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    # Transpose to (1, B, C) to match OpenCLIP format (N, B, D)
    return x.transpose(0, 1)


def Pseudo_Attention_forward(
    self,
    x: Tensor,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Pseudo attention forward that only attends to the CLS token (first token).
    
    This function is used by TimmWrapper in pseudo mode to compute attention
    weights only for the CLS token query, matching the OpenCLIP pseudo wrapper behavior.
    
    Args:
        self: The Attention module instance
        x: Input tensor of shape (B, N, C)
        attn_mask: Optional attention mask
        
    Returns:
        output: (1, B, C) - only the CLS token output, transposed to match OpenCLIP format
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    # Compute attention weights manually (no fused attention)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
    
    if attn_mask is not None:
        attn = attn + attn_mask
    
    attn_weights = attn.softmax(dim=-1)  # (B, num_heads, N, N)
    
    # Only keep CLS token attention weights
    attn_weights_cls = attn_weights[:, :, :1, :]  # (B, num_heads, 1, N)
    
    # Reshape for gradient computation: (B*num_heads, 1, N)
    # IMPORTANT: We use this reshaped tensor for both storage AND computation
    # to ensure gradient flow
    attn_weights_reshaped = attn_weights_cls.reshape(B * self.num_heads, 1, N)
    self._attn_weights = attn_weights_reshaped
    
    attn = self.attn_drop(attn_weights_reshaped)  # (B*num_heads, 1, N)
    
    # Reshape v for batched matmul: (B*num_heads, N, head_dim)
    v_reshaped = v.reshape(B * self.num_heads, N, self.head_dim)
    x = attn @ v_reshaped  # (B*num_heads, 1, head_dim)
    
    # Reshape back to (B, 1, C)
    # Note: attn_dim may differ from num_heads*head_dim in some timm versions
    # when using asymmetric attention dimensions. Fallback to default calculation.
    attn_dim = getattr(self, 'attn_dim', self.num_heads * self.head_dim)
    x = x.reshape(B, self.num_heads, 1, self.head_dim)
    x = x.transpose(1, 2).reshape(B, 1, attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    # Transpose to (1, B, C) to match OpenCLIP format (N, B, D)
    return x.transpose(0, 1)


def Attention_forward(
    self,
    x: Tensor,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Modified attention forward that stores attention weights and returns output.
    
    This function replaces the default timm Attention.forward to enable 
    capturing attention weights for gradient-based interpretation methods.
    
    The attention weights are stored on the module as self._attn_weights
    for later retrieval by the hook.
    
    Args:
        self: The Attention module instance
        x: Input tensor of shape (B, N, C)
        attn_mask: Optional attention mask
        
    Returns:
        output: (B, N, C)
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    # Always compute attention weights manually (no fused attention)
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, N)
    
    if attn_mask is not None:
        attn = attn + attn_mask
    
    attn_weights = attn.softmax(dim=-1)  # (B, num_heads, N, N)
    
    # Store attention weights on the module for later retrieval
    # Store in original shape (B, num_heads, N, N) to maintain gradient connection
    self._attn_weights = attn_weights
    
    attn = self.attn_drop(attn_weights)
    x = attn @ v  # (B, num_heads, N, head_dim)
    
    # Compute attn_dim for compatibility with older timm versions
    attn_dim = getattr(self, 'attn_dim', self.num_heads * self.head_dim)
    x = x.transpose(1, 2).reshape(B, N, attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    
    return x
