# src/lccf/backends/timm/functional.py
"""
Custom attention forward function for timm ViT models to return attention weights.
This is used by TimmGradWrapper to capture attention weights for gradient computation.
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor


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
