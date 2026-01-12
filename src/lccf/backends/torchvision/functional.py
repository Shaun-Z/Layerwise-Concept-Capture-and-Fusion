# src/lccf/backends/torchvision/functional.py
"""
Custom forward function for torchvision ViT EncoderBlock to capture attention weights.
This is used by TorchvisionGradWrapper to capture attention weights for gradient computation.
"""
import math
import torch
from torch import Tensor
from typing import Optional
from torch.nn.functional import _mha_shape_check, _canonical_mask, _none_or_dtype, _in_projection_packed, _check_key_padding_mask, pad, softmax, dropout, linear


def MultiheadAttention_forward_batch_first(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
    """
    Custom MultiheadAttention forward for batch_first=True that returns attention weights.
    Adapted from the OpenCLIP implementation for torchvision's batch_first format.
    
    Input/output format: (B, N, D) when batch_first=True
    """
    # For batch_first, we need to transpose to (N, B, D) for the MHA computation
    if self.batch_first:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
    
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    num_heads = self.num_heads
    head_dim = embed_dim // num_heads
    
    # Compute in-projection
    q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
    
    # Prep attention mask
    if attn_mask is not None:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
    
    # Handle key padding mask
    if key_padding_mask is not None:
        key_padding_mask = _canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=_none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
    
    # Reshape q, k, v for multihead attention and make them batch first
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
    
    # Handle key padding mask
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask
    
    # Compute attention
    q_scaled = q * math.sqrt(1.0 / float(head_dim))
    
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    
    dropout_p = self.dropout if self.training else 0.0
    if dropout_p > 0.0:
        attn_dropped = dropout(attn_output_weights, p=dropout_p)
    else:
        attn_dropped = attn_output_weights
    
    attn_output = torch.bmm(attn_dropped, v)
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
    attn_output = attn_output.view(tgt_len, bsz, embed_dim)
    
    # Transpose back to batch_first format
    if self.batch_first:
        attn_output = attn_output.transpose(0, 1)
    
    return attn_output, attn_output_weights


def EncoderBlock_forward(self, input: Tensor) -> Tensor:
    """
    Modified EncoderBlock forward that captures attention weights.
    
    This function replaces the default torchvision EncoderBlock.forward to enable 
    capturing attention weights for gradient-based interpretation methods.
    
    Key change: pass need_weights=True to self_attention and store the weights.
    
    Args:
        self: The EncoderBlock module instance
        input: Input tensor of shape (B, N, D)
        
    Returns:
        output: (B, N, D)
    """
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    # Changed: need_weights=True, average_attn_weights=False to get per-head weights
    x, attn_weights = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
    # Store attention weights on the block for later retrieval
    # attn_weights shape: (B*num_heads, N, N) from the custom MHA forward
    self._attn_weights = attn_weights
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y
