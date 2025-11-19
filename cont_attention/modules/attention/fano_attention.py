"""
Fourier Attention Neural Operator (FANO) - Continuum Attention Implementation
Adapted from "Continuum Attention for Neural Operators" (Calvello et al., 2024)

This module implements patch-based continuum attention where:
- Q/K/V projections are computed via Fourier spectral convolutions
- Attention operates between patches (not tokens), preserving spatial structure
- Scale factor accounts for spatial dimensions: sqrt(im_size^4)

Key classes:
- SpectralConv2d_Attention: Fourier-based Q/K/V projection per head
- ScaledDotProductAttention_Conv: Continuum attention mechanism (patch-to-patch)
- MultiheadAttention_Conv: Multi-head wrapper
- TransformerEncoderLayer_Conv: Complete FANO encoder layer
- TransformerEncoder_Operator: Stack of FANO layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TransformerEncoder_Operator(nn.Module):
    """
    Stack of encoder layers that applies multiple FANO layers sequentially.
    
    Args:
        encoder_layer: A TransformerEncoderLayer_Conv instance
        num_layers: Number of layers to stack
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder_Operator, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, num_patches, patch_size, patch_size, d_model)
            mask: Optional attention mask
        Returns:
            x: (batch, num_patches, patch_size, patch_size, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class SpectralConv2d_Attention(nn.Module):
    """
    2D Fourier spectral convolution for attention Q/K/V projection.
    
    Performs: FFT → multiply learnable complex weights on low-freq modes → iFFT
    This creates smooth, continuous Q/K/V projections suitable for PDEs.
    
    Args:
        in_channels: Input channel dimension (typically d_model)
        out_channels: Output channel dimension (typically d_k per head)
        modes1: Number of Fourier modes in first spatial dimension
        modes2: Number of Fourier modes in second spatial dimension
        nhead: Number of attention heads (separate weights per head)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, nhead):
        super(SpectralConv2d_Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = nhead
        self.scale = (1 / (in_channels * out_channels))
        
        # Learnable complex weights for Fourier domain multiplication
        # Shape: (in_channels, out_channels, modes1, modes2, nhead)
        # Note: Use torch.view_as_real/view_as_complex for DDP training
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        """
        Complex multiplication in frequency domain.
        
        Args:
            input: (batch, num_patches, in_channel, x, y) - Fourier coefficients
            weights: (in_channel, out_channel, x, y, nhead) - Learnable weights
        Returns:
            (batch, num_patches, out_channel, x, y, nhead)
        """
        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weights)

    def forward(self, x):
        """
        Forward pass: FFT → multiply modes → iFFT
        
        Args:
            x: (batch, num_patches, patch_size, patch_size, d_model)
        Returns:
            (batch, num_patches, patch_size, patch_size, d_k, nhead)
        """
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        
        # Permute to (batch, num_patches, d_model, x, y) for FFT
        x = torch.permute(x, (0, 1, 4, 2, 3))
        
        # Compute Fourier coefficients (real-to-complex FFT)
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        # Only keep low-frequency modes for efficiency and smoothness
        out_ft = torch.zeros(
            batchsize, num_patches, self.out_channels,
            x.size(-2), x.size(-1) // 2 + 1, self.nhead,
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply learned weights to low-frequency modes (start and end of spectrum)
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)
        
        # Return to physical space (inverse FFT)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-3, -2))
        
        # Permute back to (batch, num_patches, x, y, d_k, nhead)
        x = torch.permute(x, (0, 1, 3, 4, 2, 5))
        
        return x


class MultiheadAttention_Conv(nn.Module):
    """
    Multi-head attention using Fourier convolutions for Q/K/V projection.
    
    This combines SpectralConv2d_Attention for smooth Q/K/V projections
    with continuum attention computation (ScaledDotProductAttention_Conv).
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        modes1: Fourier modes in first dimension
        modes2: Fourier modes in second dimension
        im_size: Image/patch size (used for attention scaling)
        dropout: Dropout probability
    """
    def __init__(self, d_model, nhead, modes1, modes2, im_size, dropout=0.1):
        super(MultiheadAttention_Conv, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead  # Dimension per head
        self.im_size = im_size

        # Fourier-based Q/K/V operators (separate spectral conv per head)
        self.query_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        # Scaled dot-product attention (the continuum attention mechanism)
        self.scaled_dot_product_attention = ScaledDotProductAttention_Conv(d_model, im_size, dropout=dropout)

        # Output projection
        self.out_linear = nn.Linear(nhead * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        Forward pass of Fourier multi-head attention.
        
        Args:
            x: (batch, num_patches, patch_size, patch_size, d_model)
            key_padding_mask: Optional mask for padding tokens
        Returns:
            (batch, num_patches, patch_size, patch_size, d_model)
        """
        batch, num_patches, patch_size, patch_size_2, d_model = x.size()
        
        # Project to Q/K/V using Fourier convolutions
        # Output: (batch, nhead, num_patches, x, y, d_k)
        query = self.query_operator(x).permute(0, 5, 1, 2, 3, 4)
        key = self.key_operator(x).permute(0, 5, 1, 2, 3, 4)
        value = self.value_operator(x).permute(0, 5, 1, 2, 3, 4)
        
        # Apply scaled dot-product attention (continuum attention mechanism)
        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        # Combine heads and project
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size, -1)
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        
        return output


class ScaledDotProductAttention_Conv(nn.Module):
    """
    THE CONTINUUM ATTENTION MECHANISM
    
    Computes patch-to-patch attention where attention scores account for
    the full spatial structure within each patch.
    
    Key innovation: Scale factor is sqrt(im_size^4) to account for:
    - Spatial dimensions: patch_size × patch_size 
    - Feature dimensions: d_k
    Total scale: sqrt(patch_size^2 * patch_size^2) = im_size^2
    
    Args:
        d_model: Model dimension (used for head dimension calculation)
        im_size: Image/patch size (determines scale factor)
        dropout: Dropout probability
    """
    def __init__(self, d_model, im_size, dropout=0.1):
        super(ScaledDotProductAttention_Conv, self).__init__()
        # Scale factor: sqrt(im_size^4) = im_size^2
        # Accounts for spatial dimensions in patch-based attention
        self.scale = nn.Parameter(
            torch.sqrt(torch.FloatTensor([((im_size) ** 4)])),
            requires_grad=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        """
        Compute scaled dot-product attention (continuum attention).
        
        Args:
            query: (batch, nhead, num_patches, x, y, d_k)
            key: (batch, nhead, num_patches, x, y, d_k)
            value: (batch, nhead, num_patches, x, y, d_k)
            key_padding_mask: Optional mask for padding tokens
        Returns:
            output: (batch, num_patches, x, y, d_k, nhead) - after attention
            attention_weights: (batch, nhead, num_patches, num_patches) - attention scores
        """
        # CONTINUUM ATTENTION COMPUTATION
        # einsum "bnpxyd,bnqxyd->bnpq" computes:
        # For each batch (b), head (n), patch pair (p,q):
        #   Dot product across spatial dims (x,y) AND feature dim (d)
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale

        # Apply padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax to get attention weights (over key patches)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # einsum "bnpq,bnqxyd->bnpxyd":
        # For each patch p, weighted sum over all patches q
        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        
        # Permute back to (batch, num_patches, x, y, d_k, nhead)
        output = output.permute(0, 2, 3, 4, 5, 1)
        
        return output, attention_weights


class TransformerEncoderLayer_Conv(nn.Module):
    """
    Complete FANO encoder layer with Fourier attention.
    
    Components:
    - Fourier-based multi-head self-attention (continuum attention)
    - Feedforward network (pointwise MLP)
    - Layer normalization
    - Residual connections
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', etc.)
        norm_first: Whether to apply layer norm before attention (pre-norm vs post-norm)
        do_layer_norm: Whether to apply layer norm at the end
        dim_feedforward: Feedforward network dimension
        modes: Tuple of (modes1, modes2) for Fourier modes, or None to auto-compute
        patch_size: Size of patches (used if modes is None)
        im_size: Image/patch size (used for attention scaling)
        batch_first: Legacy parameter (not really used)
    """
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        activation="relu",
        norm_first=True,
        do_layer_norm=True,
        dim_feedforward=2048,
        modes=None,
        patch_size=1,
        im_size=64,
        batch_first=True,
    ):
        super(TransformerEncoderLayer_Conv, self).__init__()
        
        # Auto-compute Fourier modes if not provided
        if modes is None:
            modes2 = patch_size // 2 + 1
            modes1 = patch_size // 2 + 1
        else:
            modes1 = modes[0]
            modes2 = modes[1]

        self.patch_size = patch_size
        self.im_size = im_size
        self.size_row = im_size
        self.size_col = im_size
        self.d_model = d_model

        # Fourier-based multi-head attention
        self.self_attn = MultiheadAttention_Conv(d_model, nhead, modes1, modes2, im_size, dropout=dropout)
        
        # Feedforward network (pointwise over spatial dims)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation function
        self.activation = getattr(F, activation)
        
        # Configuration
        self.norm_first = norm_first  # Pre-norm vs post-norm architecture
        self.do_layer_norm = do_layer_norm

    def forward(self, x, mask=None):
        """
        Forward pass of FANO encoder layer.
        
        Args:
            x: (batch, num_patches, patch_size, patch_size, d_model)
            mask: Optional attention mask
        Returns:
            (batch, num_patches, patch_size, patch_size, d_model)
        """
        # Pre-norm or post-norm attention block
        if self.norm_first:
            x = self.norm1(x)

        # Self-attention with residual
        attn_output = self.self_attn(x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        
        if not self.norm_first:
            x = self.norm1(x)

        # Feedforward with residual (pointwise over spatial dimensions)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff_output)
        
        if self.do_layer_norm:
            x = self.norm2(x)

        return x

