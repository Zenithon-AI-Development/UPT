"""
Fourier Attention Neural Operator (FANO) Module

NEW MODULE - Added for continuum attention experiments
Based on "Continuum Attention for Neural Operators" (Calvello et al., 2024)
https://arxiv.org/abs/2406.06486

This module provides FANO attention mechanisms that operate on patched 2D grids,
using Fourier spectral convolutions for Q/K/V projections and continuum attention
for patch-to-patch interactions.

Exports:
    - SpectralConv2d_Attention: Fourier-based Q/K/V projection
    - MultiheadAttention_Conv: Multi-head FANO attention wrapper
    - ScaledDotProductAttention_Conv: Continuum attention core mechanism
    - TransformerEncoderLayer_Conv: Complete FANO encoder layer
    - TransformerEncoder_Operator: Stack of FANO layers
"""

from .fano_attention import (
    SpectralConv2d_Attention,
    MultiheadAttention_Conv,
    ScaledDotProductAttention_Conv,
    TransformerEncoderLayer_Conv,
    TransformerEncoder_Operator,
)

__all__ = [
    "SpectralConv2d_Attention",
    "MultiheadAttention_Conv",
    "ScaledDotProductAttention_Conv",
    "TransformerEncoderLayer_Conv",
    "TransformerEncoder_Operator",
]

