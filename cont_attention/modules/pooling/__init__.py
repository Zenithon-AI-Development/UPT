"""
Patch Pooling Strategies for FANO Encoder

NEW MODULE - Added for continuum attention experiments

After FANO blocks process patches with spatial structure, we need to convert
them to tokens for the perceiver cross-attention stage. This module provides
two philosophies:

A) Direct Pooling (Custom):
   - SpatialAvgPooling: Simple spatial average per patch
   - LearnedPooling: Learnable probe MLP per patch
   - AttentionPooling: Learnable query-based pooling
   
B) Grid Reconstruction (Faithful to FANO paper):
   - GridUniformSampling: Reshape to full grid → uniform subsample
   - GridAdaptiveSampling: Reshape to full grid → learned top-K sampling

The grid reconstruction approach is more faithful to the original FANO paper
where the full continuous field is reconstructed before any downstream processing.
"""

from .patch_pooling import (
    SpatialAvgPooling,
    LearnedPooling,
    AttentionPooling,
    GridUniformSampling,
    GridAdaptiveSampling,
)

__all__ = [
    # Direct pooling (custom, not from FANO paper)
    "SpatialAvgPooling",
    "LearnedPooling",
    "AttentionPooling",
    # Grid reconstruction (faithful to FANO paper)
    "GridUniformSampling",
    "GridAdaptiveSampling",
]

