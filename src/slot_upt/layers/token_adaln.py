"""
Token-wise Adaptive LayerNorm (AdaLN) with gating for DiT-style modulation.

Given token features x[b, L, d] and per-token conditioning c[b, L, d_c],
produce y = gate * (gamma * LN(x) + beta), where [gamma|beta|gate] = Linear(c).
"""

from typing import Optional

import torch
from torch import nn


class TokenAdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        # One affine to produce gamma, beta, gate
        self.to_affine = nn.Linear(cond_dim, dim * 3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, L, d]
        cond: [B, L, d_c]
        """
        assert x.ndim == 3
        assert cond.ndim == 3 and cond.shape[0] == x.shape[0] and cond.shape[1] == x.shape[1]
        B, L, d = x.shape
        assert d == self.dim
        aff = self.to_affine(cond)  # [B, L, 3d]
        gamma, beta, gate = aff.chunk(3, dim=-1)  # [B, L, d] each
        y = self.norm(x)
        y = gamma * y + beta
        y = gate * y
        return y


