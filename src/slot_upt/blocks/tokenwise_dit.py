"""
Token-wise DiT-style blocks with Adaptive LayerNorm conditioning.

Includes:
- TokenwiseDitBlock: Transformer encoder block (self-attn + FF) with tokenwise AdaLN
- TokenwiseDitPerceiverBlock: Cross-attention block (q over kv) with tokenwise AdaLN
- TokenwiseDitPerceiverPoolingBlock: Learnable queries pooling over kv with tokenwise AdaLN
"""

from typing import Optional

import torch
from torch import nn

from slot_upt.layers.token_adaln import TokenAdaLN


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_mul: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * hidden_mul
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenwiseDitBlock(nn.Module):
    """
    Transformer encoder block with tokenwise AdaLN conditioning.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.adaln1 = TokenAdaLN(dim, cond_dim)
        self.adaln2 = TokenAdaLN(dim, cond_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, token_condition: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d]
        token_condition: [B, L, d_c]
        """
        # Self-attn
        y = self.adaln1(x, token_condition)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop_path(y)
        # FF
        y = self.adaln2(x, token_condition)
        y = self.ff(y)
        x = x + self.drop_path(y)
        return x


class TokenwiseDitPerceiverBlock(nn.Module):
    """
    Cross-attention block: queries attend over key/value tokens with tokenwise AdaLN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim_q: Optional[int] = None,
        cond_dim_kv: Optional[int] = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.adaln_q1 = TokenAdaLN(dim, cond_dim_q) if cond_dim_q is not None else None
        self.adaln_kv1 = TokenAdaLN(dim, cond_dim_kv) if cond_dim_kv is not None else None
        self.adaln_q2 = TokenAdaLN(dim, cond_dim_q) if cond_dim_q is not None else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        q: torch.Tensor,                    # [B, Lq, d]
        kv: torch.Tensor,                   # [B, Lk, d]
        token_condition_q: Optional[torch.Tensor] = None,   # [B, Lq, c_q]
        token_condition_kv: Optional[torch.Tensor] = None,  # [B, Lk, c_kv]
    ) -> torch.Tensor:
        # Modulate
        q_mod = self.adaln_q1(q, token_condition_q) if self.adaln_q1 is not None and token_condition_q is not None else q
        kv_mod = self.adaln_kv1(kv, token_condition_kv) if self.adaln_kv1 is not None and token_condition_kv is not None else kv
        # Cross-attention (queries q over kv)
        y, _ = self.attn(q_mod, kv_mod, kv_mod, need_weights=False)
        q = q + self.drop_path(y)
        # FF on q
        y = self.adaln_q2(q, token_condition_q) if self.adaln_q2 is not None and token_condition_q is not None else q
        y = self.ff(y)
        q = q + self.drop_path(y)
        return q


class TokenwiseDitPerceiverPoolingBlock(nn.Module):
    """
    Perceiver pooling: learnable queries (n_query) attend over kv tokens with tokenwise AdaLN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_query_tokens: int,
        cond_dim_q: Optional[int] = None,
        cond_dim_kv: Optional[int] = None,
        init_scale: float = 0.02,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_query_tokens, dim) * init_scale)
        self.block = TokenwiseDitPerceiverBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim_q=cond_dim_q,
            cond_dim_kv=cond_dim_kv,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(
        self,
        kv: torch.Tensor,                                   # [B, Lk, d]
        token_condition_q: Optional[torch.Tensor] = None,   # [B, Lq, c_q] where Lq == num_query_tokens (optional)
        token_condition_kv: Optional[torch.Tensor] = None,  # [B, Lk, c_kv]
    ) -> torch.Tensor:
        B = kv.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, Lq, d]
        q = self.block(q=q, kv=kv, token_condition_q=token_condition_q, token_condition_kv=token_condition_kv)
        return q


class DropPath(nn.Module):
    """
    Stochastic depth (per sample) as used in Vision Transformers, etc.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


