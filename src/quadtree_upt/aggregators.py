from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class QuadtreeAggregatorBase(nn.Module):
    """Base class for aggregating quadtree subnodes into supernode embeddings."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(
        self,
        subnode_feats: Tensor,
        subnode_mask: Tensor,
        supernode_feats: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError


class QuadtreeMaskedMLPAggregator(QuadtreeAggregatorBase):
    """Masked-mean aggregation followed by an MLP projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
    ) -> None:
        super().__init__(input_dim, output_dim)
        hidden_dim = hidden_dim or max(output_dim, input_dim)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        in_dim = input_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        subnode_feats: Tensor,
        subnode_mask: Tensor,
        supernode_feats: Optional[Tensor] = None,
    ) -> Tensor:
        if subnode_feats.ndim != 4:
            raise ValueError("subnode_feats must be [B, M, N, C]")
        if subnode_mask.ndim != 3:
            raise ValueError("subnode_mask must be [B, M, N]")

        mask = subnode_mask.unsqueeze(-1).to(subnode_feats.dtype)
        masked = subnode_feats * mask
        summed = masked.sum(dim=2)
        counts = mask.sum(dim=2).clamp_min(1e-6)
        mean = summed / counts
        return self.mlp(mean)


class QuadtreeLocalAttentionAggregator(QuadtreeAggregatorBase):
    """Per-supernode attention over its assigned subnodes.

    The supernode feature acts as the query and the subnodes provide keys/values.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        supernode_dim: Optional[int] = None,
        attn_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.0,
        fallback: bool = True,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__(input_dim, output_dim)
        supernode_dim = supernode_dim or input_dim
        attn_dim = attn_dim or max(input_dim, output_dim)
        if attn_dim % num_heads != 0:
            raise ValueError("attn_dim must be divisible by num_heads")

        self.sub_proj = nn.Linear(input_dim, attn_dim)
        self.query_proj = nn.Linear(supernode_dim, attn_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(attn_dim, output_dim)
        self.fallback = fallback
        self.fallback_proj = (
            nn.Linear(supernode_dim, output_dim) if fallback else None
        )
        self.chunk_size = chunk_size

    def forward(
        self,
        subnode_feats: Tensor,
        subnode_mask: Tensor,
        supernode_feats: Optional[Tensor] = None,
    ) -> Tensor:
        if supernode_feats is None:
            raise ValueError("supernode_feats are required for attention aggregation")
        if subnode_feats.ndim != 4:
            raise ValueError("subnode_feats must be [B, M, N, C]")
        if subnode_mask.ndim != 3:
            raise ValueError("subnode_mask must be [B, M, N]")
        if supernode_feats.ndim != 3:
            raise ValueError("supernode_feats must be [B, M, C]")

        B, M, N, C = subnode_feats.shape
        _, _, super_dim = supernode_feats.shape
        flat_sub = subnode_feats.view(B * M, N, C)
        flat_mask = subnode_mask.view(B * M, N)
        flat_super = supernode_feats.view(B * M, 1, super_dim)

        total = flat_sub.size(0)
        chunk = self.chunk_size or total

        def process_chunk(start: int, end: int) -> Tensor:
            sub_chunk = flat_sub[start:end]
            mask_chunk = flat_mask[start:end]
            super_chunk = flat_super[start:end]

            counts = mask_chunk.sum(dim=1)
            has_children = counts > 0

            attn_output = sub_chunk.new_zeros(sub_chunk.size(0), self.output_dim)

            if has_children.any():
                active_idx = has_children.nonzero(as_tuple=False).squeeze(-1)
                sub_active = sub_chunk[active_idx]
                mask_active = mask_chunk[active_idx]
                super_active = super_chunk[active_idx]

                sub_proj = self.sub_proj(sub_active)
                query = self.query_proj(super_active)
                key_padding_mask = ~mask_active

                attn_active, _ = self.attn(
                    query=query,
                    key=sub_proj,
                    value=sub_proj,
                    key_padding_mask=key_padding_mask,
                )
                attn_active = self.out_proj(attn_active.squeeze(1))
                attn_output[active_idx] = attn_active

            if self.fallback:
                fallback_values = self.fallback_proj(super_chunk.squeeze(1))
                has_children_float = has_children.unsqueeze(-1).to(attn_output.dtype)
                attn_output = attn_output * has_children_float + fallback_values * (1.0 - has_children_float)
            else:
                attn_output = attn_output * has_children.unsqueeze(-1).to(attn_output.dtype)
            return attn_output

        if chunk >= total:
            aggregated = process_chunk(0, total)
        else:
            outputs = []
            for start in range(0, total, chunk):
                end = min(start + chunk, total)
                outputs.append(process_chunk(start, end))
            aggregated = torch.cat(outputs, dim=0)

        return aggregated.view(B, M, self.output_dim)


__all__ = [
    "QuadtreeAggregatorBase",
    "QuadtreeMaskedMLPAggregator",
    "QuadtreeLocalAttentionAggregator",
]
