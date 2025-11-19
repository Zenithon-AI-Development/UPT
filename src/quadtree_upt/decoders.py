import einops
import numpy as np
import torch
from torch import nn
from torch_geometric.utils import unbatch

from models.decoders.cfd_transformer_perceiver import CfdTransformerPerceiver


class QuadtreeCfdTransformerPerceiver(CfdTransformerPerceiver):
    def __init__(
        self,
        *,
        stats_dim: int,
        max_children: int,
        stats_head_hidden_dim: int = None,
        mask_head_hidden_dim: int = None,
        **kwargs,
    ) -> None:
        if stats_dim <= 0:
            raise ValueError("stats_dim must be positive for quadtree decoder")
        if max_children <= 0:
            raise ValueError("max_children must be positive for quadtree decoder")

        self._stats_dim = stats_dim
        self._max_children = max_children
        self._stats_head_hidden_dim = stats_head_hidden_dim
        self._mask_head_hidden_dim = mask_head_hidden_dim

        nested_kwargs = kwargs.pop("kwargs", None) or {}
        kwargs.pop("max_children", None)
        kwargs.pop("stats_dim", None)
        kwargs.pop("stats_head_hidden_dim", None)
        kwargs.pop("mask_head_hidden_dim", None)
        kwargs.update(nested_kwargs)

        super().__init__(**kwargs)

        stats_hidden = stats_head_hidden_dim or self.dim
        self.stats_head = nn.Sequential(
            nn.LayerNorm(self.dim, eps=1e-6),
            nn.Linear(self.dim, stats_hidden),
            nn.GELU(),
            nn.Linear(stats_hidden, stats_dim),
        )

        mask_hidden = mask_head_hidden_dim or self.dim
        self.mask_head = nn.Sequential(
            nn.LayerNorm(self.dim, eps=1e-6),
            nn.Linear(self.dim, mask_hidden),
            nn.GELU(),
            nn.Linear(mask_hidden, max_children),
        )

    def forward(
        self,
        x: torch.Tensor,
        query_pos: torch.Tensor,
        unbatch_idx: torch.Tensor,
        unbatch_select: torch.Tensor,
        *,
        supernode_mask: torch.Tensor = None,
        static_tokens: torch.Tensor = None,
        condition: torch.Tensor = None,
    ) -> dict:
        assert x.ndim == 3

        token_features = self.input_proj(x)
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        for blk in self.blocks:
            token_features = blk(token_features, **block_kwargs)

        stats_pred = self.stats_head(token_features)
        mask_logits = self.mask_head(token_features)
        if supernode_mask is not None:
            mask_multiplier = supernode_mask.unsqueeze(-1).to(stats_pred.dtype)
            stats_pred = stats_pred * mask_multiplier
            mask_logits = mask_logits * mask_multiplier

        pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(pos_embed)

        kv = self.perc_proj(token_features)
        kv = self.perceiver(q=query, kv=kv, **block_kwargs)
        kv = self.norm(kv)
        field_pred = self.pred(kv)

        if self.clamp is not None:
            assert self.clamp_mode == "log"
            field_pred = torch.sign(field_pred) * (
                self.clamp
                + torch.log1p(field_pred.abs())
                - np.log(1 + self.clamp)
            )

        field_pred = einops.rearrange(
            field_pred, "batch_size max_num_points dim -> (batch_size max_num_points) dim"
        )
        unbatched = unbatch(field_pred, batch=unbatch_idx)
        field_pred = torch.concat([unbatched[i] for i in unbatch_select])

        return {
            "fields": field_pred,
            "supernode_stats": stats_pred,
            "supernode_mask_logits": mask_logits,
        }


quadtree_cfd_transformer_perceiver = QuadtreeCfdTransformerPerceiver

__all__ = ["QuadtreeCfdTransformerPerceiver", "quadtree_cfd_transformer_perceiver"]
