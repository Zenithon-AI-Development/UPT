import torch
from torch import nn
from typing import Optional

from models.base.single_model_base import SingleModelBase
from modules.quadtree import (
    QuadtreeAggregatorBase,
    QuadtreeLocalAttentionAggregator,
    QuadtreeMaskedMLPAggregator,
)


class QuadtreeEncoder(SingleModelBase):
    """Encoder that aggregates quadtree subnodes into fixed-size supernode tokens."""

    def __init__(
            self,
            dim: int,
            dropout: float = 0.0,
            aggregator: str = "mlp",
            aggregator_kwargs: Optional[dict] = None,
            use_supernode_features: bool = True,
            subnode_feature_dim: Optional[int] = None,
            supernode_feature_dim: Optional[int] = None,
            max_children: int = 64,
            expected_stat_dim: Optional[int] = None,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout_rate = dropout
        self.aggregator_kind = aggregator
        self.aggregator_kwargs = aggregator_kwargs.copy() if aggregator_kwargs else {}
        self.use_supernode_features = use_supernode_features
        self.subnode_feature_dim = subnode_feature_dim
        self.supernode_feature_dim = supernode_feature_dim or subnode_feature_dim
        self.max_children = max_children
        self.expected_stat_dim = expected_stat_dim

        if self.subnode_feature_dim is None or self.supernode_feature_dim is None:
            self.logger.info(
                "QuadtreeEncoder initialized without feature dims; aggregator modules will be materialized lazily."
            )

        self.aggregator: Optional[QuadtreeAggregatorBase] = None
        self.super_proj: Optional[nn.Module] = None
        self.stats_proj: Optional[nn.Module] = None
        self.norm: Optional[nn.LayerNorm] = None
        self.dropout = nn.Dropout(self.dropout_rate)

        self.output_shape = (None, self.dim)
        self._placeholder = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def model_specific_initialization(self) -> None:
        if self.subnode_feature_dim is None or self.supernode_feature_dim is None:
            return
        stat_dim = self.expected_stat_dim or 0
        dummy_device = torch.device("cpu")
        dummy_sub = torch.zeros(1, 1, max(self.max_children, 1), self.subnode_feature_dim, device=dummy_device)
        dummy_super = torch.zeros(1, 1, self.supernode_feature_dim, device=dummy_device)
        self._ensure_modules(dummy_sub, dummy_super, stat_dim)

    # ------------------------------------------------------------------
    def _ensure_modules(
            self,
            subnode_feats: torch.Tensor,
            supernode_feats: torch.Tensor,
            stat_dim: int,
    ) -> None:
        device = subnode_feats.device
        if self.aggregator is None:
            input_dim = subnode_feats.shape[-1]
            super_dim = supernode_feats.shape[-1]
            kwargs = dict(self.aggregator_kwargs)
            if self.aggregator_kind == "mlp":
                self.aggregator = QuadtreeMaskedMLPAggregator(
                    input_dim=input_dim,
                    output_dim=self.dim,
                    **kwargs,
                ).to(device)
            elif self.aggregator_kind == "attention":
                self.aggregator = QuadtreeLocalAttentionAggregator(
                    input_dim=input_dim,
                    output_dim=self.dim,
                    supernode_dim=super_dim,
                    **kwargs,
                ).to(device)
            else:
                raise ValueError(f"Unknown aggregator '{self.aggregator_kind}'")

            if self.use_supernode_features:
                self.super_proj = nn.Linear(super_dim, self.dim).to(device)
            else:
                self.super_proj = None

            if stat_dim > 0:
                self.stats_proj = nn.Linear(stat_dim, self.dim).to(device)
            else:
                self.stats_proj = None

            self.norm = nn.LayerNorm(self.dim).to(device)
            self.dropout = nn.Dropout(self.dropout_rate).to(device)

            self.add_module("aggregator", self.aggregator)
            if self.super_proj is not None:
                self.add_module("super_proj", self.super_proj)
            if self.stats_proj is not None:
                self.add_module("stats_proj", self.stats_proj)
            self.add_module("norm", self.norm)
            return

        if stat_dim > 0:
            if self.stats_proj is None:
                self.stats_proj = nn.Linear(stat_dim, self.dim).to(device)
                self.add_module("stats_proj", self.stats_proj)
            else:
                in_features = getattr(self.stats_proj, "in_features", None)
                if in_features is not None and in_features != stat_dim:
                    raise ValueError(
                        f"stats_proj expected input dim {in_features} but received {stat_dim}"
                    )

    # ------------------------------------------------------------------
    def _compute_stats(self, quadtree_supernodes: dict, quadtree_subnodes: dict) -> Optional[torch.Tensor]:
        sub_mask = quadtree_subnodes.get("mask")
        if sub_mask is None:
            return None

        sub_mask = sub_mask.float()
        max_children = sub_mask.shape[-1]
        if max_children == 0:
            return None

        counts = sub_mask.sum(dim=-1, keepdim=True)
        fraction = counts / max_children

        stats_components = [fraction]

        depths = quadtree_subnodes.get("depths")
        if depths is not None:
            depths = depths.float()
            weighted_depth = (depths * sub_mask).sum(dim=-1, keepdim=True)
            denom = counts.clamp_min(1.0)
            mean_depth = weighted_depth / denom
            super_depth = quadtree_supernodes.get("depths")
            if super_depth is not None:
                norm = super_depth.unsqueeze(-1).float().clamp_min(1.0)
                mean_depth = mean_depth / norm
            stats_components.append(mean_depth)

        stats = torch.cat(stats_components, dim=-1)
        super_mask = quadtree_supernodes.get("mask")
        if super_mask is not None:
            stats = stats * super_mask.unsqueeze(-1).to(stats.dtype)
        zero_mask = counts == 0
        stats = stats.masked_fill(zero_mask.expand_as(stats), 0.0)
        return stats

    # ------------------------------------------------------------------
    def forward(
            self,
            x: torch.Tensor,
            *,
            quadtree_supernodes: dict,
            quadtree_subnodes: dict,
            **kwargs,
    ) -> torch.Tensor:
        if quadtree_supernodes is None or quadtree_subnodes is None:
            raise ValueError("Quadtree data required for QuadtreeEncoder")

        super_feats = quadtree_supernodes["features"]  # [B, M, C_s]
        sub_feats = quadtree_subnodes["features"]  # [B, M, N, C]
        sub_mask = quadtree_subnodes["mask"]  # [B, M, N]

        stats = self._compute_stats(quadtree_supernodes, quadtree_subnodes)
        stat_dim = 0 if stats is None else stats.shape[-1]

        self._ensure_modules(sub_feats, super_feats, stat_dim)

        aggregated = self.aggregator(
            subnode_feats=sub_feats,
            subnode_mask=sub_mask,
            supernode_feats=super_feats,
        )
        if self.super_proj is not None:
            aggregated = aggregated + self.super_proj(super_feats)
        if stats is not None and self.stats_proj is not None:
            aggregated = aggregated + self.stats_proj(stats)
        aggregated = self.norm(aggregated)
        aggregated = self.dropout(aggregated)
        return aggregated


quadtree_encoder = QuadtreeEncoder
__all__ = ["QuadtreeEncoder", "quadtree_encoder"]
