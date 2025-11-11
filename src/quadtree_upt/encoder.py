from typing import Optional

import torch
from torch import nn

from models.base.single_model_base import SingleModelBase

from .aggregators import (
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

    def _expected_augmented_dims(self) -> tuple[int, int]:
        """Return (subnode_dim, supernode_dim) after enrichment."""
        base_sub = self.subnode_feature_dim or 0
        base_super = self.supernode_feature_dim or 0
        return base_sub + 3, base_super + 5

    def model_specific_initialization(self) -> None:
        if self.subnode_feature_dim is None or self.supernode_feature_dim is None:
            return
        sub_dim, super_dim = self._expected_augmented_dims()
        stat_dim = self.expected_stat_dim or 0
        dummy_device = torch.device("cpu")
        dummy_sub = torch.zeros(1, 1, max(self.max_children, 1), sub_dim, device=dummy_device)
        dummy_super = torch.zeros(1, 1, super_dim, device=dummy_device)
        self._ensure_modules(dummy_sub, dummy_super, stat_dim)

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

    def _augment_supernode_features(self, quadtree_supernodes: dict) -> torch.Tensor:
        super_feats = quadtree_supernodes["features"]
        extras = []
        centers = quadtree_supernodes.get("centers")
        if centers is not None:
            extras.append(centers)
        half_sizes = quadtree_supernodes.get("half_sizes")
        if half_sizes is not None:
            extras.append(half_sizes)
        depths = quadtree_supernodes.get("depths")
        if depths is not None:
            depth = depths.unsqueeze(-1).float()
            max_depth = depths.amax().clamp_min(1).float()
            depth = depth / max_depth
            extras.append(depth)
        if extras:
            super_feats = torch.cat([super_feats] + extras, dim=-1)
        return super_feats

    def _augment_subnode_features(self, quadtree_subnodes: dict, quadtree_supernodes: dict) -> torch.Tensor:
        sub_feats = quadtree_subnodes["features"]
        extras = []
        rel_centers = quadtree_subnodes.get("relative_centers")
        if rel_centers is not None:
            extras.append(rel_centers)
        depths = quadtree_subnodes.get("depths")
        if depths is not None:
            depth = depths.float().unsqueeze(-1)
            super_depths = quadtree_supernodes.get("depths")
            if super_depths is not None:
                denom = super_depths.float().unsqueeze(-1).unsqueeze(-1).clamp_min(1.0)
            else:
                denom = depths.float().amax(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
            depth = depth / denom
            extras.append(depth)
        if extras:
            sub_feats = torch.cat([sub_feats] + extras, dim=-1)
        return sub_feats

    def _compute_stats(self, quadtree_supernodes: dict, quadtree_subnodes: dict) -> Optional[torch.Tensor]:
        sub_mask = quadtree_subnodes.get("mask")
        if sub_mask is None:
            return None

        sub_mask = sub_mask.float()
        max_children = sub_mask.shape[-1]
        if max_children == 0:
            return None

        active_counts = sub_mask.sum(dim=-1, keepdim=True)
        fraction = active_counts / max_children

        safe_counts = active_counts.clamp_min(1.0)
        stats_components = [fraction]

        depths = quadtree_subnodes.get("depths")
        super_depths = quadtree_supernodes.get("depths")
        norm_depth = None
        if super_depths is not None:
            norm_depth = super_depths.unsqueeze(-1).float().clamp_min(1.0)
        if depths is not None:
            depths = depths.float()
            weighted_depth = (depths * sub_mask).sum(dim=-1, keepdim=True)
            mean_depth = weighted_depth / safe_counts
            if norm_depth is not None:
                mean_depth = mean_depth / norm_depth
            stats_components.append(mean_depth)

            if norm_depth is not None:
                normalized_depths = depths / norm_depth
            else:
                normalized_depths = depths
            depth_sq = (normalized_depths - mean_depth) ** 2
            depth_var = (depth_sq * sub_mask).sum(dim=-1, keepdim=True) / safe_counts
            depth_std = torch.sqrt(depth_var.clamp_min(1e-12))
            stats_components.append(depth_std)
        else:
            zeros = sub_mask.new_zeros(active_counts.shape)
            stats_components.append(zeros)
            stats_components.append(zeros)

        rel_centers = quadtree_subnodes.get("relative_centers")
        half_sizes = quadtree_supernodes.get("half_sizes")
        if rel_centers is not None:
            radial = torch.norm(rel_centers, dim=-1)
            mean_radius = (radial * sub_mask).sum(dim=-1, keepdim=True) / safe_counts
            if half_sizes is not None:
                norm_radius = torch.norm(half_sizes.float(), dim=-1, keepdim=True).clamp_min(1e-6)
                mean_radius = mean_radius / norm_radius
            stats_components.append(mean_radius)
        else:
            stats_components.append(sub_mask.new_zeros(active_counts.shape))

        stats = torch.cat(stats_components, dim=-1)
        zero_mask = active_counts == 0
        stats = stats.masked_fill(zero_mask.expand_as(stats), 0.0)
        return stats

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

        super_feats = self._augment_supernode_features(quadtree_supernodes)
        sub_feats = self._augment_subnode_features(quadtree_subnodes, quadtree_supernodes)
        sub_mask = quadtree_subnodes["mask"]  # [B, M, N]

        stats = self._compute_stats(quadtree_supernodes, quadtree_subnodes)
        stat_dim = 0 if stats is None else stats.shape[-1]

        self._ensure_modules(sub_feats, super_feats, stat_dim)

        aggregated = self.aggregator(
            subnode_feats=sub_feats,
            subnode_mask=sub_mask,
            supernode_feats=super_feats,
        )

        super_mask = quadtree_supernodes.get("mask")
        mask_tensor = None
        if super_mask is not None:
            mask_tensor = super_mask.unsqueeze(-1).to(aggregated.dtype)
            aggregated = aggregated * mask_tensor

        if self.super_proj is not None:
            proj = self.super_proj(super_feats)
            if mask_tensor is not None:
                proj = proj * mask_tensor
            aggregated = aggregated + proj
        if stats is not None and self.stats_proj is not None:
            proj_stats = self.stats_proj(stats)
            if mask_tensor is not None:
                proj_stats = proj_stats * mask_tensor
            aggregated = aggregated + proj_stats

        aggregated = self.norm(aggregated)
        if mask_tensor is not None:
            aggregated = aggregated * mask_tensor
        aggregated = self.dropout(aggregated)
        return aggregated


quadtree_encoder = QuadtreeEncoder

__all__ = ["QuadtreeEncoder", "quadtree_encoder"]
