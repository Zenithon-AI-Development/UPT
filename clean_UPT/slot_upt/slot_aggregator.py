"""
Slot aggregator: aggregates subnode slots to supernode representations.

Implements masked mean aggregation (baseline) and provides base class
for future aggregators (MLP, attention).
"""

import torch
from torch import nn
from typing import Optional


class SlotAggregatorBase(nn.Module):
    """
    Base class for slot aggregators.
    
    Aggregates [B, T, M, N, C] → [B, T, M, d_latent]
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        """
        Args:
            input_dim: Input feature dimension (C)
            output_dim: Output dimension (d_latent)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(
        self,
        subnode_feats: torch.Tensor,
        subnode_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate slots to supernodes.
        
        Args:
            subnode_feats: [B, T, M, N, C] slot features
            subnode_mask: [B, T, M, N] mask (1 for real, 0 for empty)
        
        Returns:
            supernode_feats: [B, T, M, d_latent] supernode features
        """
        raise NotImplementedError


class MaskedMeanSlotAggregator(SlotAggregatorBase):
    """
    Masked mean aggregation: simple baseline.
    
    For each supernode m, computes masked mean over N slots:
    Z_m = sum(μ_{m,n} * X_{m,n}) / (ε + sum(μ_{m,n}))
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        eps: float = 1e-6,
        use_projection: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (C)
            output_dim: Output dimension (d_latent)
            eps: Small epsilon for numerical stability
            use_projection: If True, apply linear projection to output_dim
        """
        super().__init__(input_dim, output_dim)
        self.eps = eps
        self.use_projection = use_projection
        
        if use_projection:
            from kappamodules.layers import LinearProjection
            self.proj = LinearProjection(input_dim, output_dim)
        else:
            assert input_dim == output_dim, "If no projection, input_dim must equal output_dim"
            self.proj = nn.Identity()
    
    def forward(
        self,
        subnode_feats: torch.Tensor,
        subnode_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked mean aggregation.
        
        Args:
            subnode_feats: [B, T, M, N, C] slot features
            subnode_mask: [B, T, M, N] mask (1 for real, 0 for empty)
        
        Returns:
            supernode_feats: [B, T, M, d_latent] supernode features
        """
        # Apply mask: zero out empty slots
        masked_feats = subnode_feats * subnode_mask.unsqueeze(-1)  # [B, T, M, N, C]
        
        # Sum over slots (N dimension)
        sum_feats = masked_feats.sum(dim=3)  # [B, T, M, C]
        
        # Sum of mask values (number of active slots per supernode)
        sum_mask = subnode_mask.sum(dim=3, keepdim=True)  # [B, T, M, 1]
        
        # Compute mean: divide by (eps + sum_mask) for numerical stability
        mean_feats = sum_feats / (self.eps + sum_mask)  # [B, T, M, C]
        
        # Apply projection if needed
        supernode_feats = self.proj(mean_feats)  # [B, T, M, d_latent]
        
        return supernode_feats

