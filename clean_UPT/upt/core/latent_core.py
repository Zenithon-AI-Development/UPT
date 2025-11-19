"""
Latent core: transformer over M tokens

The latent core propagates the latent representation forward in time
or performs other transformations in the latent space.
"""

import torch
from torch import nn
from typing import Optional
from functools import partial

from kappamodules.layers import LinearProjection
from kappamodules.transformer import DitBlock, PrenormBlock


class BaseLatentCore(nn.Module):
    """
    Base class for latent core (temporal propagation).
    
    Transforms latent representation: [B, M, d] → [B, M, d]
    """
    
    def __init__(
        self,
        dim: int,
        **kwargs
    ):
        """
        Args:
            dim: Latent dimension (d)
        """
        super().__init__()
        self.dim = dim
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Transform latent representation.
        
        Args:
            x: [B, M, d] latent tokens
            condition: Global conditioning [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
            **kwargs: Additional arguments
        
        Returns:
            x: [B, M, d] transformed latent tokens
        """
        raise NotImplementedError


class TransformerLatentCore(BaseLatentCore):
    """
    Transformer-based latent core.
    
    Standard transformer over M tokens with optional conditioning.
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        cond_dim: Optional[int] = None,
        drop_path_rate: float = 0.0,
        drop_path_decay: bool = True,
        init_weights: str = "xavier_uniform",
        **kwargs
    ):
        """
        Args:
            dim: Latent dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            cond_dim: Conditioning dimension (if None, no conditioning)
            drop_path_rate: Drop path rate
            drop_path_decay: Whether to decay drop path rate
            init_weights: Weight initialization method
        """
        super().__init__(dim=dim, **kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        
        # Input projection (in case input_dim != dim)
        # For now, assume input is already [B, M, dim]
        # If needed, can add: self.input_proj = LinearProjection(input_dim, dim, ...)
        
        # Transformer blocks
        if cond_dim is not None:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        else:
            block_ctor = PrenormBlock
        
        # Drop path schedule
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
            )
            for i in range(depth)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: [B, M, d] latent tokens
            condition: [B, cond_dim] global conditioning (optional)
            extra_token_conditioning: [B, M, cond_dim] per-token conditioning (optional)
        
        Returns:
            x: [B, M, d] transformed latent tokens
        """
        assert x.ndim == 3, f"Expected [B, M, d], got {x.shape}"
        B, M, d = x.shape
        assert d == self.dim, f"Expected dim={self.dim}, got {d}"
        
        # Add per-token conditioning if provided
        if extra_token_conditioning is not None:
            assert extra_token_conditioning.shape == (B, M, self.cond_dim or self.dim)
            x = x + extra_token_conditioning
        
        # Apply transformer blocks
        blk_kwargs = {}
        if condition is not None:
            blk_kwargs["cond"] = condition
        
        for block in self.blocks:
            x = block(x, **blk_kwargs)
        
        return x

