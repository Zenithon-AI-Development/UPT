"""
Conditioners: time, params, etc.

Conditioners provide global and per-token conditioning vectors
for timestep, physical parameters, etc.
"""

import torch
from torch import nn
from typing import Optional, Dict, Any


class BaseConditioner(nn.Module):
    """
    Base class for conditioners.
    
    Conditioners provide conditioning vectors for:
    - Global conditioning: [B, cond_dim] for entire batch
    - Per-token conditioning: [B, M, cond_dim] for each token
    """
    
    def __init__(
        self,
        cond_dim: int,
        **kwargs
    ):
        """
        Args:
            cond_dim: Dimension of conditioning vectors
        """
        super().__init__()
        self.cond_dim = cond_dim
    
    def forward(
        self,
        timestep: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute global conditioning vector.
        
        Args:
            timestep: [B] timestep indices
            velocity: [B] or [B, ...] velocity or other scalar/vector features
            params: Dict of additional parameters
            **kwargs: Additional arguments
        
        Returns:
            condition: [B, cond_dim] global conditioning vector
        """
        raise NotImplementedError
    
    def forward_per_token(
        self,
        num_tokens: int,
        timestep: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute per-token conditioning vectors.
        
        Args:
            num_tokens: Number of tokens per sample (M)
            timestep: [B] timestep indices
            velocity: [B] or [B, ...] velocity or other features
            params: Dict of additional parameters
            **kwargs: Additional arguments
        
        Returns:
            condition: [B, M, cond_dim] per-token conditioning vectors
        """
        # Default: broadcast global conditioning to per-token
        global_cond = self.forward(timestep=timestep, velocity=velocity, params=params, **kwargs)
        B = global_cond.shape[0]
        return global_cond.unsqueeze(1).expand(B, num_tokens, -1)

