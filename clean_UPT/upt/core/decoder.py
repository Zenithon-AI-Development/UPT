"""
Base decoder interface: [B, M, d] → outputs

Decoders transform latent representations back to the physics domain.
"""

import torch
from torch import nn
from typing import Optional, Tuple


class BaseDecoder(nn.Module):
    """
    Base class for UPT decoders.
    
    Decoders transform latent representation to output:
    - Input: [B, M, d] latent tokens
    - Output: Variable format (depends on output type)
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        **kwargs
    ):
        """
        Args:
            latent_dim: Dimension of latent tokens (d)
            output_dim: Dimension of output features per point
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
    
    def forward(
        self,
        latent: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            latent: [B, M, d] latent tokens
            query_pos: Query positions for output (format depends on decoder type)
            condition: Global conditioning [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
            **kwargs: Additional decoder-specific arguments
        
        Returns:
            output: Decoded output (format depends on decoder type)
        """
        raise NotImplementedError
    
    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Return output shape"""
        return (None, self.output_dim)

