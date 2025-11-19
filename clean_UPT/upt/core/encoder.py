"""
Base encoder interface: data → [B, M, d]

Encoders compress high-dimensional input data into a latent representation
with M tokens of dimension d.
"""

import torch
from torch import nn
from typing import Dict, Optional, Any


class BaseEncoder(nn.Module):
    """
    Base class for UPT encoders.
    
    Encoders transform input data into a latent representation:
    - Input: Variable format (depends on data type)
    - Output: [B, M, d] where:
        B = batch size
        M = number of latent tokens
        d = latent dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_latent_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            input_dim: Dimension of input features per token/point
            latent_dim: Dimension of latent tokens (d)
            num_latent_tokens: Optional fixed number of tokens (M)
                              If None, M may vary per sample
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_latent_tokens = num_latent_tokens
    
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode input data to latent representation.
        
        Args:
            x: Input data (format depends on encoder type)
            condition: Global conditioning vector [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
                                      Hook for grid-structure embeddings, etc.
            **kwargs: Additional encoder-specific arguments
        
        Returns:
            latent: [B, M, d] latent representation
        """
        raise NotImplementedError
    
    @property
    def output_shape(self) -> tuple:
        """Return output shape (M, d) or (None, d) if M varies"""
        if self.num_latent_tokens is not None:
            return (self.num_latent_tokens, self.latent_dim)
        return (None, self.latent_dim)

