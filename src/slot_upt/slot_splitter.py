"""
Slot splitter: splits supernode representations back to subnode slots.

Implements MLP-based splitter that takes supernode features and generates
per-slot outputs using positional embeddings.
"""

import torch
from torch import nn
from typing import Optional


class SlotSplitter(nn.Module):
    """
    Slot splitter: splits supernode features to subnode slots.
    
    Maps [B, T, M, d_latent] → [B, T, M, N, C_out]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        slot_positions: torch.Tensor,
        hidden_dim: Optional[int] = None,
        ndim: int = 2,
        init_weights: str = "xavier_uniform",
    ):
        """
        Args:
            input_dim: Supernode feature dimension (d_latent)
            output_dim: Output feature dimension (C_out)
            slot_positions: [M, N, d_x] canonical slot positions
            hidden_dim: Hidden dimension for MLP (default: 4 * input_dim)
            ndim: Spatial dimension (2 or 3)
            init_weights: Weight initialization method
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.init_weights = init_weights
        
        # Register slot positions as buffer
        self.register_buffer("slot_positions", slot_positions)
        M, N, d_x = slot_positions.shape
        self.M = M
        self.N = N
        
        # Positional embedding for slot positions
        from kappamodules.layers import ContinuousSincosEmbed
        pos_embed_dim = input_dim  # Match input_dim for concatenation
        self.pos_embed = ContinuousSincosEmbed(dim=pos_embed_dim, ndim=ndim)
        
        # MLP for splitting
        hidden_dim = hidden_dim or (4 * input_dim)
        self.split_mlp = nn.Sequential(
            nn.Linear(input_dim + pos_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        if self.init_weights == "xavier_uniform":
            from kappamodules.init import init_xavier_uniform_zero_bias
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            from kappamodules.init import init_truncnormal_zero_bias
            self.apply(init_truncnormal_zero_bias)
        else:
            raise ValueError(f"Unknown init_weights: {self.init_weights}")
    
    def forward(
        self,
        supernode_feats: torch.Tensor,
        subnode_mask: torch.Tensor,
        subnode_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Split supernode features to subnode slots.
        
        Args:
            supernode_feats: [B, T, M, d_latent] supernode features
            subnode_mask: [B, T, M, N] mask (1 for real, 0 for empty)
            subnode_pos: [B, T, M, N, ndim] optional real slot positions
        
        Returns:
            subnode_feats: [B, T, M, N, C_out] slot features
        """
        B, T, M, d_latent = supernode_feats.shape
        device = supernode_feats.device
        
        # Expand supernode features to [B, T, M, N, d_latent]
        # Each supernode feature is broadcast to all N slots
        expanded_feats = supernode_feats.unsqueeze(3).expand(B, T, M, self.N, d_latent)
        
        if subnode_pos is not None:
            assert subnode_pos.shape[:4] == (B, T, M, self.N)
            pos_input = subnode_pos.to(device=device, dtype=torch.float32)
            pos_embeds = self.pos_embed(pos_input.view(-1, self.ndim))
            pos_embeds = pos_embeds.view(B, T, M, self.N, -1)
        else:
            slot_pos_flat = self.slot_positions.view(M * self.N, self.ndim)
            pos_embeds = self.pos_embed(slot_pos_flat)  # [M*N, pos_embed_dim]
            pos_embeds = pos_embeds.view(1, 1, M, self.N, -1).expand(B, T, -1, -1, -1)
        
        # Concatenate supernode features with positional embeddings
        concat_feats = torch.cat([expanded_feats, pos_embeds], dim=-1)  # [B, T, M, N, d_latent + pos_embed_dim]
        
        # Apply MLP
        subnode_feats = self.split_mlp(concat_feats)  # [B, T, M, N, C_out]
        
        # Apply mask: zero out empty slots
        subnode_feats = subnode_feats * subnode_mask.unsqueeze(-1)
        
        return subnode_feats

