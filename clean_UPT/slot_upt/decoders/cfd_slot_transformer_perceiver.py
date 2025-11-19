"""
Slot-based decoder: applies transformer, queries at supernode positions, then splits to slots.

Replaces query-based decoding with slot splitter while keeping transformer blocks unchanged.
"""

from functools import partial

import einops
import numpy as np
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock, DitBlock
from kappamodules.vit import VitBlock
from torch import nn

from models.base.single_model_base import SingleModelBase
from slot_upt.slot_splitter import SlotSplitter


class CfdSlotTransformerPerceiver(SingleModelBase):
    """
    Slot-based decoder: transformer + query at supernodes + slot splitter.
    
    Replaces query-based Perceiver decoding with slot splitter. Handles time-varying
    data by reshaping after transformer blocks.
    """
    
    def __init__(
        self,
        dim,
        depth,
        num_attn_heads,
        use_last_norm=False,
        perc_dim=None,
        perc_num_attn_heads=None,
        drop_path_rate=0.0,
        clamp=None,
        clamp_mode="log",
        init_weights="xavier_uniform",
        M=None,
        N=None,
        slot_positions=None,
        ndim=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_attn_heads = perc_num_attn_heads or num_attn_heads
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.perc_dim = perc_dim
        self.perc_num_attn_heads = perc_num_attn_heads
        self.use_last_norm = use_last_norm
        self.drop_path_rate = drop_path_rate
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        self.init_weights = init_weights
        self.ndim = ndim
        
        # Get M, N from static_ctx or kwargs
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"
        
        # Get slot positions (should be provided or computed)
        if slot_positions is None:
            from slot_upt.slot_assignment import get_slot_positions
            slot_positions = get_slot_positions(self.M, self.N, ndim=ndim, device=torch.device("cpu"))
        
        # input/output shape
        _, num_channels = self.output_shape
        seqlen, input_dim = self.input_shape
        
        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)
        
        # blocks (same as baseline)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = VitBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=dim, num_heads=num_attn_heads, init_weights=init_weights, drop_path=drop_path_rate)
            for _ in range(self.depth)
        ])
        
        # Project latent to supernode dimension
        self.supernode_proj = LinearProjection(dim, perc_dim, init_weights=init_weights)
        
        # Query at supernode positions (instead of arbitrary query positions)
        # We'll query the latent tokens at the supernode positions
        # For simplicity, we'll use the supernode centers as query positions
        # Supernode positions are the centers of each voxel
        # Average slot positions per supernode to get supernode centers
        supernode_pos = slot_positions.mean(dim=1)  # [M, ndim]
        self.register_buffer("supernode_pos", supernode_pos)
        
        # Positional embedding for supernode positions
        self.pos_embed = ContinuousSincosEmbed(dim=perc_dim, ndim=ndim)
        self.query_mlp = nn.Sequential(
            LinearProjection(perc_dim, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim, init_weights=init_weights),
        )
        
        # Perceiver block for querying latent at supernode positions
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(dim=perc_dim, num_heads=perc_num_attn_heads, init_weights=init_weights)
        
        # Slot splitter: splits supernode features to slots
        self.slot_splitter = SlotSplitter(
            input_dim=perc_dim,
            output_dim=num_channels,
            slot_positions=slot_positions,
            hidden_dim=4 * perc_dim,
            ndim=ndim,
            init_weights=init_weights,
        )
        
        self.norm = nn.LayerNorm(perc_dim, eps=1e-6) if use_last_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        subnode_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        condition=None,
        static_tokens=None,
    ):
        """
        Forward pass: transformer → query at supernodes → slot splitter.
        
        Args:
            x: [B*T, n_latent, dim] latent tokens
            subnode_mask: [B, T, M, N] mask
            batch_idx: [B*T*M] batch indices (for time-varying data)
            condition: [B*T, cond_dim] conditioning (optional)
            static_tokens: (not used, kept for compatibility)
        
        Returns:
            subnode_feats: [B, T, M, N, C_out] slot features
        """
        BT, n_latent, dim = x.shape
        device = x.device
        
        # Infer batch size and time from batch_idx
        # For now, assume we can infer from context or pass explicitly
        # We'll need to reshape based on expected B, T
        # For simplicity, assume we get B, T from subnode_mask
        B, T, M, N = subnode_mask.shape
        
        # Step 1: Apply decoder transformer blocks (same as baseline)
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)
        
        # Step 2: Query latent at supernode positions
        # Project to perceiver dimension
        x = self.supernode_proj(x)  # [B*T, n_latent, perc_dim]
        
        # Create query from supernode positions
        # supernode_pos: [M, ndim]
        pos_embed = self.pos_embed(self.supernode_pos)  # [M, perc_dim]
        query = self.query_mlp(pos_embed)  # [M, perc_dim]
        
        # Expand query for batch and time
        query = query.unsqueeze(0).expand(BT, -1, -1)  # [B*T, M, perc_dim]
        
        # Query latent tokens at supernode positions
        supernode_feats = self.perceiver(q=query, kv=x, **block_kwargs)  # [B*T, M, perc_dim]
        supernode_feats = self.norm(supernode_feats)
        
        # Step 3: Reshape to include time dimension
        # [B*T, M, perc_dim] → [B, T, M, perc_dim]
        supernode_feats = supernode_feats.view(B, T, M, self.perc_dim)
        
        # Step 4: Apply slot splitter
        subnode_feats = self.slot_splitter(supernode_feats, subnode_mask)  # [B, T, M, N, C_out]
        
        # Step 5: Apply clamping if needed
        if self.clamp is not None:
            assert self.clamp_mode == "log"
            subnode_feats = torch.sign(subnode_feats) * (
                self.clamp + torch.log(1 + subnode_feats.abs()) - np.log(1 + self.clamp)
            )
        
        return subnode_feats

