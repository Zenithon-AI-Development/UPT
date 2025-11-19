"""
Slot-based encoder: aggregates slots to supernodes, then applies transformer + Perceiver.

Replaces CfdPool with slot aggregator while keeping transformer/Perceiver blocks unchanged.
"""

from functools import partial

import einops
import torch
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from torch import nn

from models.base.single_model_base import SingleModelBase
from slot_upt.slot_aggregator import MaskedMeanSlotAggregator
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class CfdSlotPoolTransformerPerceiver(SingleModelBase):
    """
    Slot-based encoder: slot aggregator + transformer + Perceiver.
    
    Replaces CfdPool with slot aggregator. Handles time-varying data by flattening
    time dimension before transformer blocks.
    """
    
    def __init__(
        self,
        gnn_dim,
        enc_dim,
        perc_dim,
        enc_depth,
        enc_num_attn_heads,
        perc_num_attn_heads,
        num_latent_tokens=None,
        use_enc_norm=False,
        drop_path_rate=0.0,
        init_weights="xavier_uniform",
        M=None,
        N=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_latent_tokens = num_latent_tokens
        self.use_enc_norm = use_enc_norm
        self.drop_path_rate = drop_path_rate
        self.init_weights = init_weights
        
        # Get M, N from static_ctx or kwargs
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"
        
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        self.num_input_timesteps = self.static_ctx.get("num_input_timesteps", 1)
        slot_input_dim = input_dim // self.num_input_timesteps if input_dim % self.num_input_timesteps == 0 else input_dim
        
        # Slot aggregator (replaces CfdPool)
        self.slot_aggregator = MaskedMeanSlotAggregator(
            input_dim=slot_input_dim,
            output_dim=gnn_dim,
            use_projection=True,
        )
        
        # Blocks (same as baseline)
        self.enc_norm = nn.LayerNorm(gnn_dim, eps=1e-6) if use_enc_norm else nn.Identity()
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=enc_dim, num_heads=enc_num_attn_heads, init_weights=init_weights, drop_path=drop_path_rate)
            for _ in range(enc_depth)
        ])
        
        # Perceiver pooling (same as baseline)
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.static_ctx["condition_dim"],
                    init_weights=init_weights,
                ),
            )
        else:
            block_ctor = partial(
                PerceiverPoolingBlock,
                perceiver_kwargs=dict(init_weights=init_weights),
            )
        self.perceiver = block_ctor(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_latent_tokens,
        )
        
        # output shape
        self.output_shape = (num_latent_tokens, perc_dim)
    
    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="perceiver.query")]
    
    def forward(
        self,
        subnode_feats: torch.Tensor,
        subnode_mask: torch.Tensor,
        batch_idx: torch.Tensor,
        condition=None,
        static_tokens=None,
    ):
        """
        Forward pass: aggregate slots → supernodes → transformer → Perceiver.
        
        Args:
            subnode_feats: [B, T, M, N, C] slot features
            subnode_mask: [B, T, M, N] mask
            batch_idx: [B*T*M] batch indices (for time-varying data)
            condition: [B*T, cond_dim] conditioning (optional)
            static_tokens: (not used, kept for compatibility)
        
        Returns:
            latent: [B*T, n_latent, perc_dim] latent tokens
        """
        B, T, M, N, C = subnode_feats.shape
        device = subnode_feats.device
        
        # Step 1: Aggregate slots to supernodes
        # [B, T, M, N, C] → [B, T, M, gnn_dim]
        supernode_feats = self.slot_aggregator(subnode_feats, subnode_mask)  # [B, T, M, gnn_dim]
        
        # Step 2: Flatten time dimension for transformer
        # [B, T, M, gnn_dim] → [B*T, M, gnn_dim]
        supernode_feats = supernode_feats.view(B * T, M, self.gnn_dim)
        
        # Step 3: Apply encoder blocks (same as baseline)
        block_kwargs = {}
        if condition is not None:
            # Condition is [B*T, cond_dim] - already flattened
            block_kwargs["cond"] = condition
        
        x = self.enc_norm(supernode_feats)
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)
        
        # Step 4: Apply Perceiver pooling (same as baseline)
        x = self.perc_proj(x)
        x = self.perceiver(kv=x, **block_kwargs)
        
        # Output: [B*T, n_latent, perc_dim]
        return x

