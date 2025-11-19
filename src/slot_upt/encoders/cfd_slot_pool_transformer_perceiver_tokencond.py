"""
Slot-based encoder with tokenwise conditioning (Stage 2).

Replaces Dit/Prenorm blocks with TokenwiseDit blocks and uses a tokenwise
Perceiver pooling block. Expects per-supernode token conditioning.
"""

from functools import partial

import torch
from torch import nn
from kappamodules.layers import LinearProjection

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from slot_upt.slot_aggregator import MaskedMeanSlotAggregator
from slot_upt.blocks.tokenwise_dit import TokenwiseDitBlock, TokenwiseDitPerceiverPoolingBlock


class CfdSlotPoolTransformerPerceiverTokenCond(SingleModelBase):
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
        cond_token_dim: int = 256,
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
        self.cond_token_dim = cond_token_dim

        # Get M, N from static_ctx or kwargs
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"

        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        self.num_input_timesteps = self.static_ctx.get("num_input_timesteps", 1)
        slot_input_dim = input_dim // self.num_input_timesteps if input_dim % self.num_input_timesteps == 0 else input_dim

        # Slot aggregator
        self.slot_aggregator = MaskedMeanSlotAggregator(
            input_dim=slot_input_dim,
            output_dim=gnn_dim,
            use_projection=True,
        )

        # Blocks
        self.enc_norm = nn.LayerNorm(gnn_dim, eps=1e-6) if use_enc_norm else nn.Identity()
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        self.blocks = nn.ModuleList([
            TokenwiseDitBlock(dim=enc_dim, num_heads=enc_num_attn_heads, cond_dim=cond_token_dim, drop_path=drop_path_rate)
            for _ in range(enc_depth)
        ])

        # Perceiver pooling (tokenwise kv-conditioning)
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        self.perceiver = TokenwiseDitPerceiverPoolingBlock(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_latent_tokens,
            cond_dim_q=None,
            cond_dim_kv=cond_token_dim,
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
        token_condition: torch.Tensor,    # [B*T, M, cond_token_dim]
        condition=None,
        static_tokens=None,
    ):
        """
        Forward pass: aggregate slots → supernodes → tokenwise blocks → Perceiver pooling.
        """
        B, T, M, N, C = subnode_feats.shape
        # Step 1: Aggregate slots to supernodes
        supernode_feats = self.slot_aggregator(subnode_feats, subnode_mask)  # [B, T, M, gnn_dim]
        # Step 2: Flatten time
        x = supernode_feats.view(B * T, M, self.gnn_dim)
        # Step 3: Encoder tokenwise blocks
        x = self.enc_norm(x)
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, token_condition=token_condition)
        # Step 4: Perceiver pooling with kv conditioning
        x = self.perc_proj(x)
        x = self.perceiver(kv=x, token_condition_kv=token_condition)
        return x


