"""
Slot-based decoder with tokenwise conditioning (Stage 2).
"""

import numpy as np
import torch
from torch import nn
from typing import Optional
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection

from models.base.single_model_base import SingleModelBase
from slot_upt.blocks.tokenwise_dit import TokenwiseDitBlock, TokenwiseDitPerceiverBlock
from slot_upt.slot_splitter import SlotSplitter
from slot_upt.layers.token_adaln import TokenAdaLN
from slot_upt.slot_assignment import get_slot_positions


class CfdSlotTransformerPerceiverTokenCond(SingleModelBase):
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
        cond_latent_dim: int = 256,
        cond_token_dec_dim: int = 256,
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
        self.cond_latent_dim = cond_latent_dim
        self.cond_token_dec_dim = cond_token_dec_dim

        # Get M, N from static_ctx or kwargs
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"

        # slot positions
        if slot_positions is None:
            slot_positions = get_slot_positions(self.M, self.N, ndim=ndim, device=torch.device("cpu"))

        # input/output shape
        _, num_channels = self.output_shape
        seqlen, input_dim = self.input_shape

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # tokenwise latent blocks (over latent tokens)
        self.blocks = nn.ModuleList([
            TokenwiseDitBlock(dim=dim, num_heads=num_attn_heads, cond_dim=cond_latent_dim, drop_path=drop_path_rate)
            for _ in range(self.depth)
        ])

        # Project latent to perceiver dim
        self.supernode_proj = LinearProjection(dim, perc_dim, init_weights=init_weights)

        # supernode queries from canonical positions
        supernode_pos = slot_positions.mean(dim=1)  # [M, ndim]
        self.register_buffer("supernode_pos", supernode_pos)
        self.pos_embed = ContinuousSincosEmbed(dim=perc_dim, ndim=ndim)
        self.query_mlp = nn.Sequential(
            LinearProjection(perc_dim, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim, init_weights=init_weights),
        )

        # cross-attention perceiver with tokenwise conditioning (q conditioned by per-supernode token condition)
        self.perceiver = TokenwiseDitPerceiverBlock(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            cond_dim_q=cond_token_dec_dim,
            cond_dim_kv=None,
        )

        # post-perceiver token AdaLN before splitter
        self.post_perceiver_adaln = TokenAdaLN(dim=perc_dim, cond_dim=cond_token_dec_dim)

        # Slot splitter
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
        x: torch.Tensor,                       # [B*T, n_latent, dim]
        subnode_mask: torch.Tensor,            # [B, T, M, N]
        batch_idx: torch.Tensor,               # [B*T*M]
        latent_token_condition: torch.Tensor,  # [B*T, n_latent, cond_latent_dim]
        token_condition_dec: torch.Tensor,     # [B*T, M, cond_token_dec_dim]
        subnode_pos: Optional[torch.Tensor] = None,
    ):
        BT, n_latent, dim = x.shape
        B, T, M, N = subnode_mask.shape
        # latent blocks
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, token_condition=latent_token_condition)
        # project to perceiver dim
        x = self.supernode_proj(x)  # [B*T, n_latent, perc_dim]
        # build supernode queries
        pos_embed = self.pos_embed(self.supernode_pos)  # [M, perc_dim]
        q = self.query_mlp(pos_embed).unsqueeze(0).expand(BT, -1, -1)  # [B*T, M, perc_dim]
        # cross-attend with per-supernode conditioning on q
        supernode_feats = self.perceiver(q=q, kv=x, token_condition_q=token_condition_dec)  # [B*T, M, perc_dim]
        # post-perceiver AdaLN
        supernode_feats = self.post_perceiver_adaln(supernode_feats, token_condition_dec)
        supernode_feats = self.norm(supernode_feats)
        # reshape to [B, T, M, perc_dim]
        supernode_feats = supernode_feats.view(B, T, M, self.perc_dim)
        # split to slots
        subnode_feats = self.slot_splitter(supernode_feats, subnode_mask, subnode_pos=subnode_pos)  # [B, T, M, N, C]
        if self.clamp is not None:
            assert self.clamp_mode == "log"
            subnode_feats = torch.sign(subnode_feats) * (
                self.clamp + torch.log(1 + subnode_feats.abs()) - np.log(1 + self.clamp)
            )
        return subnode_feats


