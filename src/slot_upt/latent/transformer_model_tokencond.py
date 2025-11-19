"""
Latent transformer with tokenwise conditioning (Stage 2).
"""

import torch
from torch import nn
from kappamodules.layers import LinearProjection

from models.base.single_model_base import SingleModelBase
from slot_upt.blocks.tokenwise_dit import TokenwiseDitBlock


class TransformerModelTokenCond(SingleModelBase):
    def __init__(
        self,
        dim,
        depth,
        num_attn_heads,
        drop_path_rate=0.0,
        drop_path_decay=True,
        init_weights="xavier_uniform",
        init_last_proj_zero=False,
        cond_token_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero
        self.cond_token_dim = cond_token_dim

        # input/output shape
        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # drop path schedule
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # tokenwise blocks
        self.blocks = nn.ModuleList([
            TokenwiseDitBlock(
                dim=dim,
                num_heads=num_attn_heads,
                cond_dim=cond_token_dim,
                drop_path=dpr[i],
            )
            for i in range(self.depth)
        ])

    def forward(self, x, latent_token_condition: torch.Tensor):
        """
        x: [B, L, dim]
        latent_token_condition: [B, L, cond_token_dim]
        """
        assert x.ndim == 3
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, token_condition=latent_token_condition)
        return x


