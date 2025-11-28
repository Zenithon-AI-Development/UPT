from functools import partial
import time

import torch
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_pool import CfdPool
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class CfdPoolTransformerPerceiver(SingleModelBase):
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
            gnn_init_weights=None,
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
        gnn_init_weights = gnn_init_weights or init_weights
        self.gnn_init_weights = gnn_init_weights

        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        self.mesh_embed = CfdPool(
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            init_weights=gnn_init_weights,
        )

        # blocks
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

        # perceiver pooling
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

    def forward(self, x, mesh_pos, mesh_edges, batch_idx, condition=None, static_tokens=None):
        timings = {}
        memories = {}
        # embed mesh
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.perf_counter()
        x = self.mesh_embed(x, mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx)
        timings["mesh_embed_ms"] = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["mesh_embed"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        # project static_tokens to encoder dim
        # static_tokens = self.static_token_proj(static_tokens)
        # concat static tokens
        # x = torch.cat([static_tokens, x], dim=1)

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.perf_counter()
        x = self.enc_norm(x)
        timings["enc_norm_ms"] = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["enc_norm"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.perf_counter()
        x = self.enc_proj(x)
        timings["enc_proj_ms"] = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["enc_proj"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        blocks_total = 0.0
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        for blk in self.blocks:
            start = time.perf_counter()
            x = blk(x, **block_kwargs)
            blocks_total += (time.perf_counter() - start) * 1000.0
        if self.blocks:
            timings["transformer_blocks_ms"] = blocks_total
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["transformer_blocks"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        # perceiver
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.perf_counter()
        x = self.perc_proj(x)
        timings["perc_proj_ms"] = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["perc_proj"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        start = time.perf_counter()
        x = self.perceiver(kv=x, **block_kwargs)
        timings["perceiver_ms"] = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            try:
                memories["perceiver"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass

        self._last_timings = timings
        self._last_memories = memories
        return x
