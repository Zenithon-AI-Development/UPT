"""
Stage 2 Slot-based UPT with grid-conditioned tokenwise AdaLN.
"""

import torch
import torch.nn.functional as F
from torch import nn

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create
from utils.amp_utils import NoopContext
from slot_upt.slot_assignment import scatter_slots_to_cells, get_slot_positions
from slot_upt.conditioners.grid_structure_conditioner import (
    GridStructureSummary,
    GridStructureMLP,
    GridConditionCombiner,
    GridToLatentCondition,
)


class CfdSlotSimformerModelStage2(CompositeModelBase):
    """
    Slot-based UPT with per-supernode grid-conditioned tokenwise AdaLN (Stage 2).
    """

    expects_subnode_level = True

    def __init__(
        self,
        encoder_tokencond,
        latent_tokencond,
        decoder_tokencond,
        conditioner=None,
        force_decoder_fp32=True,
        M=None,
        N=None,
        # grid conditioning cfg
        L_max: int = 3,
        d_grid: int = 128,
        d_token_cond: int = 256,
        d_latent_cond: int = 256,
        use_pos_mean: bool = True,
        use_pos_var: bool = True,
        pooling: str = "attn",
        **kwargs,
    ):
        num_input_timesteps = kwargs.pop("num_input_timesteps", None)
        super().__init__(**kwargs)
        self.force_decoder_fp32 = force_decoder_fp32

        # M, N
        self.M = M or self.static_ctx.get("num_supernodes", None)
        self.N = N or self.static_ctx.get("num_slots_per_supernode", None)
        assert self.M is not None and self.N is not None, "M and N must be specified"
        self.num_input_timesteps = num_input_timesteps or self.static_ctx.get("num_input_timesteps") or 1
        self.static_ctx["num_supernodes"] = self.M
        self.static_ctx["num_slots_per_supernode"] = self.N
        self.static_ctx["num_input_timesteps"] = self.num_input_timesteps

        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )

        # base conditioner (time/velocity)
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape,
        )
        # Establish condition dim in static ctx for backwards compatibility (unused by tokenwise blocks)
        if self.conditioner is not None and "condition_dim" in self.conditioner.static_ctx:
            self.static_ctx["condition_dim"] = self.conditioner.static_ctx["condition_dim"]

        # encoder (tokenwise)
        self.encoder = create(
            encoder_tokencond,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
            M=self.M,
            N=self.N,
            cond_token_dim=d_token_cond,
        )
        assert self.encoder.output_shape is not None

        # latent (tokenwise)
        self.latent = create(
            latent_tokencond,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
            cond_token_dim=d_latent_cond,
        )

        # decoder (tokenwise)
        self.decoder = create(
            decoder_tokencond,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
            M=self.M,
            N=self.N,
            cond_latent_dim=d_latent_cond,
            cond_token_dec_dim=d_token_cond,
        )

        # grid summary + embeddings
        self.grid_summary = GridStructureSummary(
            L_max=L_max,
            ndim=self.static_ctx.get("ndim", 2),
            use_pos_mean=use_pos_mean,
            use_pos_var=use_pos_var,
        )
        self.grid_mlp = GridStructureMLP(
            input_dim=self.grid_summary.output_dim,
            d_grid=d_grid,
        )
        # combiner heads
        # - token condition for encoder & decoder supernode streams
        self.combine_token = GridConditionCombiner(
            d_base=self.static_ctx.get("condition_dim", d_token_cond),  # fallback if unknown
            d_grid=d_grid,
            d_token=d_token_cond,
        )
        # - latent token condition via pooling/attention from grid embeddings
        n_latent = self.encoder.output_shape[0]
        self.grid_to_latent = GridToLatentCondition(
            d_grid=d_grid,
            d_latent=d_latent_cond,
            n_latent=n_latent,
            mode=pooling,
        )

    @property
    def submodels(self):
        return dict(
            **(dict(conditioner=self.conditioner) if self.conditioner is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    def forward(
        self,
        subnode_feats,
        subnode_mask,
        slot2cell,
        timestep,
        velocity,
        batch_idx,
        unbatch_idx,
        unbatch_select,
        num_cells=None,
        subnode_level=None,
        subnode_pos=None,
        target=None,
        detach_reconstructions=True,
        reconstruct_prev_x=False,
        reconstruct_dynamics=False,
    ):
        outputs = {}
        B, T, M, N, C = subnode_feats.shape
        device = subnode_feats.device

        # base condition [B*T, d_base]
        if self.conditioner is not None:
            if timestep.ndim == 1 and timestep.shape[0] == B:
                timestep_expanded = timestep.unsqueeze(1).expand(B, T).reshape(B * T)
            else:
                timestep_expanded = timestep.reshape(-1)

            if velocity is None:
                velocity_expanded = None
            else:
                if velocity.ndim == 0:
                    velocity_expanded = velocity.expand(timestep_expanded.numel())
                elif velocity.ndim == 1 and velocity.shape[0] == B:
                    velocity_expanded = velocity.unsqueeze(1).expand(B, T).reshape(B * T)
                else:
                    velocity_expanded = velocity.reshape(-1)

            condition = self.conditioner(timestep=timestep_expanded, velocity=velocity_expanded)
        else:
            condition = None
        outputs["condition"] = condition

        # canonical slot positions
        ndim = self.static_ctx.get("ndim", 2)
        slot_positions = get_slot_positions(self.M, self.N, ndim=ndim, device=device)

        # grid summaries g[b,t,m]
        s = self.grid_summary(
            subnode_mask=subnode_mask,
            slot_positions=slot_positions,
            subnode_pos=subnode_pos,
            subnode_level=subnode_level,
        )
        g = self.grid_mlp(s)  # [B, T, M, d_grid]

        # token conditions for encoder/decoder: combine base + g
        if condition is None:
            # if no base cond, use zeros
            d_base = self.combine_token.net[0].in_features - g.size(-1)
            base = torch.zeros(B * T, d_base, device=device, dtype=g.dtype)
        else:
            base = condition  # [B*T, d_base]
        token_condition = self.combine_token(base, g)  # [B*T, M, d_token_cond]
        token_condition_dec = token_condition  # share for decoder supernode stream

        # latent token condition via pooling/attention
        latent_token_condition = self.grid_to_latent(g)  # [B*T, n_latent, d_latent_cond]
        # stats
        outputs["stats/grid_token_cond_norm"] = token_condition.norm(p=2, dim=-1).mean()
        outputs["stats/latent_token_cond_norm"] = latent_token_condition.norm(p=2, dim=-1).mean()

        # encode to latent tokens
        prev_dynamics = self.encoder(
            subnode_feats=subnode_feats,
            subnode_mask=subnode_mask,
            batch_idx=batch_idx,
            token_condition=token_condition,
        )
        outputs["prev_dynamics"] = prev_dynamics

        # latent core with tokenwise conditioning
        dynamics = self.latent(
            prev_dynamics,
            latent_token_condition=latent_token_condition,
        )
        outputs["dynamics"] = dynamics

        # decode to slots
        if self.force_decoder_fp32:
            with torch.autocast(device_type=str(dynamics.device).split(":")[0], enabled=False):
                subnode_feats_hat = self.decoder(
                    dynamics.float(),
                    subnode_mask=subnode_mask,
                    batch_idx=batch_idx,
                    subnode_pos=subnode_pos,
                    latent_token_condition=latent_token_condition.float(),
                    token_condition_dec=token_condition_dec.float(),
                )
        else:
            subnode_feats_hat = self.decoder(
                dynamics,
                subnode_mask=subnode_mask,
                batch_idx=batch_idx,
                subnode_pos=subnode_pos,
                latent_token_condition=latent_token_condition,
                token_condition_dec=token_condition_dec,
            )

        # scatter back to cells
        num_cells_arg = num_cells if num_cells is not None else (slot2cell.max().item() + 1)
        x_hat = scatter_slots_to_cells(subnode_feats_hat, slot2cell, num_cells_arg)
        outputs["x_hat"] = x_hat

        # Optional reconstructions (keep consistent API; use baseline behavior)
        if reconstruct_dynamics:
            outputs["dynamics_hat"] = dynamics  # placeholder to keep losses compatible
        if reconstruct_prev_x:
            outputs["prev_x_hat"] = x_hat  # placeholder

        return outputs


