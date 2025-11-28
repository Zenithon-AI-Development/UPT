import einops
import torch
import torch.nn.functional as F
import time

from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create
from utils.amp_utils import NoopContext


class CfdSimformerModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            latent,
            decoder,
            force_decoder_fp32=True,
            conditioner=None,
            geometry_encoder=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.force_decoder_fp32 = force_decoder_fp32
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # timestep embed
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape,
        )
        # desc2latent
        self.geometry_encoder = create(
            geometry_encoder,
            model_from_kwargs,
            **common_kwargs,
        )
        # set static_ctx["num_static_tokens"]
        if self.geometry_encoder is not None:
            assert self.geometry_encoder.output_shape is not None and len(self.geometry_encoder.output_shape) == 2
            self.static_ctx["num_static_tokens"] = self.geometry_encoder.output_shape[0]
        else:
            self.static_ctx["num_static_tokens"] = 0
        # set static_ctx["dim"]
        if self.conditioner is not None:
            self.static_ctx["dim"] = self.conditioner.dim
        elif self.geometry_encoder is not None:
            self.static_ctx["dim"] = self.geometry_encoder.output_shape[1]
        else:
            self.static_ctx["dim"] = latent["kwargs"]["dim"]
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        assert self.encoder.output_shape is not None
        # dynamics
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
        )
        # decoder
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
        )

    @property
    def submodels(self):
        return dict(
            **(dict(conditioner=self.conditioner) if self.conditioner is not None else {}),
            **(dict(geometry_encoder=self.geometry_encoder) if self.geometry_encoder is not None else {}),
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(
            self,
            x,
            geometry2d,
            timestep,
            velocity,
            mesh_pos,
            query_pos,
            mesh_edges,
            batch_idx,
            unbatch_idx,
            unbatch_select,
            target=None,
            detach_reconstructions=True,
            reconstruct_prev_x=False,
            reconstruct_dynamics=False,
            quadtree_supernodes=None,
            quadtree_subnodes=None,
            record_component_timings=False,
            record_component_memory=False,
    ):
        outputs = {}
        timings = {} if record_component_timings else None
        memories = {} if record_component_memory else None

        def _sync_tensor(tensor):
            if not (record_component_timings or record_component_memory):
                return
            if tensor is None:
                return
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)

        def _elapsed(start_time):
            return (time.perf_counter() - start_time) * 1000.0

        # encode timestep t
        if self.conditioner is not None:
            cond_start_time = time.perf_counter() if record_component_timings else None
            if record_component_memory and torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            condition = self.conditioner(timestep=timestep, velocity=velocity)
            _sync_tensor(condition)
            if record_component_timings:
                timings["conditioner_ms"] = _elapsed(cond_start_time)
            if record_component_memory and torch.cuda.is_available():
                try:
                    memories["mem/model/conditioner_bytes"] = int(torch.cuda.max_memory_allocated())
                except Exception:
                    pass
        else:
            condition = None
        outputs["condition"] = condition

        # encode geometry
        if self.geometry_encoder is not None:
            static_tokens = self.geometry_encoder(geometry2d)
            outputs["static_tokens"] = static_tokens
            raise NotImplementedError("static tokens are deprecated")
        else:
            static_tokens = None

        # encode data ((x_{t-2}, x_{t-1} -> dynamic_{t-1})
        encoder_kwargs = dict(
            mesh_pos=mesh_pos,
            mesh_edges=mesh_edges,
            batch_idx=batch_idx,
            condition=condition,
            static_tokens=static_tokens,
        )
        if quadtree_supernodes is not None:
            encoder_kwargs["quadtree_supernodes"] = quadtree_supernodes
        if quadtree_subnodes is not None:
            encoder_kwargs["quadtree_subnodes"] = quadtree_subnodes
        _sync_tensor(x)
        encoder_start_time = time.perf_counter() if record_component_timings else None
        if record_component_memory and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        prev_dynamics = self.encoder(
            x,
            **encoder_kwargs,
        )
        _sync_tensor(prev_dynamics)
        if record_component_timings:
            timings["encoder_ms"] = _elapsed(encoder_start_time)
            encoder_detail = getattr(self.encoder, "_last_timings", None)
            if encoder_detail:
                for name, duration in encoder_detail.items():
                    timings[f"encoder/{name}"] = float(duration)
        if record_component_memory and torch.cuda.is_available():
            try:
                memories["mem/model/encoder_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
            # try to fetch encoder-internal memories if available
            encoder_mem_detail = getattr(self.encoder, "_last_memories", None)
            if isinstance(encoder_mem_detail, dict):
                for name, peak_bytes in encoder_mem_detail.items():
                    try:
                        memories[f"mem/model/encoder/{name}_bytes"] = int(peak_bytes)
                    except Exception:
                        pass
        outputs["prev_dynamics"] = prev_dynamics

        # predict current latent (dynamic_{t-1} -> dynamic_t)
        latent_start_time = time.perf_counter() if record_component_timings else None
        if record_component_memory and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        dynamics = self.latent(
            prev_dynamics,
            condition=condition,
            static_tokens=static_tokens,
        )
        _sync_tensor(dynamics)
        if record_component_timings:
            timings["processor_ms"] = _elapsed(latent_start_time)
        if record_component_memory and torch.cuda.is_available():
            try:
                memories["mem/model/latent_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        outputs["dynamics"] = dynamics

        def _decode(latent_tensor, query_tensor, condition_tensor):
            decoder_kwargs = {}
            if quadtree_supernodes is not None:
                mask = quadtree_supernodes.get("mask")
                if mask is not None:
                    decoder_kwargs["supernode_mask"] = mask
            result = self.decoder(
                latent_tensor,
                query_pos=query_tensor,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=condition_tensor,
                **decoder_kwargs,
            )
            if isinstance(result, dict):
                return (
                    result.get("fields"),
                    result.get("supernode_stats"),
                    result.get("supernode_mask_logits"),
                )
            return result, None, None

        # decode next_latent to next_data (dynamic_t -> x_t)
        decoder_start_time = time.perf_counter() if record_component_timings else None
        if record_component_memory and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        if self.force_decoder_fp32:
            with torch.autocast(device_type=str(dynamics.device).split(":")[0], enabled=False):
                x_hat, stats_pred, mask_logits = _decode(
                    dynamics.float(),
                    query_pos.float(),
                    condition.float() if condition is not None else None,
                )
        else:
            x_hat, stats_pred, mask_logits = _decode(
                dynamics,
                query_pos,
                condition,
            )
        _sync_tensor(x_hat)
        if record_component_timings:
            timings["decoder_ms"] = _elapsed(decoder_start_time)
        if record_component_memory and torch.cuda.is_available():
            try:
                memories["mem/model/decoder_bytes"] = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        outputs["x_hat"] = x_hat
        if stats_pred is not None:
            outputs["supernode_stats_pred"] = stats_pred
        if mask_logits is not None:
            outputs["supernode_mask_logits"] = mask_logits
        if record_component_timings:
            outputs["timings"] = timings
        if record_component_memory:
            outputs["memories"] = memories

        # reconstruct dynamics_t from (x_{t-1}, \hat{x}_t)
        if reconstruct_dynamics:
            next_timestep = torch.clamp_max(timestep + 1, max=self.conditioner.num_total_timesteps - 1)
            next_condition = self.conditioner(timestep=next_timestep, velocity=velocity)
            num_output_channels = x_hat.size(1)
            if target is None:
                x_hat_or_gt = x_hat
                if detach_reconstructions:
                    x_hat_or_gt = x_hat_or_gt.detach()
            else:
                x_hat_or_gt = target
            recon_encoder_kwargs = dict(encoder_kwargs)
            recon_encoder_kwargs["condition"] = next_condition
            dynamics_hat = self.encoder(
                torch.concat([x[:, num_output_channels:], x_hat_or_gt], dim=1),
                **recon_encoder_kwargs,
            )
            outputs["dynamics_hat"] = dynamics_hat

        # reconstruct x_{t-1} from dynamic_{t-1}
        if reconstruct_prev_x:
            prev_timestep = F.relu(timestep - 1)
            prev_condition = self.conditioner(timestep=prev_timestep, velocity=velocity)
            if self.force_decoder_fp32:
                with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                    prev_x_hat, _, _ = _decode(
                        prev_dynamics.detach().float() if detach_reconstructions else prev_dynamics.float(),
                        query_pos.float(),
                        prev_condition.float() if prev_condition is not None else None,
                    )
            else:
                prev_x_hat, _, _ = _decode(
                    prev_dynamics.detach() if detach_reconstructions else prev_dynamics,
                    query_pos,
                    prev_condition,
                )
            outputs["prev_x_hat"] = prev_x_hat

        return outputs
    @torch.no_grad()
    def rollout(
            self,
            x,
            geometry2d,
            velocity,
            mesh_pos,
            query_pos,
            mesh_edges,
            batch_idx,
            unbatch_idx,
            unbatch_select,
            num_rollout_timesteps=None,
            mode="image",
            intermediate_results=True,
            clip=None,
    ):
        # check num_rollout_timesteps
        max_timesteps = self.data_container.get_dataset().getdim_timestep()
        num_rollout_timesteps = num_rollout_timesteps or max_timesteps
        assert 0 < num_rollout_timesteps <= max_timesteps
        # setup
        x_hats = []
        timestep = torch.zeros(1, device=x.device, dtype=torch.long)
        condition = None

        def _decode(latent_tensor, query_tensor, condition_tensor):
            result = self.decoder(
                latent_tensor,
                query_pos=query_tensor,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=condition_tensor,
            )
            if isinstance(result, dict):
                return result.get("fields")
            return result

        if mode == "latent":
            # rollout via latent (depending on dynamics_transformer, encoder is either not used at all or only for t0)
            # initial forward
            if self.conditioner is not None:
                condition = self.conditioner(timestep=timestep, velocity=velocity)
            # encode mesh
            dynamics = self.encoder(
                x,
                mesh_pos=mesh_pos,
                mesh_edges=mesh_edges,
                batch_idx=batch_idx,
                condition=condition,
            )
            # predict initial latent
            dynamics = self.latent(
                dynamics,
                condition=condition,
            )
            if intermediate_results:
                if self.force_decoder_fp32:
                    with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                        x_hat = _decode(
                            dynamics.float(),
                            query_pos.float(),
                            condition.float() if condition is not None else None,
                        )
                else:
                    x_hat = _decode(
                        dynamics,
                        query_pos,
                        condition,
                    )
                x_hats.append(x_hat)
            # rollout
            for i in range(num_rollout_timesteps - 1):
                # encode timestep
                if self.conditioner is not None:
                    # increase timestep
                    timestep.add_(1)
                    condition = self.conditioner(timestep=timestep, velocity=velocity)
                # predict next latent
                dynamics = self.latent(
                    dynamics,
                    condition=condition,
                )
                if intermediate_results or i == num_rollout_timesteps - 2:
                    # decode dynamic to data
                    if self.force_decoder_fp32:
                        with torch.autocast(device_type=str(x.device).split(":")[0], enabled=False):
                            x_hat = _decode(
                                dynamics.float(),
                                query_pos.float(),
                                condition.float() if condition is not None else None,
                            )
                    else:
                        x_hat = _decode(
                            dynamics,
                            query_pos,
                            condition,
                        )
                    if clip is not None:
                        x_hat = x_hat.clip(-clip, clip)
                    x_hats.append(x_hat)
        elif mode == "image":
            assert intermediate_results
            # initial forward pass (to get static_tokens)
            outputs = self(
                x,
                geometry2d=geometry2d,
                velocity=velocity,
                timestep=timestep,
                mesh_pos=mesh_pos,
                query_pos=query_pos,
                mesh_edges=mesh_edges,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
            )
            x_hat = outputs["x_hat"]
            x_hats.append(x_hat)

            for _ in range(num_rollout_timesteps - 1):
                # shift last prediction into history
                x = torch.concat([x[:, x_hat.size(1):], x_hat], dim=1)
                # increase timestep
                timestep.add_(1)
                # predict next timestep
                outputs = self(
                    x,
                    geometry2d=geometry2d,
                    velocity=velocity,
                    timestep=timestep,
                    mesh_pos=mesh_pos,
                    query_pos=query_pos,
                    mesh_edges=mesh_edges,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                )
                x_hat = outputs["x_hat"]
                if clip is not None:
                    x_hat = x_hat.clip(-clip, clip)
                x_hats.append(x_hat)
        else:
            raise NotImplementedError

        if not intermediate_results:
            assert len(x_hats) == 1
        # num_rollout_timesteps * (batch_size * num_points, num_channels)
        # -> (batch_size * num_points, num_channels, num_rollout_timesteps)
        return torch.stack(x_hats, dim=2)
