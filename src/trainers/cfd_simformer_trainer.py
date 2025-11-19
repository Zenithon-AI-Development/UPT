import os
from functools import cached_property
from pathlib import Path

import kappamodules.utils.tensor_cache as tc
import torch
import torch.nn.functional as F
from kappadata.wrappers import ModeWrapper
from torch import nn
from torch_geometric.nn.pool import radius_graph
from torch_scatter import segment_csr

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.cfd_simformer_collator import CfdSimformerCollator
from losses import loss_fn_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class CfdSimformerTrainer(SgdTrainer):
    def __init__(
            self,
            loss_function,
            detach_reconstructions=False,
            reconstruct_from_target=False,
            reconstruct_prev_x_weight=0,
            reconstruct_dynamics_weight=0,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            max_batch_size=None,
            mask_loss_start_checkpoint=None,
            mask_loss_threshold=None,
            log_first_batch_stats=False,
            supernode_stats_weight=0.0,
            supernode_mask_weight=0.0,
            **kwargs
    ):
        # automatic batchsize is not supported with mesh data
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.loss_function = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)
        self.detach_reconstructions = detach_reconstructions
        self.reconstruct_from_target = reconstruct_from_target
        self.reconstruct_prev_x_weight = reconstruct_prev_x_weight
        self.reconstruct_dynamics_weight = reconstruct_dynamics_weight
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.mask_loss_start_checkpoint = create(mask_loss_start_checkpoint, Checkpoint)
        if self.mask_loss_start_checkpoint is not None:
            assert self.mask_loss_start_checkpoint.is_minimally_specified
            self.mask_loss_start_checkpoint = self.mask_loss_start_checkpoint.to_fully_specified(
                updates_per_epoch=self.update_counter.updates_per_epoch,
                effective_batch_size=self.update_counter.effective_batch_size,
            )
        self.mask_loss_threshold = mask_loss_threshold
        self.supernode_stats_weight = supernode_stats_weight
        self.supernode_mask_weight = supernode_mask_weight
        self.num_supernodes = None
        self.log_first_batch_stats = log_first_batch_stats

    def get_trainer_callbacks(self, model=None):
        keys = ["degree/input"]
        patterns = ["loss_stats", "tensor_stats", "metrics"]
        return [
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSimformerCollator)
        self.num_supernodes = collator.collator.num_supernodes
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        if self.reconstruct_prev_x_weight > 0 or self.reconstruct_dynamics_weight > 0:
            # make sure query is coupled with input
            assert dataset.couple_query_with_input
        else:
            if self.end_checkpoint.is_zero:
                # eval run -> doesnt matter
                pass
            else:
                # check that num_query_points is used if no latent rollout losses are used
                # there is no reason not to use it without reconstruction losses
                assert dataset.root_dataset.num_query_points is not None
        return input_shape

    @property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSimformerCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer
            self._dump_counter = 0
            self._dump_done = False
            self._log_first_batch_stats = bool(getattr(self.trainer, "log_first_batch_stats", False))
            self._activation_stats = {}
            self._gradient_stats = {}
            self._pending_grad_keys = set()
            self._debug_handles = []
            self._debug_written = False
            if self._log_first_batch_stats:
                self._register_debug_hooks()
            self._timing_count = 0

        def to_device(self, item, batch, dataset_mode):
            data = ModeWrapper.get_item(mode=dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            mesh_pos = self.to_device(item="mesh_pos", batch=batch, dataset_mode=dataset_mode)
            batch_idx = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            data = dict(
                x=self.to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                geometry2d=self.to_device(item="geometry2d", batch=batch, dataset_mode=dataset_mode),
                timestep=self.to_device(item="timestep", batch=batch, dataset_mode=dataset_mode),
                velocity=self.to_device(item="velocity", batch=batch, dataset_mode=dataset_mode),
                query_pos=self.to_device(item="query_pos", batch=batch, dataset_mode=dataset_mode),
                mesh_pos=mesh_pos,
                batch_idx=batch_idx,
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device, non_blocking=True),
                unbatch_select=ctx["unbatch_select"].to(self.model.device, non_blocking=True),
                target=self.to_device(item="target", batch=batch, dataset_mode=dataset_mode),
            )
            if "quadtree_supernodes" in ctx:
                data["quadtree_supernodes"] = {k: v.to(self.model.device, non_blocking=True) for k, v in ctx["quadtree_supernodes"].items()}
            if "quadtree_subnodes" in ctx:
                data["quadtree_subnodes"] = {k: v.to(self.model.device, non_blocking=True) for k, v in ctx["quadtree_subnodes"].items()}
            mesh_edges = ModeWrapper.get_item(item="mesh_edges", batch=batch, mode=dataset_mode)
            has_quadtree = "quadtree_supernodes" in ctx
            if mesh_edges is None and not has_quadtree:
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                if self.trainer.num_supernodes is None:
                    flow = "source_to_target"
                    supernode_idxs = None
                else:
                    flow = "target_to_source"
                    supernode_idxs = ctx["supernode_idxs"].to(self.model.device, non_blocking=True)
                mesh_edges = radius_graph(
                    x=mesh_pos,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                    batch=batch_idx,
                    loop=True,
                    flow=flow,
                )
                if supernode_idxs is not None:
                    is_supernode_edge = torch.isin(mesh_edges[0], supernode_idxs)
                    mesh_edges = mesh_edges[:, is_supernode_edge]
                mesh_edges = mesh_edges.T
            elif mesh_edges is not None:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                assert self.trainer.num_supernodes is None
                mesh_edges = mesh_edges.to(self.model.device, non_blocking=True)
            else:
                mesh_edges = None
            data["mesh_edges"] = mesh_edges
            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch=batch)

            x = data.pop("x")
            target = data.pop("target")
            batch_idx = data["batch_idx"]
            batch_size = batch_idx.max() + 1

            profile_flag = os.getenv("PROFILE_COMPONENT_TIMES", "0").lower() not in {"0", "false", ""}
            if profile_flag and self._timing_count == 0:
                self.trainer.logger.info("record_component_timings enabled")

            forward_kwargs = {}
            if self.trainer.reconstruct_from_target:
                forward_kwargs["target"] = target
            model_outputs = self.model(
                x,
                **data,
                **forward_kwargs,
                detach_reconstructions=self.trainer.detach_reconstructions,
                reconstruct_prev_x=self.trainer.reconstruct_prev_x_weight > 0,
                reconstruct_dynamics=self.trainer.reconstruct_dynamics_weight > 0,
                record_component_timings=profile_flag,
            )
            if not torch.isfinite(model_outputs["x_hat"]).all():
                raise RuntimeError("NaN or Inf detected in model output 'x_hat'")

            if not self._dump_done:
                out_dir = self.trainer.path_provider.stage_output_path / "tensors"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(x.detach().cpu(), out_dir / f"{self._dump_counter:04d}_x.pt")
                torch.save(target.detach().cpu(), out_dir / f"{self._dump_counter:04d}_target.pt")
                torch.save(model_outputs["x_hat"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_xhat.pt")
                torch.save(data["timestep"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_timestep.pt")
                torch.save(data["velocity"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_velocity.pt")
                torch.save(data["query_pos"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_querypos.pt")
                torch.save(data["mesh_pos"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_meshpos.pt")
                torch.save(batch_idx.detach().cpu(), out_dir / f"{self._dump_counter:04d}_batchidx.pt")
                mesh_edges = data["mesh_edges"]
                if mesh_edges is not None:
                    torch.save(mesh_edges.detach().cpu(), out_dir / f"{self._dump_counter:04d}_meshedges.pt")
                torch.save(model_outputs["condition"].detach().cpu() if model_outputs.get("condition") is not None else None, out_dir / f"{self._dump_counter:04d}_condition.pt")
                torch.save(model_outputs["prev_dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_prev_dynamics.pt")
                torch.save(model_outputs["dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_dynamics.pt")
                self._dump_counter += 1
                self._dump_done = True

            infos = {}
            losses = {}

            timings = model_outputs.get("timings") if profile_flag else None
            if profile_flag and timings:
                self._timing_count += 1
                timing_msg = ", ".join(f"{name}={float(duration):.3f}ms" for name, duration in timings.items())
                self.trainer.logger.info(f"component_timings[{self._timing_count}]: {timing_msg}")
                for name, duration in timings.items():
                    infos[f"timings/{name}"] = float(duration)

            # next timestep loss
            x_hat_loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"],
                target=target,
                reduction="none",
            )
            if not torch.isfinite(x_hat_loss).all():
                raise RuntimeError("NaN or Inf detected in loss before reduction")
            infos.update(
                {
                    "loss_stats/x_hat/min": x_hat_loss.min(),
                    "loss_stats/x_hat/max": x_hat_loss.max(),
                    "loss_stats/x_hat/gt1": (x_hat_loss > 1).sum() / x_hat_loss.numel(),
                    "loss_stats/x_hat/eq0": (x_hat_loss == 0).sum() / x_hat_loss.numel(),
                }
            )
            if self.trainer.mask_loss_start_checkpoint is not None:
                if self.trainer.mask_loss_start_checkpoint > self.trainer.update_counter.cur_checkpoint:
                    x_hat_loss_mask = x_hat_loss > self.trainer.mask_loss_threshold
                    x_hat_loss = x_hat_loss[x_hat_loss_mask]
                    infos["loss_stats/x_hat/gt_loss_threshold"] = x_hat_loss_mask.sum() / x_hat_loss_mask.numel()
            if reduction == "mean":
                losses["x_hat"] = x_hat_loss.mean()
            elif reduction == "mean_per_sample":
                batch_idx = data["batch_idx"]
                counts = torch.bincount(batch_idx, minlength=batch_size)
                indptr = torch.zeros(batch_size + 1, device=batch_idx.device, dtype=batch_idx.dtype)
                indptr[1:] = counts.cumsum(dim=0)
                losses["x_hat"] = segment_csr(src=x_hat_loss.mean(dim=1), indptr=indptr, reduce="mean")
            else:
                raise NotImplementedError
            total_loss = losses["x_hat"]
            stats_weight = self.trainer.supernode_stats_weight
            stats_pred = model_outputs.get("supernode_stats_pred")
            supernode_data = data.get("quadtree_supernodes")
            stats_target = None
            stats_mask = None
            if supernode_data is not None:
                stats_target = supernode_data.get("stats")
                stats_mask = supernode_data.get("mask")
            if stats_weight and stats_pred is not None and stats_target is not None:
                stats_target = stats_target.to(stats_pred.dtype)
                if stats_mask is not None:
                    mask = stats_mask.unsqueeze(-1).to(stats_pred.dtype)
                    diff = (stats_pred - stats_target) * mask
                    denom = mask.sum().clamp_min(1.0)
                else:
                    diff = stats_pred - stats_target
                    denom = diff.numel()
                stats_loss = diff.pow(2).sum() / denom
                losses["supernode_stats"] = stats_loss
                total_loss = total_loss + stats_weight * stats_loss

            mask_weight = self.trainer.supernode_mask_weight
            mask_logits = model_outputs.get("supernode_mask_logits")
            mask_target = None
            if (quadtree_subnodes := data.get("quadtree_subnodes")) is not None:
                mask_target = quadtree_subnodes.get("mask")
            if mask_weight and mask_logits is not None and mask_target is not None:
                mask_target = mask_target.to(mask_logits.device, dtype=mask_logits.dtype)
                per_entry_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none")
                if stats_mask is not None:
                    valid = stats_mask.unsqueeze(-1).to(mask_logits.dtype)
                    per_entry_loss = per_entry_loss * valid
                    denom = valid.sum().clamp_min(1.0)
                else:
                    denom = mask_target.numel()
                mask_loss = per_entry_loss.sum() / denom
                losses["supernode_mask"] = mask_loss
                total_loss = total_loss + mask_weight * mask_loss

            if self.trainer.reconstruct_prev_x_weight > 0:
                num_channels = model_outputs["prev_x_hat"].size(1)
                prev_x_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_x_hat"],
                    target=x[:, -num_channels:],
                    reduction="none",
                )
                if reduction == "mean":
                    timestep = data["timestep"]
                    timestep_per_point = torch.gather(timestep, dim=0, index=batch_idx)
                    prev_x_hat_loss = prev_x_hat_loss[timestep_per_point != 0]
                    if prev_x_hat_loss.numel() == 0:
                        prev_x_hat_loss = prev_x_hat_loss.new_tensor(0.0)
                    else:
                        if self.trainer.mask_loss_start_checkpoint is not None:
                            if self.trainer.mask_loss_start_checkpoint > self.trainer.update_counter.cur_checkpoint:
                                prev_x_hat_loss_mask = prev_x_hat_loss > self.trainer.mask_loss_threshold
                                prev_x_hat_loss = prev_x_hat_loss[prev_x_hat_loss_mask]
                                infos["loss_stats/prev_x_hat/gt_loss_threshold"] = (
                                    prev_x_hat_loss_mask.sum() / prev_x_hat_loss_mask.numel()
                                )
                        prev_x_hat_loss = prev_x_hat_loss.mean()
                elif reduction == "mean_per_sample":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                losses["prev_x_hat"] = prev_x_hat_loss
                total_loss = total_loss + self.trainer.reconstruct_prev_x_weight * prev_x_hat_loss

            if self.trainer.reconstruct_dynamics_weight > 0:
                dynamics_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["dynamics_hat"],
                    target=model_outputs["dynamics"],
                    reduction="none",
                )
                max_timestep = self.model.conditioner.num_total_timesteps - 1
                timestep = data["timestep"]
                if reduction == "mean":
                    dynamics_hat_mask = timestep != max_timestep
                    if dynamics_hat_mask.sum() > 0:
                        dynamics_hat_loss = dynamics_hat_loss[dynamics_hat_mask].mean()
                    else:
                        dynamics_hat_loss = tc.zeros(size=(1,), device=timestep.device)
                elif reduction == "mean_per_sample":
                    dynamics_hat_loss[timestep == max_timestep] = 0.
                    dynamics_hat_loss = dynamics_hat_loss.flatten(start_dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
                losses["dynamics_hat"] = dynamics_hat_loss
                total_loss = total_loss + self.trainer.reconstruct_dynamics_weight * dynamics_hat_loss

            with torch.no_grad():
                eps = torch.finfo(target.dtype).eps
                batch_idx = data["batch_idx"]
                counts = torch.bincount(batch_idx, minlength=batch_size)
                indptr = torch.zeros(batch_size + 1, device=batch_idx.device, dtype=torch.long)
                indptr[1:] = counts.cumsum(dim=0)
                diff = model_outputs["x_hat"] - target
                abs_diff_per_point = diff.abs().sum(dim=1)
                target_abs_per_point = target.abs().sum(dim=1)
                rel_l1_num = segment_csr(src=abs_diff_per_point, indptr=indptr, reduce="sum")
                rel_l1_den = segment_csr(src=target_abs_per_point, indptr=indptr, reduce="sum")
                rel_l1_per_sample = rel_l1_num / (rel_l1_den + eps)
                sq_diff_per_point = diff.pow(2).sum(dim=1)
                target_sq_per_point = target.pow(2).sum(dim=1)
                rel_l2_num = segment_csr(src=sq_diff_per_point, indptr=indptr, reduce="sum").sqrt()
                rel_l2_den = segment_csr(src=target_sq_per_point, indptr=indptr, reduce="sum").sqrt()
                rel_l2_per_sample = rel_l2_num / (rel_l2_den + eps)
                if reduction == "mean_per_sample":
                    infos["metrics/rel_l1"] = rel_l1_per_sample
                    infos["metrics/rel_l2"] = rel_l2_per_sample
                else:
                    infos["metrics/rel_l1"] = rel_l1_per_sample.mean()
                    infos["metrics/rel_l2"] = rel_l2_per_sample.mean()

            infos.setdefault("degree/input", 0.0)
            if self.trainer.num_supernodes is None and data["mesh_edges"] is not None:
                infos["degree/input"] = len(data["mesh_edges"]) / len(x)
            elif self.trainer.num_supernodes is not None and data["mesh_edges"] is not None:
                infos["degree/input"] = len(data["mesh_edges"]) / (self.trainer.num_supernodes * batch_size)

            if self._log_first_batch_stats and not self._debug_written and not self._pending_grad_keys:
                self._write_debug_stats()

            return dict(total=total_loss, **losses), infos

        # ------------------------------------------------------------------ #
        # Debug instrumentation
        # ------------------------------------------------------------------ #

        def _register_debug_hooks(self):
            submodules = {}
            if hasattr(self.model, "submodels"):
                submodules.update({k: v for k, v in self.model.submodels.items() if v is not None})
            else:
                submodules = {}
            # also include top-level model for completeness
            submodules["model"] = self.model

            for name, module in submodules.items():
                if module is None:
                    continue

                def _hook(mod, inputs, output, module_name=name):
                    self._forward_debug_hook(module_name, output)

                handle = module.register_forward_hook(_hook)
                self._debug_handles.append(handle)

        def _forward_debug_hook(self, name, output):
            if self._debug_written:
                return
            for suffix, tensor in self._iter_tensors(output):
                key = f"{name}{('.' + suffix) if suffix else ''}"
                if key in self._activation_stats:
                    continue
                self._activation_stats[key] = self._tensor_stats(tensor, include_requires_grad=True)
                if tensor is not None and tensor.requires_grad:
                    self._pending_grad_keys.add(key)
                    tensor.register_hook(self._make_grad_hook(key))

        def _make_grad_hook(self, key):
            def _hook(grad):
                if self._debug_written:
                    return grad
                if grad is None:
                    self._gradient_stats.setdefault(key, {"requires_grad": True, "grad_available": False})
                elif key not in self._gradient_stats:
                    self._gradient_stats[key] = self._tensor_stats(grad)
                    self._gradient_stats[key]["requires_grad"] = True
                    self._gradient_stats[key]["grad_available"] = True
                self._pending_grad_keys.discard(key)
                if self._pending_grad_keys == set():
                    self._write_debug_stats()
                return grad

            return _hook

        @staticmethod
        def _iter_tensors(obj, prefix=""):
            if torch.is_tensor(obj):
                yield prefix, obj
            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    sub_prefix = f"{prefix}.{idx}" if prefix else str(idx)
                    yield from CfdSimformerTrainer.Model._iter_tensors(item, sub_prefix)
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    sub_prefix = f"{prefix}.{key}" if prefix else str(key)
                    yield from CfdSimformerTrainer.Model._iter_tensors(item, sub_prefix)
            else:
                return

        @staticmethod
        def _tensor_stats(tensor, include_requires_grad=False):
            stats = {"requires_grad": bool(tensor.requires_grad) if tensor is not None else False}
            if tensor is None or tensor.numel() == 0:
                stats.update(
                    {
                        "shape": [] if tensor is None else list(tensor.shape),
                        "numel": 0,
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }
                )
                return stats
            with torch.no_grad():
                data = tensor.detach()
                if data.dtype.is_floating_point or data.dtype.is_complex:
                    mean = float(data.mean().item())
                    std = float(data.std(unbiased=False).item())
                else:
                    data_float = data.float()
                    mean = float(data_float.mean().item())
                    std = float(data_float.std(unbiased=False).item())
                min_val = float(data.min().item())
                max_val = float(data.max().item())
                stats.update(
                    {
                        "shape": list(data.shape),
                        "numel": int(data.numel()),
                        "mean": mean,
                        "std": std,
                        "min": min_val,
                        "max": max_val,
                    }
                )
                if include_requires_grad:
                    stats["requires_grad"] = bool(tensor.requires_grad)
            return stats

        def _write_debug_stats(self):
            if self._debug_written or not self._log_first_batch_stats:
                return
            diagnostics_dir = self.trainer.path_provider.stage_output_path / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "activations": self._activation_stats,
                "gradients": self._gradient_stats,
            }
            torch.save(payload, diagnostics_dir / "first_batch_stats.pt")
            self._debug_written = True
            self._teardown_debug_hooks()

        def _teardown_debug_hooks(self):
            for handle in self._debug_handles:
                handle.remove()
            self._debug_handles = []
            self._pending_grad_keys.clear()
            self._log_first_batch_stats = False
