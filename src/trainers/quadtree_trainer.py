import os
from functools import cached_property
import time
from pathlib import Path

import kappamodules.utils.tensor_cache as tc
import torch
import torch.nn.functional as F
from kappadata.wrappers import ModeWrapper
from torch import nn
from torch_scatter import segment_csr

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.quadtree_collator import QuadtreeCollator
from losses import loss_fn_from_kwargs
from utils.checkpoint import Checkpoint
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class QuadtreeTrainer(SgdTrainer):
    def __init__(
            self,
            loss_function,
            detach_reconstructions=False,
            reconstruct_from_target=False,
            reconstruct_prev_x_weight=0,
            reconstruct_dynamics_weight=0,
            max_batch_size=None,
            mask_loss_start_checkpoint=None,
            mask_loss_threshold=None,
            log_first_batch_stats=False,
            supernode_stats_weight=0.0,
            supernode_mask_weight=0.0,
            w_field: float = 1.0,
            w_flux: float = 1.0,
            **kwargs
    ):
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
        self.w_field = float(w_field)
        self.w_flux = float(w_flux)
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
        assert isinstance(collator.collator, QuadtreeCollator)
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        return input_shape

    @property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, QuadtreeCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "timestep velocity target cond_vec target_h1 target_h5 scalar_target_h1 scalar_target_h5"

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
            if data is None:
                return None
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch, dataset_mode=None, record_timings: bool = False):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            prep_timings = {} if record_timings else None
            prep_mems = {}  # bytes
            if record_timings:
                prep_start_time = time.perf_counter()
            if record_timings and torch.cuda.is_available():
                torch.cuda.synchronize()
            if record_timings:
                to_device_start = time.perf_counter()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
            
            # Extract quadtree data from context
            quadtree_batch_idx = ctx["quadtree_batch_idx"].to(self.model.device, non_blocking=True)
            per_sample_quadtrees = ctx["per_sample_quadtrees"]
            
            # Extract timestep, velocity, target, etc. from batch if available
            data = dict(
                timestep=self.to_device(item="timestep", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="timestep") else None,
                velocity=self.to_device(item="velocity", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="velocity") else None,
                target=self.to_device(item="target", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="target") else None,
                cond_vec=self.to_device(item="cond_vec", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="cond_vec") else None,
                target_h1_raw=self.to_device(item="target_h1", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="target_h1") else None,
                target_h5_raw=self.to_device(item="target_h5", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="target_h5") else None,
                scalar_target_h1=self.to_device(item="scalar_target_h1", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="scalar_target_h1") else None,
                scalar_target_h5=self.to_device(item="scalar_target_h5", batch=batch, dataset_mode=dataset_mode) if ModeWrapper.has_item(mode=dataset_mode, item="scalar_target_h5") else None,
            )
            data["batch_idx"] = quadtree_batch_idx
            data["per_sample_quadtrees"] = per_sample_quadtrees
            target_h1 = data.pop("target_h1_raw", None)
            target_h5 = data.pop("target_h5_raw", None)
            target_ref = data.get("target")
            if target_h1 is not None and target_ref is not None and target_ref.ndim == 2:
                expected_C = target_ref.shape[1]
                expected_N = target_ref.shape[0]
                if target_h1.ndim == 1:
                    if target_h1.numel() == expected_N:
                        target_h1 = target_h1.unsqueeze(1).repeat(1, expected_C)
                    elif target_h1.numel() == target_ref.numel():
                        target_h1 = target_h1.view_as(target_ref)
                elif target_h1.ndim == 2:
                    if target_h1.shape[1] == 1 and expected_C > 1:
                        target_h1 = target_h1.repeat(1, expected_C)
                    elif target_h1.shape[0] != expected_N or target_h1.shape[1] != expected_C:
                        pass
            if target_h5 is not None and target_ref is not None and target_ref.ndim == 2:
                expected_C = target_ref.shape[1]
                expected_N = target_ref.shape[0]
                if target_h5.ndim == 1:
                    if target_h5.numel() == expected_N:
                        target_h5 = target_h5.unsqueeze(1).repeat(1, expected_C)
                    elif target_h5.numel() == target_ref.numel():
                        target_h5 = target_h5.view_as(target_ref)
                elif target_h5.ndim == 2 and target_h5.shape[1] == 1 and expected_C > 1:
                    target_h5 = target_h5.repeat(1, expected_C)
            data["target_h1"] = target_h1
            data["target_h5"] = target_h5
            if record_timings and torch.cuda.is_available():
                torch.cuda.synchronize()
            if record_timings:
                prep_timings["prepare_to_device_ms"] = (time.perf_counter() - to_device_start) * 1000.0
                if torch.cuda.is_available():
                    try:
                        prep_mems["mem/model/prepare/to_device_bytes"] = int(torch.cuda.max_memory_allocated())
                    except Exception:
                        pass
            if record_timings:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prep_timings["prepare_ms"] = (time.perf_counter() - prep_start_time) * 1000.0
                prep_peak = 0
                for k in ["mem/model/prepare/to_device_bytes"]:
                    prep_peak = max(prep_peak, int(prep_mems.get(k, 0)))
                if prep_peak:
                    prep_mems["mem/model/prepare_bytes"] = int(prep_peak)
                return data, prep_timings, prep_mems
            return data

        def forward(self, batch, reduction="mean"):
            record_prepare_timings = bool(getattr(self.trainer, "timing_cfg", None) and self.trainer.timing_cfg.get("enabled", False))
            record_memory = bool(getattr(self.trainer, "memory_cfg", None) and self.trainer.memory_cfg.get("enabled", False))
            # ensure prepare collects memory peaks even if timing is disabled
            prepared = self.prepare(batch=batch, record_timings=(record_prepare_timings or record_memory))
            if isinstance(prepared, tuple):
                # backward compatibility: could be (data, timings) or (data, timings, mems)
                if len(prepared) == 3:
                    data, prep_timings, prep_mems = prepared
                else:
                    data, prep_timings = prepared
                    prep_mems = {}
            else:
                data, prep_timings = prepared, None
                prep_mems = {}

            target = data.pop("target")
            batch_idx = data["batch_idx"]
            batch_size = batch_idx.max() + 1
            per_sample_quadtrees = data.pop("per_sample_quadtrees")

            # enable inner component timings strictly via config timing.enabled
            timing_cfg = getattr(self.trainer, "timing_cfg", None)
            enabled_by_cfg = bool(isinstance(timing_cfg, dict) and timing_cfg.get("enabled", False))
            profile_flag = enabled_by_cfg
            #profile_flag = (os.getenv("PROFILE_COMPONENT_TIMES", "0").lower() not in {"0", "false", ""})
            if profile_flag and self._timing_count == 0:
                self.trainer.logger.info("record_component_timings enabled")

            forward_kwargs = {}
            if self.trainer.reconstruct_from_target:
                forward_kwargs["target"] = target
            # Dual-horizon forward (Δt=1 and Δt=5) when extended targets are present
            cond_vec = data.get("cond_vec", None)
            target_h1 = data.get("target_h1", None)
            target_h5 = data.get("target_h5", None)
            scalar_target_h1 = data.get("scalar_target_h1", None)
            scalar_target_h5 = data.get("scalar_target_h5", None)
            
            use_dual = target_h1 is not None and target_h5 is not None
            
            # Debug logging for loss configuration (remove after debugging)
            if self._timing_count == 0:
                loss_fn = self.trainer.loss_function
                inner_fn = getattr(loss_fn, "loss_function", loss_fn)
                name_lower = type(inner_fn).__name__.lower()
                self.trainer.logger.info(f"[DEBUG] loss_fn type: {type(loss_fn).__name__}, inner_fn type: {type(inner_fn).__name__}")
                self.trainer.logger.info(f"[DEBUG] use_rel_l2 will be: {('rel' in name_lower and 'l2' in name_lower)}")
                self.trainer.logger.info(f"[DEBUG] use_dual: {use_dual} (target_h1 is None: {target_h1 is None}, target_h5 is None: {target_h5 is None})")

            def _call_model(horizon_val):
                horizon_tensor = None
                if cond_vec is not None:
                    batch_size_local = (data["batch_idx"].max() + 1)
                    horizon_tensor = torch.full((batch_size_local,), float(horizon_val), device=self.model.device, dtype=torch.float32)
                
                # Prepare quadtree data for encoder
                # For each sample, concatenate input quadtrees (4 timesteps)
                # Extract features, positions, and depths from each timestep's quadtree
                all_features = []
                all_node_positions = []
                all_node_depths = []
                all_batch_indices = []
                
                for sample_idx, sample_quadtrees in enumerate(per_sample_quadtrees):
                    input_quadtrees = sample_quadtrees['input_quadtrees']
                    
                    # Process each timestep's quadtree
                    for qd in input_quadtrees:
                        point_hier = qd['point_hierarchies'].to(self.model.device)
                        features = qd['features'].to(self.model.device)
                        pyramids = qd['pyramids'].to(self.model.device)
                        
                        M = point_hier.shape[0]
                        if pyramids.ndim == 3:
                            pyramids = pyramids[0]
                        
                        # Compute depths from pyramid structure
                        max_level = pyramids.shape[1] - 1
                        node_depths = torch.zeros(M, dtype=torch.float32, device=self.model.device)
                        for level in range(max_level + 1):
                            start_idx = int(pyramids[1, level].item())
                            if level < max_level:
                                end_idx = int(pyramids[1, level + 1].item())
                            else:
                                end_idx = M
                            if end_idx > start_idx:
                                node_depths[start_idx:end_idx] = level
                        
                        # Compute positions from point_hierarchies and depths
                        level_powers = 2.0 ** node_depths
                        node_positions = (point_hier.float() / level_powers.unsqueeze(-1)) * 2.0 - 1.0
                        
                        all_features.append(features)
                        all_node_positions.append(node_positions)
                        all_node_depths.append(node_depths)
                        all_batch_indices.extend([sample_idx] * M)
                
                if len(all_features) == 0:
                    raise ValueError("No input quadtree nodes found in batch")
                
                # Concatenate all nodes
                x_quadtree = torch.cat(all_features, dim=0)
                node_positions_quadtree = torch.cat(all_node_positions, dim=0)
                node_depths_quadtree = torch.cat(all_node_depths, dim=0)
                quadtree_batch_idx = torch.tensor(all_batch_indices, dtype=torch.long, device=self.model.device)
                
                return self.model(
                    x=x_quadtree,
                    node_positions=node_positions_quadtree,
                    node_depths=node_depths_quadtree,
                    batch_idx=quadtree_batch_idx,
                    condition=cond_vec,
                    **{k: v for k, v in data.items() if k not in ["target", "target_h1", "target_h5", "scalar_target_h1", "scalar_target_h5", "cond_vec", "per_sample_quadtrees", "batch_idx"]},
                    **forward_kwargs,
                    horizon=horizon_tensor,
                    detach_reconstructions=self.trainer.detach_reconstructions,
                    reconstruct_prev_x=self.trainer.reconstruct_prev_x_weight > 0,
                    reconstruct_dynamics=self.trainer.reconstruct_dynamics_weight > 0,
                    record_component_timings=profile_flag,
                    record_component_memory=record_memory,
                )

            if use_dual:
                outputs_h1 = _call_model(1.0)
                outputs_h5 = _call_model(5.0)
                # aggregate timings/memory across horizons (expose per-h1/h5 and totals)
                if profile_flag or record_memory:
                    t1 = outputs_h1.get("timings", {}) or {}
                    t5 = outputs_h5.get("timings", {}) or {}
                    m1 = outputs_h1.get("memories", {}) or {}
                    m5 = outputs_h5.get("memories", {}) or {}
                    # decoder timings
                    if "decoder_ms" in t1 or "decoder_ms" in t5:
                        d1 = float(t1.get("decoder_ms", 0.0))
                        d5 = float(t5.get("decoder_ms", 0.0))
                        t1["decoder_h1_ms"] = d1
                        t1["decoder_h5_ms"] = d5
                        t1["decoder_ms"] = d1 + d5
                    # scalar head timings (if present)
                    if "scalar_head_ms" in t1 or "scalar_head_ms" in t5:
                        s1 = float(t1.get("scalar_head_ms", 0.0))
                        s5 = float(t5.get("scalar_head_ms", 0.0))
                        t1["scalar_head_h1_ms"] = s1
                        t1["scalar_head_h5_ms"] = s5
                        t1["scalar_head_ms"] = s1 + s5
                    outputs_h1["timings"] = t1
                    # decoder memory (aggregate peak as max, keep per-horizon too)
                    if ("mem/model/decoder_bytes" in m1) or ("mem/model/decoder_bytes" in m5):
                        md1 = int(m1.get("mem/model/decoder_bytes", 0))
                        md5 = int(m5.get("mem/model/decoder_bytes", 0))
                        m1["mem/model/decoder_h1_bytes"] = md1
                        m1["mem/model/decoder_h5_bytes"] = md5
                        m1["mem/model/decoder_bytes"] = max(md1, md5)
                    # scalar head memory
                    if ("mem/model/scalar_head_bytes" in m1) or ("mem/model/scalar_head_bytes" in m5):
                        ms1 = int(m1.get("mem/model/scalar_head_bytes", 0))
                        ms5 = int(m5.get("mem/model/scalar_head_bytes", 0))
                        m1["mem/model/scalar_head_h1_bytes"] = ms1
                        m1["mem/model/scalar_head_h5_bytes"] = ms5
                        m1["mem/model/scalar_head_bytes"] = max(ms1, ms5)
                    outputs_h1["memories"] = m1
                model_outputs = outputs_h1  # for diagnostics/memories/timings base (now aggregated)
                model_outputs["_aux_outputs_h5"] = outputs_h5
            else:
                model_outputs = _call_model(1.0)
            if prep_timings is not None:
                timings = model_outputs.get("timings", {})
                timings.update({
                    "prepare_ms": prep_timings.get("prepare_ms", 0.0),
                    "prepare_to_device_ms": prep_timings.get("prepare_to_device_ms", 0.0),
                })
                model_outputs["timings"] = timings
            if record_memory:
                mems = model_outputs.get("memories", {})
                # merge prepare memory
                mems.update(prep_mems)
                # define a forward peak as max of known forward subdivisions
                forward_peak = 0
                for k in [
                    "mem/model/prepare_bytes",
                    "mem/model/conditioner_bytes",
                    "mem/model/encoder_bytes",
                    "mem/model/latent_bytes",
                    "mem/model/decoder_bytes",
                    "mem/model/scalar_head_bytes",
                ]:
                    forward_peak = max(forward_peak, int(mems.get(k, 0)))
                if forward_peak:
                    mems["mem/forward_bytes"] = int(forward_peak)
                model_outputs["memories"] = mems
            if not torch.isfinite(model_outputs["x_hat"]).all():
                raise RuntimeError("NaN or Inf detected in model output 'x_hat'")

            if not self._dump_done:
                out_dir = self.trainer.path_provider.stage_output_path / "tensors"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(target.detach().cpu(), out_dir / f"{self._dump_counter:04d}_target.pt")
                torch.save(model_outputs["x_hat"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_xhat.pt")
                if data.get("timestep") is not None:
                    torch.save(data["timestep"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_timestep.pt")
                if data.get("velocity") is not None:
                    torch.save(data["velocity"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_velocity.pt")
                torch.save(batch_idx.detach().cpu(), out_dir / f"{self._dump_counter:04d}_batchidx.pt")
                torch.save(model_outputs["condition"].detach().cpu() if model_outputs.get("condition") is not None else None, out_dir / f"{self._dump_counter:04d}_condition.pt")
                if "prev_dynamics" in model_outputs:
                    torch.save(model_outputs["prev_dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_prev_dynamics.pt")
                if "dynamics" in model_outputs:
                    torch.save(model_outputs["dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_dynamics.pt")
                self._dump_counter += 1
                self._dump_done = True

            infos = {}
            if record_memory:
                mems = model_outputs.get("memories", {})
                if mems:
                    # expose in infos for the trainer to pick up (_memory)
                    infos["memories"] = mems
            losses = {}

            timings = model_outputs.get("timings") if profile_flag else None
            if profile_flag and timings:
                self._timing_count += 1
                timing_msg = ", ".join(f"{name}={float(duration):.3f}ms" for name, duration in timings.items())
                self.trainer.logger.info(f"component_timings[{self._timing_count}]: {timing_msg}")
                for name, duration in timings.items():
                    infos[f"timings/{name}"] = float(duration)

            # next timestep loss
            loss_timer_start = time.perf_counter() if profile_flag else None
            if record_memory and torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            
            # Check if using rel_l2 loss (need sample-level aggregation for token-based data)
            # Check if using rel_l2 loss by looking at the loss function class name
            loss_fn = self.trainer.loss_function
            inner_fn = getattr(loss_fn, "loss_function", loss_fn)
            name_lower = type(inner_fn).__name__.lower()
            use_rel_l2 = ("rel" in name_lower and "l2" in name_lower)
            eps = torch.finfo(target.dtype).eps
            
            # Helper to compute sample-level rel_l2 loss with proper token aggregation
            # Computes ||pred - tgt||_2 / ||tgt||_2 (same as metrics/rel_l2)
            def compute_sample_rel_l2(pred, tgt, b_idx, b_size):
                # Force shape matching - pred should be (N, C) where C=4 channels
                # Handle common shape mismatches robustly
                original_pred_shape = pred.shape
                original_tgt_shape = tgt.shape
                
                # Aggressive debug - always log first few calls
                if self._timing_count < 3:
                    self.trainer.logger.error(f"[SHAPE DEBUG] compute_sample_rel_l2: pred.shape={pred.shape}, pred.numel()={pred.numel()}, pred.ndim={pred.ndim}")
                    self.trainer.logger.error(f"[SHAPE DEBUG] compute_sample_rel_l2: tgt.shape={tgt.shape}, tgt.numel()={tgt.numel()}, tgt.ndim={tgt.ndim}")
                    self.trainer.logger.error(f"[SHAPE DEBUG] compute_sample_rel_l2: b_idx.shape={b_idx.shape}, b_size={b_size}")
                
                # Ensure pred is 2D
                if pred.ndim == 1:
                    # pred is 1D - reshape assuming it's (N*C,) -> (N, C)
                    # Assume C=4 (from output_shape)
                    if pred.numel() % 4 == 0:
                        pred = pred.view(-1, 4)
                    else:
                        # Try to infer from batch
                        N_from_batch = b_idx.max().item() + 1 if len(b_idx) > 0 else 1
                        if pred.numel() % N_from_batch == 0:
                            C = pred.numel() // N_from_batch
                            pred = pred.view(-1, C)
                
                # Ensure tgt matches pred shape
                if pred.ndim == 2:
                    if tgt.ndim == 1:
                        # tgt is 1D - need to reshape to match pred
                        if tgt.numel() == pred.shape[0]:
                            # tgt is (N,) - expand to (N, C) by repeating
                            tgt = tgt.unsqueeze(1).repeat(1, pred.shape[1])
                        elif tgt.numel() == pred.numel():
                            # tgt is flattened (N*C,) - reshape to (N, C)
                            tgt = tgt.view_as(pred)
                        else:
                            # Last resort: assume missing channels and repeat
                            if tgt.numel() == pred.shape[0]:
                                tgt = tgt.unsqueeze(1).repeat(1, pred.shape[1])
                            else:
                                # Try to infer
                                if tgt.numel() % pred.shape[0] == 0:
                                    C = tgt.numel() // pred.shape[0]
                                    tgt = tgt.view(pred.shape[0], C)
                                else:
                                    # Force match by repeating
                                    tgt = tgt[:pred.shape[0]].unsqueeze(1).repeat(1, pred.shape[1])
                    elif tgt.ndim == 2:
                        # Both 2D but shapes don't match
                        if tgt.shape[0] == pred.shape[0]:
                            if tgt.shape[1] == 1 and pred.shape[1] > 1:
                                # tgt is (N, 1) - repeat to (N, C)
                                tgt = tgt.repeat(1, pred.shape[1])
                            elif tgt.shape[1] > 1 and pred.shape[1] == 1:
                                # pred is (N, 1) - repeat to (N, C)
                                pred = pred.repeat(1, tgt.shape[1])
                            elif tgt.shape[1] != pred.shape[1]:
                                # Different channel counts - use pred's channels
                                if tgt.shape[1] < pred.shape[1]:
                                    tgt = tgt.repeat(1, pred.shape[1] // tgt.shape[1])
                                else:
                                    pred = pred.repeat(1, tgt.shape[1] // pred.shape[1])
                
                # Final check - if shapes still don't match, log and raise
                if pred.shape != tgt.shape:
                    self.trainer.logger.error(f"[SHAPE MISMATCH] After fix: pred={pred.shape} (orig={original_pred_shape}), tgt={tgt.shape} (orig={original_tgt_shape})")
                    raise RuntimeError(f"Cannot match shapes: pred={pred.shape}, tgt={tgt.shape}")
                
                counts = torch.bincount(b_idx, minlength=b_size)
                indptr = torch.zeros(b_size + 1, device=b_idx.device, dtype=torch.long)
                indptr[1:] = counts.cumsum(dim=0)
                # Final safety check - ensure shapes match before subtraction
                try:
                    diff = pred - tgt
                except RuntimeError as e:
                    if "size of tensor" in str(e) and "must match" in str(e):
                        # Shape mismatch - try one more aggressive fix
                        self.trainer.logger.error(f"[SHAPE ERROR] Before final fix: pred={pred.shape} (numel={pred.numel()}), tgt={tgt.shape} (numel={tgt.numel()})")
                        # Force tgt to match pred
                        if pred.ndim == 2 and tgt.ndim == 2:
                            if tgt.shape[0] == pred.shape[0] and tgt.shape[1] != pred.shape[1]:
                                if tgt.shape[1] == 1:
                                    tgt = tgt.repeat(1, pred.shape[1])
                                elif pred.shape[1] % tgt.shape[1] == 0:
                                    tgt = tgt.repeat(1, pred.shape[1] // tgt.shape[1])
                        elif pred.ndim == 2 and tgt.ndim == 1:
                            if tgt.numel() == pred.shape[0]:
                                tgt = tgt.unsqueeze(1).repeat(1, pred.shape[1])
                            elif tgt.numel() == pred.numel():
                                tgt = tgt.view_as(pred)
                        self.trainer.logger.error(f"[SHAPE ERROR] After final fix: pred={pred.shape}, tgt={tgt.shape}")
                        if pred.shape != tgt.shape:
                            raise RuntimeError(f"Final shape fix failed: pred={pred.shape}, tgt={tgt.shape}, original_error={e}")
                        diff = pred - tgt
                    else:
                        raise
                sq_diff_per_point = diff.pow(2).sum(dim=1)
                tgt_sq_per_point = tgt.pow(2).sum(dim=1)
                rel_l2_num = segment_csr(src=sq_diff_per_point, indptr=indptr, reduce="sum").sqrt()
                rel_l2_den = segment_csr(src=tgt_sq_per_point, indptr=indptr, reduce="sum").sqrt() + eps
                return rel_l2_num / rel_l2_den  # [batch_size] - rel_l2 per sample (same as metric)
            
            # Compute field loss (single or dual-horizon)
            if use_rel_l2:
                # Use sample-level rel_l2 aggregation (same as metrics/rel_l2 but differentiable)
                if use_dual:
                    # Debug shapes before computation
                    if self._timing_count == 0:
                        self.trainer.logger.info(f"[DEBUG] Before compute_sample_rel_l2:")
                        self.trainer.logger.info(f"  outputs_h1['x_hat'].shape={outputs_h1['x_hat'].shape}")
                        self.trainer.logger.info(f"  target_h1.shape={target_h1.shape}")
                        self.trainer.logger.info(f"  batch_idx.shape={batch_idx.shape}, batch_size={batch_size}")
                    rel_l2_h1 = compute_sample_rel_l2(outputs_h1["x_hat"], target_h1, batch_idx, batch_size)
                    rel_l2_h5 = compute_sample_rel_l2(outputs_h5["x_hat"], target_h5, batch_idx, batch_size)
                    x_hat_loss_per_sample = 0.5 * (rel_l2_h1 + rel_l2_h5)
                else:
                    x_hat_loss_per_sample = compute_sample_rel_l2(model_outputs["x_hat"], target, batch_idx, batch_size)
                
                if reduction == "mean":
                    losses["x_hat"] = x_hat_loss_per_sample.mean()
                elif reduction == "mean_per_sample":
                    losses["x_hat"] = x_hat_loss_per_sample
                else:
                    raise NotImplementedError
                
                # For logging stats, use the per-sample values
                with torch.no_grad():
                    infos.update({
                        "loss_stats/x_hat/min": x_hat_loss_per_sample.min(),
                        "loss_stats/x_hat/max": x_hat_loss_per_sample.max(),
                        "loss_stats/x_hat/gt1": (x_hat_loss_per_sample > 1).sum() / x_hat_loss_per_sample.numel(),
                        "loss_stats/x_hat/eq0": (x_hat_loss_per_sample == 0).sum() / x_hat_loss_per_sample.numel(),
                    })
            else:
                # Use configured loss function (e.g., MSE) - original behavior
                if use_dual:
                    x_hat_loss_h1 = self.trainer.loss_function(
                        prediction=outputs_h1["x_hat"],
                        target=target_h1,
                        reduction="none",
                    )
                    x_hat_loss_h5 = self.trainer.loss_function(
                        prediction=outputs_h5["x_hat"],
                        target=target_h5,
                        reduction="none",
                    )
                    x_hat_loss = 0.5 * (x_hat_loss_h1 + x_hat_loss_h5)
                else:
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
                    counts = torch.bincount(batch_idx, minlength=batch_size)
                    indptr = torch.zeros(batch_size + 1, device=batch_idx.device, dtype=batch_idx.dtype)
                    indptr[1:] = counts.cumsum(dim=0)
                    losses["x_hat"] = segment_csr(src=x_hat_loss.mean(dim=1), indptr=indptr, reduce="mean")
                else:
                    raise NotImplementedError
            
            if record_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                try:
                    loss_peak = int(torch.cuda.max_memory_allocated())
                except Exception:
                    loss_peak = 0
                mems = model_outputs.get("memories", {})
                mems["mem/loss_bytes"] = int(loss_peak)
                model_outputs["memories"] = mems
            
            total_loss = self.trainer.w_field * losses["x_hat"]
            # per-horizon logging for fields
            with torch.no_grad():
                if use_dual:
                    field_mse_h1 = ((outputs_h1["x_hat"] - target_h1) ** 2).mean()
                    field_mse_h5 = ((outputs_h5["x_hat"] - target_h5) ** 2).mean()
                    field_mae_h1 = (outputs_h1["x_hat"] - target_h1).abs().mean()
                    field_mae_h5 = (outputs_h5["x_hat"] - target_h5).abs().mean()
                    infos["loss/field_h1"] = field_mse_h1
                    infos["loss/field_h5"] = field_mse_h5
                    infos["metrics/field_mae_h1"] = field_mae_h1
                    infos["metrics/field_mae_h5"] = field_mae_h5
                    infos["loss/field_mean"] = 0.5 * (field_mse_h1 + field_mse_h5)
                    infos["metrics/field_mae_mean"] = 0.5 * (field_mae_h1 + field_mae_h5)
                else:
                    field_mse_h1 = ((model_outputs["x_hat"] - target) ** 2).mean()
                    field_mae_h1 = (model_outputs["x_hat"] - target).abs().mean()
                    infos["loss/field_h1"] = field_mse_h1
                    infos["metrics/field_mae_h1"] = field_mae_h1
            # Scalar loss (if scalar head and targets available)
            # Uses the same loss function as field loss for consistent scale
            flux_h1 = outputs_h1.get("flux_hat") if use_dual else model_outputs.get("flux_hat")
            flux_h5 = outputs_h5.get("flux_hat") if use_dual else None
            if flux_h1 is not None and ((use_dual and scalar_target_h1 is not None and scalar_target_h5 is not None) or (not use_dual and data.get("scalar_target_h1") is not None)):
                if use_rel_l2:
                    # Use unsquared relative L2 for flux loss (to match metrics and field loss)
                    # per-sample: ||pred - tgt||_2 / (||tgt||_2 + eps)
                    def _flux_rel_l2(pred, tgt):
                        num = torch.linalg.vector_norm(pred - tgt, ord=2, dim=-1)
                        den = torch.linalg.vector_norm(tgt, ord=2, dim=-1) + eps
                        return num / den
                    if use_dual:
                        rel_h1 = _flux_rel_l2(flux_h1, scalar_target_h1)
                        rel_h5 = _flux_rel_l2(flux_h5, scalar_target_h5) if flux_h5 is not None else rel_h1
                        flux_loss_per_sample = 0.5 * (rel_h1 + rel_h5)
                    else:
                        tgt1 = data["scalar_target_h1"]
                        rel_h1 = _flux_rel_l2(flux_h1, tgt1)
                        flux_loss_per_sample = rel_h1
                    if reduction == "mean":
                        flux_loss = flux_loss_per_sample.mean()
                    elif reduction == "mean_per_sample":
                        flux_loss = flux_loss_per_sample
                    else:
                        raise NotImplementedError
                    with torch.no_grad():
                        infos["loss/flux_h1"] = rel_h1.mean() if reduction != "mean_per_sample" else rel_h1
                        if use_dual:
                            infos["loss/flux_h5"] = rel_h5.mean() if reduction != "mean_per_sample" else rel_h5
                            if reduction != "mean_per_sample":
                                infos["loss/flux_mean"] = 0.5 * (infos["loss/flux_h1"] + infos["loss/flux_h5"])
                        # keep MAE diagnostics
                        if use_dual:
                            infos["metrics/flux_mae_h1"] = (flux_h1 - scalar_target_h1).abs().mean()
                            infos["metrics/flux_mae_h5"] = (flux_h5 - scalar_target_h5).abs().mean()
                            infos["metrics/flux_mae_mean"] = 0.5 * (infos["metrics/flux_mae_h1"] + infos["metrics/flux_mae_h5"])
                        else:
                            infos["metrics/flux_mae_h1"] = (flux_h1 - tgt1).abs().mean()
                else:
                    # Use configured loss function (e.g., MSE or squared RelL2Loss)
                    if use_dual:
                        flux_loss_h1 = self.trainer.loss_function(
                            prediction=flux_h1,
                            target=scalar_target_h1,
                            reduction="mean",
                        )
                        flux_loss_h5 = self.trainer.loss_function(
                            prediction=flux_h5,
                            target=scalar_target_h5,
                            reduction="mean",
                        ) if flux_h5 is not None else flux_loss_h1
                        flux_loss = 0.5 * (flux_loss_h1 + flux_loss_h5)
                        with torch.no_grad():
                            infos["loss/flux_h1"] = flux_loss_h1
                            infos["loss/flux_h5"] = flux_loss_h5
                            infos["loss/flux_mean"] = flux_loss
                            infos["metrics/flux_mae_h1"] = (flux_h1 - scalar_target_h1).abs().mean()
                            infos["metrics/flux_mae_h5"] = (flux_h5 - scalar_target_h5).abs().mean()
                            infos["metrics/flux_mae_mean"] = 0.5 * (infos["metrics/flux_mae_h1"] + infos["metrics/flux_mae_h5"])
                    else:
                        # fallback: next-step only if dual not provided
                        tgt1 = data["scalar_target_h1"]
                        flux_loss = self.trainer.loss_function(
                            prediction=flux_h1,
                            target=tgt1,
                            reduction="mean",
                        )
                        with torch.no_grad():
                            infos["loss/flux_h1"] = flux_loss
                            infos["metrics/flux_mae_h1"] = (flux_h1 - tgt1).abs().mean()
                losses["flux_hat"] = flux_loss
                total_loss = total_loss + self.trainer.w_flux * flux_loss
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
                prev_x_hat = model_outputs["prev_x_hat"]
                num_channels = prev_x_hat.size(1)
                # Extract last timestep features from quadtree input
                # For quadtree, we need to extract from per_sample_quadtrees
                prev_target_features = []
                for sample_idx, sample_quadtrees in enumerate(per_sample_quadtrees):
                    input_quadtrees = sample_quadtrees['input_quadtrees']
                    if len(input_quadtrees) > 0:
                        last_quadtree = input_quadtrees[-1]
                        last_features = last_quadtree['features']
                        prev_target_features.append(last_features)
                if len(prev_target_features) > 0:
                    prev_target = torch.cat(prev_target_features, dim=0)
                    if prev_target.shape[1] != num_channels:
                        prev_target = prev_target[:, :num_channels]
                else:
                    prev_target = prev_x_hat
                prev_x_hat_loss = self.trainer.loss_function(
                    prediction=prev_x_hat,
                    target=prev_target,
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

            if profile_flag and loss_timer_start is not None:
                infos["timings/loss_ms"] = (time.perf_counter() - loss_timer_start) * 1000.0

            with torch.no_grad():
                eps = torch.finfo(target.dtype).eps
                batch_idx = data["batch_idx"]
                counts = torch.bincount(batch_idx, minlength=batch_size)
                indptr = torch.zeros(batch_size + 1, device=batch_idx.device, dtype=torch.long)
                indptr[1:] = counts.cumsum(dim=0)
                # Field metrics (token-based): compute rel L1/L2 with per-sample aggregation
                def field_rel_metrics(pred, tgt):
                    diff = pred - tgt
                    abs_diff_per_point = diff.abs().sum(dim=1)
                    tgt_abs_per_point = tgt.abs().sum(dim=1)
                    rel_l1_num = segment_csr(src=abs_diff_per_point, indptr=indptr, reduce="sum")
                    rel_l1_den = segment_csr(src=tgt_abs_per_point, indptr=indptr, reduce="sum")
                    rel_l1 = rel_l1_num / (rel_l1_den + eps)
                    sq_diff_per_point = diff.pow(2).sum(dim=1)
                    tgt_sq_per_point = tgt.pow(2).sum(dim=1)
                    rel_l2_num = segment_csr(src=sq_diff_per_point, indptr=indptr, reduce="sum").sqrt()
                    rel_l2_den = segment_csr(src=tgt_sq_per_point, indptr=indptr, reduce="sum").sqrt()
                    rel_l2 = rel_l2_num / (rel_l2_den + eps)
                    return rel_l1, rel_l2
                # Determine if we're in a gyro-style setup (so we label field_* vs flux_*)
                is_gyro = cond_vec is not None
                if use_dual:
                    field_rel_l1_h1, field_rel_l2_h1 = field_rel_metrics(outputs_h1["x_hat"], target_h1)
                    field_rel_l1_h5, field_rel_l2_h5 = field_rel_metrics(outputs_h5["x_hat"], target_h5)
                    if is_gyro:
                        infos["metrics/field_rel_l1_h1"] = field_rel_l1_h1.mean() if reduction != "mean_per_sample" else field_rel_l1_h1
                        infos["metrics/field_rel_l1_h5"] = field_rel_l1_h5.mean() if reduction != "mean_per_sample" else field_rel_l1_h5
                        infos["metrics/field_rel_l2_h1"] = field_rel_l2_h1.mean() if reduction != "mean_per_sample" else field_rel_l2_h1
                        infos["metrics/field_rel_l2_h5"] = field_rel_l2_h5.mean() if reduction != "mean_per_sample" else field_rel_l2_h5
                        if reduction != "mean_per_sample":
                            infos["metrics/field_rel_l1_mean"] = 0.5 * (infos["metrics/field_rel_l1_h1"] + infos["metrics/field_rel_l1_h5"])
                            infos["metrics/field_rel_l2_mean"] = 0.5 * (infos["metrics/field_rel_l2_h1"] + infos["metrics/field_rel_l2_h5"])
                    else:
                        # Non-gyro experiments shouldn't get subdivided names
                        if reduction != "mean_per_sample":
                            infos["metrics/rel_l1"] = 0.5 * (field_rel_l1_h1.mean() + field_rel_l1_h5.mean())
                            infos["metrics/rel_l2"] = 0.5 * (field_rel_l2_h1.mean() + field_rel_l2_h5.mean())
                        else:
                            infos["metrics/rel_l1"] = 0.5 * (field_rel_l1_h1 + field_rel_l1_h5)
                            infos["metrics/rel_l2"] = 0.5 * (field_rel_l2_h1 + field_rel_l2_h5)
                else:
                    field_rel_l1_h1, field_rel_l2_h1 = field_rel_metrics(model_outputs["x_hat"], target)
                    if is_gyro:
                        infos["metrics/field_rel_l1_h1"] = field_rel_l1_h1.mean() if reduction != "mean_per_sample" else field_rel_l1_h1
                        infos["metrics/field_rel_l2_h1"] = field_rel_l2_h1.mean() if reduction != "mean_per_sample" else field_rel_l2_h1
                        # Backwards-compatible generic keys (point to field h1)
                        if reduction != "mean_per_sample":
                            infos["metrics/rel_l1"] = infos["metrics/field_rel_l1_h1"]
                            infos["metrics/rel_l2"] = infos["metrics/field_rel_l2_h1"]
                        else:
                            infos["metrics/rel_l1"] = field_rel_l1_h1
                            infos["metrics/rel_l2"] = field_rel_l2_h1
                    else:
                        # Non-gyro: only generic names
                        if reduction != "mean_per_sample":
                            infos["metrics/rel_l1"] = field_rel_l1_h1.mean()
                            infos["metrics/rel_l2"] = field_rel_l2_h1.mean()
                        else:
                            infos["metrics/rel_l1"] = field_rel_l1_h1
                            infos["metrics/rel_l2"] = field_rel_l2_h1
                # Flux metrics (vector per sample): rel L1/L2
                flux_h1 = outputs_h1.get("flux_hat") if use_dual else model_outputs.get("flux_hat")
                if flux_h1 is not None:
                    if use_dual and scalar_target_h5 is not None and outputs_h5.get("flux_hat") is not None:
                        fh1 = flux_h1
                        th1 = scalar_target_h1
                        fh5 = outputs_h5["flux_hat"]
                        th5 = scalar_target_h5
                        rel_l1_h1 = (fh1 - th1).abs().sum(dim=-1) / (th1.abs().sum(dim=-1) + eps)
                        rel_l2_h1 = (fh1 - th1).pow(2).sum(dim=-1).sqrt() / (th1.pow(2).sum(dim=-1).sqrt() + eps)
                        rel_l1_h5 = (fh5 - th5).abs().sum(dim=-1) / (th5.abs().sum(dim=-1) + eps)
                        rel_l2_h5 = (fh5 - th5).pow(2).sum(dim=-1).sqrt() / (th5.pow(2).sum(dim=-1).sqrt() + eps)
                        infos["metrics/flux_rel_l1_h1"] = rel_l1_h1.mean()
                        infos["metrics/flux_rel_l2_h1"] = rel_l2_h1.mean()
                        infos["metrics/flux_rel_l1_h5"] = rel_l1_h5.mean()
                        infos["metrics/flux_rel_l2_h5"] = rel_l2_h5.mean()
                        infos["metrics/flux_rel_l1_mean"] = 0.5 * (infos["metrics/flux_rel_l1_h1"] + infos["metrics/flux_rel_l1_h5"])
                        infos["metrics/flux_rel_l2_mean"] = 0.5 * (infos["metrics/flux_rel_l2_h1"] + infos["metrics/flux_rel_l2_h5"])
                    else:
                        th1 = data.get("scalar_target_h1")
                        if th1 is not None:
                            rel_l1_h1 = (flux_h1 - th1).abs().sum(dim=-1) / (th1.abs().sum(dim=-1) + eps)
                            rel_l2_h1 = (flux_h1 - th1).pow(2).sum(dim=-1).sqrt() / (th1.pow(2).sum(dim=-1).sqrt() + eps)
                            infos["metrics/flux_rel_l1_h1"] = rel_l1_h1.mean()
                            infos["metrics/flux_rel_l2_h1"] = rel_l2_h1.mean()

            infos.setdefault("degree/input", 0.0)
            # Compute average nodes per sample for quadtree
            total_nodes = batch_idx.shape[0]
            infos["degree/input"] = total_nodes / batch_size if batch_size > 0 else 0.0

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
                    yield from QuadtreeTrainer.Model._iter_tensors(item, sub_prefix)
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    sub_prefix = f"{prefix}.{key}" if prefix else str(key)
                    yield from QuadtreeTrainer.Model._iter_tensors(item, sub_prefix)
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
