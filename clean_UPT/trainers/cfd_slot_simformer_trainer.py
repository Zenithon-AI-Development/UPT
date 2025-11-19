from functools import cached_property

import kappamodules.utils.tensor_cache as tc
import torch
from kappadata.wrappers import ModeWrapper
from torch_scatter import segment_csr

from datasets.collators.cfd_slot_simformer_collator import CfdSlotSimformerCollator
from trainers.cfd_simformer_trainer import CfdSimformerTrainer


class CfdSlotSimformerTrainer(CfdSimformerTrainer):
    @cached_property
    def dataset_mode(self):
        return "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    @property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSlotSimformerCollator)
        # keep track of slot configuration for logging purposes
        self.num_supernodes = collator.collator.M
        self.num_slots_per_supernode = collator.collator.N
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        return input_shape

    @property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(collator.collator, CfdSlotSimformerCollator)
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    def get_trainer_model(self, model=None):
        return self.Model(model=model, trainer=self)

    class Model(CfdSimformerTrainer.Model):
        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch

            to_device = self.to_device
            data = dict(
                x=to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                timestep=to_device(item="timestep", batch=batch, dataset_mode=dataset_mode),
                velocity=to_device(item="velocity", batch=batch, dataset_mode=dataset_mode),
                target=to_device(item="target", batch=batch, dataset_mode=dataset_mode),
                geometry2d=to_device(item="geometry2d", batch=batch, dataset_mode=dataset_mode),
                query_pos=to_device(item="query_pos", batch=batch, dataset_mode=dataset_mode),
                mesh_pos=to_device(item="mesh_pos", batch=batch, dataset_mode=dataset_mode),
            )

            mesh_edges = ModeWrapper.get_item(mode=dataset_mode, item="mesh_edges", batch=batch)
            if mesh_edges is not None:
                data["mesh_edges"] = mesh_edges.to(self.model.device, non_blocking=True)
            else:
                data["mesh_edges"] = None

            data["batch_idx_cells"] = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            data["unbatch_idx"] = ctx["unbatch_idx"].to(self.model.device, non_blocking=True)
            data["unbatch_select"] = ctx["unbatch_select"].to(self.model.device, non_blocking=True)
            data["subnode_feats"] = ctx["subnode_feats"].to(self.model.device, non_blocking=True)
            data["subnode_mask"] = ctx["subnode_mask"].to(self.model.device, non_blocking=True)
            data["slot2cell"] = ctx["slot2cell"].to(self.model.device, non_blocking=True).long()

            B, T, M, _ = data["subnode_mask"].shape
            data["batch_idx_supernodes"] = torch.arange(B * T, device=self.model.device).repeat_interleave(M)
            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch=batch)

            x = data.pop("x")
            target = data.pop("target")
            subnode_feats = data.pop("subnode_feats")
            subnode_mask = data.pop("subnode_mask")
            slot2cell = data.pop("slot2cell")
            batch_idx_supernodes = data.pop("batch_idx_supernodes")
            batch_idx_cells = data.pop("batch_idx_cells")
            timestep = data.pop("timestep")
            velocity = data.pop("velocity")
            geometry2d = data.pop("geometry2d")
            query_pos = data.pop("query_pos")
            mesh_pos = data.pop("mesh_pos")
            mesh_edges = data.pop("mesh_edges")
            unbatch_idx = data.pop("unbatch_idx")
            unbatch_select = data.pop("unbatch_select")

            batch_size = batch_idx_cells.max().item() + 1

            forward_kwargs = {}
            if self.trainer.reconstruct_from_target:
                forward_kwargs["target"] = target
            model_outputs = self.model(
                subnode_feats=subnode_feats,
                subnode_mask=subnode_mask,
                slot2cell=slot2cell,
                timestep=timestep,
                velocity=velocity,
                batch_idx=batch_idx_supernodes,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                detach_reconstructions=self.trainer.detach_reconstructions,
                reconstruct_prev_x=self.trainer.reconstruct_prev_x_weight > 0,
                reconstruct_dynamics=self.trainer.reconstruct_dynamics_weight > 0,
                **forward_kwargs,
            )

            if not torch.isfinite(model_outputs["x_hat"]).all():
                raise RuntimeError("NaN or Inf detected in model output 'x_hat'")

            if not self._dump_done:
                out_dir = self.trainer.path_provider.stage_output_path / "tensors"
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(x.detach().cpu(), out_dir / f"{self._dump_counter:04d}_x.pt")
                torch.save(target.detach().cpu(), out_dir / f"{self._dump_counter:04d}_target.pt")
                torch.save(model_outputs["x_hat"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_xhat.pt")
                torch.save(timestep.detach().cpu(), out_dir / f"{self._dump_counter:04d}_timestep.pt")
                torch.save(velocity.detach().cpu(), out_dir / f"{self._dump_counter:04d}_velocity.pt")
                torch.save(subnode_feats.detach().cpu(), out_dir / f"{self._dump_counter:04d}_subnode_feats.pt")
                torch.save(subnode_mask.detach().cpu(), out_dir / f"{self._dump_counter:04d}_subnode_mask.pt")
                torch.save(slot2cell.detach().cpu(), out_dir / f"{self._dump_counter:04d}_slot2cell.pt")
                torch.save(geometry2d.detach().cpu(), out_dir / f"{self._dump_counter:04d}_geometry2d.pt")
                torch.save(query_pos.detach().cpu(), out_dir / f"{self._dump_counter:04d}_querypos.pt")
                torch.save(mesh_pos.detach().cpu(), out_dir / f"{self._dump_counter:04d}_meshpos.pt")
                condition = model_outputs.get("condition")
                torch.save(
                    condition.detach().cpu() if condition is not None else None,
                    out_dir / f"{self._dump_counter:04d}_condition.pt",
                )
                torch.save(model_outputs["prev_dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_prev_dynamics.pt")
                torch.save(model_outputs["dynamics"].detach().cpu(), out_dir / f"{self._dump_counter:04d}_dynamics.pt")
                self._dump_counter += 1
                self._dump_done = True

            infos = {}
            losses = {}

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
                query_pos_len = query_pos.size(1)
                query_batch_idx = torch.arange(batch_size, device=self.model.device).repeat_interleave(query_pos_len)
                indices, counts = query_batch_idx.unique(return_counts=True)
                padded_counts = torch.zeros(len(indices) + 1, device=counts.device, dtype=counts.dtype)
                padded_counts[indices + 1] = counts
                indptr = padded_counts.cumsum(dim=0)
                losses["x_hat"] = segment_csr(src=x_hat_loss.mean(dim=1), indptr=indptr, reduce="mean")
            else:
                raise NotImplementedError
            total_loss = losses["x_hat"]

            batch_idx = batch_idx_cells

            if self.trainer.reconstruct_prev_x_weight > 0:
                num_channels = model_outputs["prev_x_hat"].size(1)
                prev_x_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_x_hat"],
                    target=x[:, -num_channels:],
                    reduction="none",
                )
                if reduction == "mean":
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
                if reduction == "mean":
                    dynamics_hat_mask = timestep != max_timestep
                    if dynamics_hat_mask.sum() > 0:
                        dynamics_hat_loss = dynamics_hat_loss[dynamics_hat_mask].mean()
                    else:
                        dynamics_hat_loss = tc.zeros(size=(1,), device=timestep.device)
                elif reduction == "mean_per_sample":
                    dynamics_hat_loss[timestep == max_timestep] = 0.0
                    dynamics_hat_loss = dynamics_hat_loss.flatten(start_dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
                losses["dynamics_hat"] = dynamics_hat_loss
                total_loss = total_loss + self.trainer.reconstruct_dynamics_weight * dynamics_hat_loss

            infos.setdefault("degree/input", 0.0)
            return dict(total=total_loss, **losses), infos

