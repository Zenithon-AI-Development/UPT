from pathlib import Path
import hashlib
from typing import Optional, Tuple

import einops
import numpy as np
import torch

from the_well.data import WellDataset

from .base.dataset_base import DatasetBase


STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


class WellTrl2dDataset(DatasetBase):
    """Fresh TRL2D adapter with normalization computed directly from training data."""

    def __init__(
        self,
        well_base_path: str = "/home/workspace/projects/data/datasets_david/datasets/",
        root: Optional[str] = None,
        well_dataset_name: str = "turbulent_radiative_layer_2D",
        split: str = "train",
        num_input_timesteps: float = 4,
        num_output_timesteps: int = 1,
        dt_stride: int = 1,
        square_pad: bool = False,
        max_num_timesteps: Optional[int] = None,
        norm: str = "mean0std1_auto",
        clamp: Optional[float] = None,
        clamp_mode: str = "log",
        grid_scaling: Tuple[float, float] = (200.0, 300.0),
        max_num_sequences: Optional[int] = None,
        **kwargs,
    ):
        for key in [
            "num_input_points",
            "num_input_points_ratio",
            "num_input_points_mode",
            "num_query_points",
            "num_query_points_mode",
            "couple_query_with_input",
            "radius_graph_r",
            "radius_graph_max_num_neighbors",
            "num_supernodes",
            "grid_resolution",
            "standardize_query_pos",
            "version",
        ]:
            kwargs.pop(key, None)
        kwargs.pop("root", None)
        super().__init__(**kwargs)

        self.split = split
        self._rollout = num_input_timesteps == float("inf")
        self.max_num_sequences = max_num_sequences
        self.max_num_timesteps = int(max_num_timesteps) if max_num_timesteps is not None else None
        self.dt_stride = max(1, int(dt_stride))

        if self._rollout:
            if self.max_num_timesteps is None:
                raise ValueError("max_num_timesteps must be provided when num_input_timesteps=.inf (rollout).")
            n_steps_input = 1
            n_steps_output = self.max_num_timesteps - 1
        else:
            n_steps_input = int(num_input_timesteps)
            n_steps_output = int(num_output_timesteps)

        if root is not None:
            root_path = Path(root)
            if (root_path / split).exists():
                dataset_path = root_path.parent
                base_path = dataset_path.parent
                well_dataset_name = dataset_path.name
            else:
                base_path = root_path
            well_base_path = str(base_path)
        else:
            base_path = Path(well_base_path)
        self._stats_id = self._build_stats_id(base_path, well_dataset_name)

        self.well = WellDataset(
            well_base_path=base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            min_dt_stride=self.dt_stride,
            max_dt_stride=self.dt_stride,
            use_normalization=False,
            normalization_type=None,
        )

        self._indices = list(range(len(self.well)))
        if self.max_num_sequences is not None:
            self._indices = self._indices[: self.max_num_sequences]
        if not self._indices:
            raise RuntimeError(f"No samples available for TRL2D split={split}")

        sample0 = self._item(0)
        x0 = torch.as_tensor(sample0["input_fields"], dtype=torch.float32)
        _, H, W, C = x0.shape
        self.H, self.W, self.C = int(H), int(W), int(C)

        metadata = sample0.get("metadata", {})
        total_timesteps = metadata.get("total_timesteps") or metadata.get("num_total_snapshots")
        if total_timesteps is None:
            total_timesteps = 101
        total_timesteps = int(total_timesteps)
        if self.max_num_timesteps is not None:
            total_timesteps = min(total_timesteps, self.max_num_timesteps)
        self._num_total_snapshots = max(total_timesteps, 2)
        self._num_transitions = self._num_total_snapshots - 1

        if self._rollout:
            self._shape_x = (None, self.C)
            self._shape_y = (None, self.C, n_steps_output)
        else:
            self._shape_x = (None, n_steps_input * self.C)
            if n_steps_output == 1:
                self._shape_y = (None, self.C)
            else:
                self._shape_y = (None, n_steps_output * self.C)

        self.clamp = None if clamp in (None, 0, 0.0) else float(clamp)
        self.clamp_mode = clamp_mode
        self.eps = 1e-6

        self._mesh_pos_xy = self._build_mesh(grid_scaling)
        self._geometry2d = torch.zeros(2, self.H, self.W, dtype=torch.float32)
        self._velocity_dummy = torch.tensor(0.0, dtype=torch.float32)

        stats_path = STATS_DIR / f"{self._stats_id}_train.pt"
        self._stats_mode = norm
        if norm == "mean0std1_auto":
            if split == "train":
                stats = self._compute_dataset_stats()
                torch.save(stats, stats_path)
                self._load_stats(stats)
            else:
                if not stats_path.exists():
                    raise FileNotFoundError(
                        f"Normalization stats not found at {stats_path}. Instantiate the training split first."
                    )
                stats = torch.load(stats_path)
                self._load_stats(stats)
            self._stats_mode = "mean0std1"
        else:
            # identity stats (no normalization)
            self.mean = torch.zeros(self.C, dtype=torch.float32)
            self.std = torch.ones(self.C, dtype=torch.float32)

    @staticmethod
    def _build_stats_id(base_path: Path, dataset_name: str) -> str:
        root = (base_path / dataset_name).resolve()
        root_id = hashlib.md5(str(root).encode()).hexdigest()[:12]
        return f"well_trl2d_{root_id}"

    def _build_mesh(self, grid_scaling: Tuple[float, float]) -> torch.Tensor:
        max_x, max_y = grid_scaling
        xs = torch.linspace(0.0, max_x, self.W, dtype=torch.float32)
        ys = torch.linspace(0.0, max_y, self.H, dtype=torch.float32)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        return torch.stack([X.T.reshape(-1), Y.T.reshape(-1)], dim=1).contiguous()

    def _load_stats(self, stats: dict) -> None:
        if "mean" in stats and "std" in stats:
            self.mean = stats["mean"].to(torch.float32)
            self.std = stats["std"].to(torch.float32).clamp_min(self.eps)
            return
        # legacy log stats
        self.mean = torch.zeros(self.C, dtype=torch.float32)
        self.std = torch.ones(self.C, dtype=torch.float32)
        self.mean[:1] = stats["dens_log_mean"].to(torch.float32)
        self.std[:1] = stats["dens_log_std"].to(torch.float32).clamp_min(self.eps)
        self.mean[1:2] = stats["pres_log_mean"].to(torch.float32)
        self.std[1:2] = stats["pres_log_std"].to(torch.float32).clamp_min(self.eps)
        if self.C > 2:
            self.mean[2:] = stats["vel_mean"].to(torch.float32)
            self.std[2:] = stats["vel_std"].to(torch.float32).clamp_min(self.eps)

    def _compute_dataset_stats(self) -> dict:
        count = 0
        mean = torch.zeros(self.C, dtype=torch.float64)
        m2 = torch.zeros(self.C, dtype=torch.float64)

        for idx in self._indices:
            sample = self._item(idx)
            input_fields = torch.as_tensor(sample["input_fields"], dtype=torch.float64)
            output_fields = torch.as_tensor(sample["output_fields"], dtype=torch.float64)
            tensors = [input_fields]
            if output_fields.numel() > 0:
                tensors.append(output_fields)
            combined = torch.cat(tensors, dim=0)
            flat = combined.reshape(-1, self.C)
            if flat.numel() == 0:
                continue
            count, mean, m2 = self._update_stats(count, mean, m2, flat)

        if count < 2:
            raise RuntimeError("Unable to compute statistics for TRL2D dataset (insufficient samples).")
        var = m2 / (count - 1)
        std = var.sqrt().clamp_min(self.eps)
        return {
            "mean": mean.to(torch.float32),
            "std": std.to(torch.float32),
        }

    @staticmethod
    def _update_stats(count, mean, m2, batch):
        batch_count = batch.shape[0]
        if batch_count == 0:
            return count, mean, m2
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean).pow(2)).sum(dim=0)
        if count == 0:
            return batch_count, batch_mean, batch_m2
        delta = batch_mean - mean
        total = count + batch_count
        mean = mean + delta * (batch_count / total)
        m2 = m2 + batch_m2 + delta.pow(2) * (count * batch_count / total)
        return total, mean, m2

    def __len__(self) -> int:
        return len(self._indices)

    def _item(self, idx: int):
        return self.well[self._indices[int(idx)]]

    def getshape_x(self):
        return self._shape_x

    def getshape_target(self):
        return self._shape_y

    def getshape_timestep(self):
        return (self._num_transitions,)

    def getitem_timestep(self, idx, ctx=None):
        if self._rollout:
            return torch.tensor(0, dtype=torch.long)
        return torch.tensor(idx % self._num_transitions, dtype=torch.long)

    def getitem_all_pos(self, idx, ctx=None):
        return self._mesh_pos_xy

    def getitem_mesh_pos(self, idx, ctx=None):
        return self._mesh_pos_xy

    def getitem_query_pos(self, idx, ctx=None):
        return self._mesh_pos_xy

    def getitem_mesh_edges(self, idx, ctx=None):
        return None

    def getitem_geometry2d(self, idx, ctx=None):
        return self._geometry2d

    def getshape_geometry2d(self):
        return (2, self.H, self.W)

    def getitem_velocity(self, idx, ctx=None):
        return self._velocity_dummy

    def getshape_velocity(self):
        return ()

    def getitem_x(self, idx, ctx=None):
        sample = self._item(idx)
        x = torch.as_tensor(sample["input_fields"], dtype=torch.float32)
        x = self._apply_norm(x)
        if self._rollout:
            x0 = einops.rearrange(x[0], "h w c -> (h w) c")
            return self._clamp_if_needed(x0)
        x = einops.rearrange(x, "t h w c -> (h w) (t c)")
        return self._clamp_if_needed(x)

    def getitem_target(self, idx, ctx=None):
        sample = self._item(idx)
        y = torch.as_tensor(sample["output_fields"], dtype=torch.float32)
        y = self._apply_norm(y)
        if self._rollout:
            y = einops.rearrange(y, "t h w c -> (h w) c t")
            return self._clamp_if_needed(y)
        if y.shape[0] == 1:
            y = einops.rearrange(y[0], "h w c -> (h w) c")
        else:
            y = einops.rearrange(y, "t h w c -> (h w) (t c)")
        return self._clamp_if_needed(y)

    def getitem_target_t0(self, idx, ctx=None):
        sample = self._item(idx)
        x0 = torch.as_tensor(sample["input_fields"][0], dtype=torch.float32)
        x0 = self._apply_norm(x0)
        x0 = einops.rearrange(x0, "h w c -> (h w) c")
        return self._clamp_if_needed(x0)

    def getitem_reconstruction_input(self, idx, ctx=None):
        return self.getitem_target(idx, ctx)

    def getitem_reconstruction_output(self, idx, ctx=None):
        return self.getitem_target(idx, ctx)

    def _apply_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._stats_mode != "mean0std1":
            return tensor
        shape = [1] * tensor.ndim
        shape[-1] = self.C
        mean = self.mean.view(*shape).to(tensor.device, tensor.dtype)
        std = self.std.view(*shape).to(tensor.device, tensor.dtype)
        return (tensor - mean) / std

    def _clamp_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.clamp is None:
            return tensor
        if self.clamp_mode == "hard":
            return tensor.clamp(-self.clamp, self.clamp)
        if self.clamp_mode == "log":
            mask = tensor.abs() > self.clamp
            if mask.any():
                values = tensor[mask]
                result = tensor.clone()
                result[mask] = torch.sign(values) * (
                    self.clamp + torch.log1p(values.abs()) - np.log1p(self.clamp)
                )
                return result
            return tensor
        raise NotImplementedError(f"Unsupported clamp_mode '{self.clamp_mode}'")


well_trl2d_dataset = WellTrl2dDataset
__all__ = ["WellTrl2dDataset", "well_trl2d_dataset"]

