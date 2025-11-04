from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List
import os
import math
import h5py
import torch
import einops
import numpy as np

from .base.dataset_base import DatasetBase


def _list_timesteps(f: h5py.File) -> List[str]:
    keys = [k for k in f.keys() if k.startswith("timestep_")]
    # ensure sorted by integer suffix
    def _key(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return -1
    keys.sort(key=_key)
    return keys


class AmrGridDataset(DatasetBase):
    """
    Reader for AMR HDF5 trajectories remeshed to a fixed 64x64 grid per timestep.

    Expected file layout:
      root: directory containing files like 'dataset_amr_*.hdf5'
      each file:
        /time_grid: (T,)
        /timestep_XXXX/fields: (H*W, C)  e.g., (4096, 8) with H=W=64
        (optional) /timestep_XXXX/coordinates: may be empty; we synthesize a regular grid

    Modes:
      * Training: num_input_timesteps = k (int)
          x: (N, k*C), target: (N, C)  at time t+1
      * Rollout: num_input_timesteps = .inf  (with max_num_timesteps=T)
          x: (N, C)  at t=0, target: (N, C, T-1) for t=1..T-1
    """

    def __init__(
        self,
        root: Union[str, os.PathLike] = "/home/workspace/flash/64x64_amr/data",
        split: str = "train",                # 'train' | 'valid' | 'test'
        file_glob: str = "*.hdf5",
        num_input_timesteps: Union[int, float] = 4,  # or .inf for rollout
        num_output_timesteps: int = 1,
        max_num_timesteps: Optional[int] = None,
        max_num_sequences: Optional[int] = None,
        # normalization / clamping
        norm: str = "none",                  # 'none' | 'mean0std1' (requires mean/std below or auto)
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        clamp: Optional[float] = None,       # None or >0
        clamp_mode: str = "log",             # 'hard' | 'log'
        # coordinate scaling to align with CFD conventions
        grid_scaling: Tuple[float, float] = (200.0, 300.0),
        **kwargs,
    ):
        # swallow CFD-only keys
        for k in [
            "num_input_points", "num_input_points_ratio", "num_input_points_mode",
            "num_query_points", "num_query_points_mode", "couple_query_with_input",
            "radius_graph_r", "radius_graph_max_num_neighbors", "num_supernodes",
            "grid_resolution", "standardize_query_pos", "version",
        ]:
            kwargs.pop(k, None)
        super().__init__(**kwargs)

        self.root = Path(root)
        self.split = split
        self.file_glob = file_glob
        self._rollout = (num_input_timesteps == float("inf"))
        self.k_in = None if self._rollout else int(num_input_timesteps)
        self.k_out = int(num_output_timesteps)
        self.max_num_sequences = max_num_sequences
        self._stats_mode = norm
        self.clamp = None if clamp in (None, 0, 0.0) else float(clamp)
        self.clamp_mode = clamp_mode

        # files
        files = sorted([p for p in self.root.glob(self.file_glob) if p.is_file()])
        if len(files) == 0:
            raise FileNotFoundError(f"No files found in {self.root.as_posix()} with pattern '{self.file_glob}'.")
        # naive split by index if needed
        if split in ("train", "valid", "test") and len(files) >= 5:
            n = len(files)
            n_train = int(n * 0.8)
            n_valid = int(n * 0.1)
            if split == "train":
                files = files[:n_train]
            elif split == "valid":
                files = files[n_train:n_train + n_valid]
            else:
                files = files[n_train + n_valid:]
        if self.max_num_sequences is not None:
            files = files[: self.max_num_sequences]
        self.files = files

        # probe first file for T, H, W, C
        with h5py.File(self.files[0], "r") as f:
            steps = _list_timesteps(f)
            if len(steps) == 0:
                raise ValueError("No timestep_XXXX groups found.")
            ds = f[steps[0]]["fields"]  # (N, C)
            n_points, C = ds.shape
            H = W = int(math.isqrt(int(n_points)))
            assert H * W == n_points, f"fields length {n_points} not a perfect square"
            T = len(steps)
            self.T_full, self.H, self.W, self.C = int(T), int(H), int(W), int(C)

        # time bookkeeping
        if max_num_timesteps is not None:
            self.T_full = min(self.T_full, int(max_num_timesteps))
        self._num_transitions = self.T_full - 1
        if not self._rollout:
            assert self.k_in >= 1 and self.k_in <= self.T_full - 1, \
                f"num_input_timesteps={self.k_in} invalid for T={self.T_full}"

        # coordinates (synthetic regular grid)
        max_x, max_y = grid_scaling
        xs = torch.linspace(0.0, max_x, self.W)
        ys = torch.linspace(0.0, max_y, self.H)
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        self._mesh_pos_xy = torch.stack([X.T.reshape(-1), Y.T.reshape(-1)], dim=1).contiguous()
        self._geometry2d = torch.zeros(2, self.H, self.W, dtype=torch.float32)

        # Declared shapes
        if self._rollout:
            self._shape_x = (None, self.C)
            self._shape_y = (None, self.C, self._num_transitions)
        else:
            self._shape_x = (None, self.k_in * self.C)
            self._shape_y = (None, self.C if self.k_out == 1 else self.k_out * self.C)

        # normalization
        self.eps = 1e-6
        if self._stats_mode == "mean0std1_auto":
            self.mean, self.std = self._compute_dataset_stats()
            self.std = self.std.clamp_min(self.eps)
            self._stats_mode = "mean0std1"
        elif self._stats_mode == "mean0std1":
            if mean is None or std is None:
                raise ValueError("norm='mean0std1' requires mean and std")
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std, dtype=torch.float32)
            assert self.mean.numel() == self.C and self.std.numel() == self.C
        else:
            self.mean = torch.zeros(self.C, dtype=torch.float32)
            self.std = torch.ones(self.C, dtype=torch.float32)

    def _load_traj(self, file_path: Path) -> torch.Tensor:
        """Load up to T_full timesteps and return (T, H, W, C) float32 tensor."""
        frames: List[np.ndarray] = []
        with h5py.File(file_path, "r") as f:
            steps = _list_timesteps(f)[: self.T_full]
            for s in steps:
                arr = f[s]["fields"][:]  # (N, C)
                # reshape to (H, W, C) assuming row-major
                t = arr.reshape(self.H, self.W, self.C)
                frames.append(t)
        tnp = np.stack(frames, axis=0)  # (T, H, W, C)
        return torch.from_numpy(tnp).to(torch.float32)

    def _norm_inplace(self, t: torch.Tensor):
        if self._stats_mode != "mean0std1":
            return t
        shape = [1, 1, 1, self.C]
        t.sub_(self.mean.view(*shape)).div_(self.std.view(*shape))
        return t

    def _compute_dataset_stats(self):
        count = 0
        mean = torch.zeros(self.C, dtype=torch.float32)
        M2 = torch.zeros(self.C, dtype=torch.float32)
        for fp in self.files:
            traj = self._load_traj(fp)  # (T,H,W,C)
            x = traj.reshape(-1, self.C)
            n = x.shape[0]
            if n == 0:
                continue
            batch_mean = x.mean(dim=0)
            batch_M2 = ((x - batch_mean).pow(2)).sum(dim=0)
            if count == 0:
                mean = batch_mean
                M2 = batch_M2
                count = n
            else:
                delta = batch_mean - mean
                total = count + n
                mean = mean + delta * (n / total)
                M2 = M2 + batch_M2 + delta.pow(2) * (count * n / total)
                count = total
        var = M2 / max(count - 1, 1)
        std = var.sqrt()
        return mean, std

    # ------------- Kappa/UPT plumbing -------------
    def __len__(self):
        return len(self.files)

    def getshape_timestep(self):
        return (self._num_transitions,)

    def getitem_timestep(self, idx, ctx=None):
        if self._rollout:
            return torch.tensor(0, dtype=torch.long)
        base = int(idx) % self._num_transitions
        t = max((self.k_in or 1) - 1, base)
        t = min(t, self.T_full - 2)
        return torch.tensor(t, dtype=torch.long)

    def getshape_x(self):       return self._shape_x
    def getshape_target(self):  return self._shape_y

    # positions & geometry
    def getitem_all_pos(self, idx, ctx=None):   return self._mesh_pos_xy
    def getitem_mesh_pos(self, idx, ctx=None):  return self._mesh_pos_xy
    def getitem_query_pos(self, idx, ctx=None): return self._mesh_pos_xy
    def getitem_mesh_edges(self, idx, ctx=None): return None
    def getitem_geometry2d(self, idx, ctx=None): return self._geometry2d
    def getshape_geometry2d(self):               return (2, self.H, self.W)

    # conditioner extras
    def getitem_velocity(self, idx, ctx=None):   return torch.tensor(0.0, dtype=torch.float32)
    def getshape_velocity(self):                 return ()

    def _clamp_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.clamp is None:
            return x
        if self.clamp_mode == "hard":
            return x.clamp_(-self.clamp, self.clamp)
        if self.clamp_mode == "log":
            mask = x.abs() > self.clamp
            if mask.any():
                vals = x[mask]
                x = x.clone()
                x[mask] = torch.sign(vals) * (
                    self.clamp + torch.log1p(vals.abs()) - math.log1p(self.clamp)
                )
            return x
        raise NotImplementedError

    # data access
    def _load_and_norm(self, file_idx: int) -> torch.Tensor:
        traj = self._load_traj(self.files[file_idx])   # (T,H,W,C)
        self._norm_inplace(traj)
        return traj

    def getitem_x(self, idx, ctx=None):
        traj = self._load_and_norm(int(idx))
        if self._rollout:
            x0 = traj[0]                               # (H, W, C)
            x0 = einops.rearrange(x0, "h w c -> (h w) c")
            return self._clamp_tensor(x0)
        t = int(self.getitem_timestep(idx))
        start = max(0, t - (self.k_in or 1) + 1)
        x_win = traj[start : t + 1]                    # (k, H, W, C)
        x_win = einops.rearrange(x_win, "t h w c -> (h w) (t c)")
        return self._clamp_tensor(x_win)

    def getitem_target(self, idx, ctx=None):
        traj = self._load_and_norm(int(idx))
        if self._rollout:
            y = traj[1:]                               # (T-1, H, W, C)
            y = einops.rearrange(y, "t h w c -> (h w) c t")
            return self._clamp_tensor(y)
        t = int(self.getitem_timestep(idx))
        y = traj[t + 1]                                # (H, W, C)
        y = einops.rearrange(y, "h w c -> (h w) c")
        return self._clamp_tensor(y)

    def getitem_target_t0(self, idx, ctx=None):
        traj = self._load_and_norm(int(idx))
        x0 = traj[0]
        x0 = einops.rearrange(x0, "h w c -> (h w) c")
        return self._clamp_tensor(x0)


# UPT factory alias
amr_grid_dataset = AmrGridDataset
__all__ = ["AmrGridDataset", "amr_grid_dataset"]


