# src/datasets/well_zpinch_dataset.py
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import os
import math
import h5py
import torch
import einops
import numpy as np

from .base.dataset_base import DatasetBase


def _discover_array_key(h5: h5py.File, prefer: Optional[str]) -> str:
    """
    Find the dataset key to read.
    - If `prefer` is provided and exists, use it.
    - Else, pick the first dataset with ndim in {4, 5} and at least 3 spatial-ish dims.
    """
    if prefer is not None and prefer in h5:
        return prefer
    # breadth-first over file tree
    q = [h5]
    while q:
        g = q.pop(0)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                if v.ndim in (4, 5):
                    return v.name  # full path
            elif isinstance(v, h5py.Group):
                q.append(v)
    raise KeyError("Could not auto-discover a dataset inside HDF5 with ndim ∈ {4,5}.")


def _canonize_axes(arr: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """
    Reorder common shapes to canonical (T, H, W, C).
    Accepts (T, H, W, C) or (T, C, H, W).
    Returns (view, T, H, W, C) without copying when possible.
    """
    if arr.ndim == 5:
        # Interpret (N_traj, T, H, W, C) or (N_traj, T, C, H, W)
        # We'll assume caller indexes traj -> gets a 4D array here.
        raise ValueError("Pass a single-trajectory slice (4D) to _canonize_axes.")
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array after choosing one trajectory; got shape {arr.shape}")

    # try (T, H, W, C)
    if arr.shape[-1] <= 32 and arr.shape[1] == arr.shape[2]:
        T, H, W, C = arr.shape
        return arr, T, H, W, C
    # try (T, C, H, W)
    if arr.shape[1] <= 32 and arr.shape[2] == arr.shape[3]:
        T, C, H, W = arr.shape
        view = np.transpose(arr, (0, 2, 3, 1))  # (T, H, W, C)
        return view, T, H, W, C
    # fallback: attempt to infer the channel axis as the smallest non-time dim
    T = arr.shape[0]
    rest = arr.shape[1:]
    c_ax = 1 + int(np.argmin(rest))
    perm = (0,) + tuple(i for i in range(1, arr.ndim) if i != c_ax) + (c_ax,)
    view = np.transpose(arr, perm)
    T, H, W, C = view.shape
    return view, T, H, W, C


class WellZpinchDataset(DatasetBase):
    """
    Reader for z-pinch HDF5 trajectories, matching SimFormer/UPT expectations.

    File layout expectations:
      - Split folders:
          {root}/{split}/*.hdf5
      - Each file contains one trajectory array shaped either:
          (T, H, W, C)    or    (T, C, H, W)
        (If there are groups, set `h5_key` to the dataset path, e.g. '/data/fields')

    Modes:
      * Training: num_input_timesteps = k (int)
          x: (N, k*C), target: (N, C)  at time t+1, using window ending at t
      * Rollout: num_input_timesteps = .inf  (with max_num_timesteps=T)
          x: (N, C)  at t=0, target: (N, C, T-1) for t=1..T-1
    """

    def __init__(
        self,
        root: Union[str, os.PathLike] = "/home/workspace/projects/data/datasets_david/datasets/zpinch/data",
        split: str = "train",                # 'train' | 'valid' | 'test'
        h5_glob: str = "*.hdf5",
        h5_key: Optional[str] = None,        # path to dataset inside HDF5; auto-discover if None
        num_input_timesteps: Union[int, float] = 4,  # or .inf for rollout
        num_output_timesteps: int = 1,       # keep 1 for training
        max_num_timesteps: Optional[int] = None,  # if None, inferred from file
        max_num_sequences: Optional[int] = None,  # limit number of files
        # normalization / clamping
        norm: str = "none",                  # 'none' | 'mean0std1' (requires mean/std below)
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        clamp: Optional[float] = None,       # None or >0
        clamp_mode: str = "log",             # 'hard' | 'log'
        # coordinate scaling (kept compatible with CFD codepaths)
        grid_scaling: Tuple[float, float] = (200.0, 300.0),
        **kwargs,
    ):
        # swallow CFD-only keys so DatasetBase doesn't choke
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
        self.h5_key = h5_key
        self._rollout = (num_input_timesteps == float("inf"))
        self.k_in = None if self._rollout else int(num_input_timesteps)
        self.k_out = int(num_output_timesteps)
        self.max_num_sequences = max_num_sequences
        self._stats_mode = norm
        self.clamp = None if clamp in (None, 0, 0.0) else float(clamp)
        self.clamp_mode = clamp_mode

        # files
        split_dir = self.root / split
        files = sorted([p for p in split_dir.glob(h5_glob) if p.is_file()])
        if len(files) == 0:
            raise FileNotFoundError(f"No HDF5 files found in {split_dir.as_posix()} with pattern '{h5_glob}'.")
        if self.max_num_sequences is not None:
            files = files[: self.max_num_sequences]
        self.files = files

        # Probe the first file for T,H,W,C and dataset key
        with h5py.File(self.files[0], "r") as h5:
            key = _discover_array_key(h5, self.h5_key)
            arr = h5[key]
            # accept full trajectory arrays (T,H,W,C) or (T,C,H,W)
            view, T, H, W, C = _canonize_axes(arr[...])  # small cost once at init
            self._h5_key = key
            self.T_full, self.H, self.W, self.C = int(T), int(H), int(W), int(C)

        # Time bookkeeping
        if max_num_timesteps is not None:
            self.T_full = min(self.T_full, int(max_num_timesteps))
        self._num_transitions = self.T_full - 1
        if not self._rollout:
            assert self.k_in >= 1 and self.k_in <= self.T_full - 1, \
                f"num_input_timesteps={self.k_in} invalid for T={self.T_full}"

        # Coordinates (same scaling as your CFD dataset for compatibility)
        max_x, max_y = grid_scaling
        xs = torch.linspace(0.0, max_x, self.W)   # width
        ys = torch.linspace(0.0, max_y, self.H)   # height
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

        self.eps = 1e-6
        if norm == "mean0std1_auto":
            stats_file = self.root / f".zpinch_stats_{self.split}.pt"
            # if stats_file.exists():
            #     d = torch.load(stats_file)
            #     self.mean, self.std = d["mean"], d["std"]
            # else:
            self.mean, self.std = self._compute_dataset_stats()
            # avoid zero std
            self.std = self.std.clamp_min(self.eps)
            torch.save({"mean": self.mean, "std": self.std}, stats_file)
            self._stats_mode = "mean0std1"

        # Normalization
        # if self._stats_mode == "mean0std1":
        #     if mean is None or std is None:
        #         raise ValueError(
        #             "norm='mean0std1' requires `mean` and `std` (length C). "
        #             "Either set norm='none' or provide statistics."
        #         )
        #     mean = torch.tensor(mean, dtype=torch.float32)
        #     std = torch.tensor(std, dtype=torch.float32)
        #     assert mean.numel() == self.C and std.numel() == self.C, \
        #         f"mean/std must have length C={self.C}"
        #     self.mean = mean
        #     self.std = std
        # else:
        #     self.mean = torch.zeros(self.C, dtype=torch.float32)
        #     self.std = torch.ones(self.C, dtype=torch.float32)

        # --- Normalization
        if self._stats_mode == "mean0std1":
            # If stats were computed in 'mean0std1_auto', self.mean/std already exist.
            if getattr(self, "mean", None) is not None and getattr(self, "std", None) is not None:
                pass  # use the computed attributes
            else:
                if mean is None or std is None:
                    raise ValueError(
                        "norm='mean0std1' requires `mean` and `std` (length C). "
                        "Either set norm='none' or provide statistics."
                    )
                mean = torch.tensor(mean, dtype=torch.float32)
                std = torch.tensor(std, dtype=torch.float32)
                assert mean.numel() == self.C and std.numel() == self.C, \
                    f"mean/std must have length C={self.C}"
                self.mean = mean
                self.std = std
        else:
            # 'none' etc.: identity stats
            self.mean = torch.zeros(self.C, dtype=torch.float32)
            self.std = torch.ones(self.C, dtype=torch.float32)

    def _compute_dataset_stats(self):
        # per-channel Welford over all (T,H,W) samples in this split
        count = 0
        mean = torch.zeros(self.C, dtype=torch.float32)
        M2   = torch.zeros(self.C, dtype=torch.float32)
        for fp in self.files:
            traj = self._load_traj(fp)  # (T,H,W,C), float32
            x = traj.reshape(-1, self.C)  # collapse T,H,W
            # batch update (Chan) for speed:
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

    # ---------------- Kappa/UPT plumbing ----------------
    def __len__(self):
        # Each file is one sample; timestep is chosen internally (like CFD)
        return len(self.files)

    def getshape_timestep(self):
        # used by the conditioner; 1-D shape describing the range length
        return (self._num_transitions,)

    def getitem_timestep(self, idx, ctx=None):
        # pick a timestep deterministically from idx; trainer can also shuffle indices
        # window ends at t in [k_in-1, T_full-2]; target at t+1
        if self._rollout:
            return torch.tensor(0, dtype=torch.long)
        # distribute indices across transitions to cover the range
        base = int(idx) % self._num_transitions
        t = max(self.k_in - 1, base)
        t = min(t, self.T_full - 2)  # ensure we have a t+1
        return torch.tensor(t, dtype=torch.long)

    # shapes for collator/trainer
    def getshape_x(self):       return self._shape_x
    def getshape_target(self):  return self._shape_y

    # positions & geometry
    def getitem_all_pos(self, idx, ctx=None):   return self._mesh_pos_xy
    def getitem_mesh_pos(self, idx, ctx=None):  return self._mesh_pos_xy
    def getitem_query_pos(self, idx, ctx=None): return self._mesh_pos_xy
    def getitem_mesh_edges(self, idx, ctx=None): return None
    def getitem_geometry2d(self, idx, ctx=None): return self._geometry2d
    def getshape_geometry2d(self):               return (2, self.H, self.W)

    # conditioner extras (dummy velocity like before)
    def getitem_velocity(self, idx, ctx=None):   return torch.tensor(0.0, dtype=torch.float32)
    def getshape_velocity(self):                 return ()

    # ---------------- Data access ----------------
    def _load_traj(self, file_path: Path) -> torch.Tensor:
        """
        Returns a torch float32 tensor with shape (T, H, W, C).
        Uses zero-copy from NumPy when feasible.
        """
        with h5py.File(file_path, "r") as h5:
            # Load density and pressure from t0_fields (following WELL structure)
            if 't0_fields' in h5:
                density = torch.from_numpy(h5['t0_fields/density'][:self.T_full]).float()  # (T,H,W)
                pressure = torch.from_numpy(h5['t0_fields/pressure'][:self.T_full]).float()  # (T,H,W)
                
                # Stack as (T,H,W,C=2) - NO log-transform
                t = torch.stack([density, pressure], dim=-1)
                return t
            else:
                # Fallback for other formats
                key = self._h5_key if self._h5_key in h5 else _discover_array_key(h5, self.h5_key)
                ds = h5[key]
                arr = ds[:self.T_full]
                view, T, H, W, C = _canonize_axes(arr)
                t = torch.from_numpy(view).to(torch.float32)
                return t  # (T, H, W, C)

    def _norm_inplace(self, t: torch.Tensor):
        if self._stats_mode != "mean0std1":
            return t
        # t: (..., C) anywhere
        shape = [1] * t.ndim
        shape[-1] = self.C
        t.sub_(self.mean.view(*shape)).div_(self.std.view(*shape))
        return t

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

    def getitem_x(self, idx, ctx=None):
        traj = self._load_traj(self.files[int(idx)])   # (T, H, W, C)
        self._norm_inplace(traj)
        if self._rollout:
            x0 = traj[0]                               # (H, W, C)
            x0 = einops.rearrange(x0, "h w c -> (h w) c")
            return self._clamp_tensor(x0)
        # training window
        t = int(self.getitem_timestep(idx))
        start = max(0, t - self.k_in + 1)
        x_win = traj[start : t + 1]                    # (k, H, W, C)
        x_win = einops.rearrange(x_win, "t h w c -> (h w) (t c)")
        return self._clamp_tensor(x_win)

    def getitem_target(self, idx, ctx=None):
        traj = self._load_traj(self.files[int(idx)])   # (T, H, W, C)
        self._norm_inplace(traj)
        if self._rollout:
            # stack t=1..T-1 along last dim: (N, C, T-1)
            y = traj[1:]                               # (T-1, H, W, C)
            y = einops.rearrange(y, "t h w c -> (h w) c t")
            return self._clamp_tensor(y)
        # training target at t+1
        t = int(self.getitem_timestep(idx))
        y = traj[t + 1]                                # (H, W, C)
        y = einops.rearrange(y, "h w c -> (h w) c")
        return self._clamp_tensor(y)

    # Optional aliases used by some callbacks
    def getitem_target_t0(self, idx, ctx=None):
        traj = self._load_traj(self.files[int(idx)])
        self._norm_inplace(traj)
        x0 = traj[0]
        x0 = einops.rearrange(x0, "h w c -> (h w) c")
        return self._clamp_tensor(x0)

    def getitem_reconstruction_input(self, idx, ctx=None):
        return self.getitem_target(idx, ctx)

    def getitem_reconstruction_output(self, idx, ctx=None):
        return self.getitem_target(idx, ctx)

well_zpinch_dataset = WellZpinchDataset
__all__ = ["WellZpinchDataset", "well_zpinch_dataset"]
