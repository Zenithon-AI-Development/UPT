import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import torch
import einops
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph, radius

from .base.dataset_base import DatasetBase


def _list_h5(root: Path) -> List[Path]:
    files: List[Path] = []
    for suf in ("*.h5", "*.hdf5", "*.hdf"):
        files.extend(sorted(root.rglob(suf)))
    return files


def _list_timesteps(f: h5py.File) -> List[str]:
    return sorted([k for k in f.keys() if k.startswith("timestep_")])


class AmrPointCloudDataset(DatasetBase):
    """
    Lagrangian point-cloud dataset for AMR trajectories with variable points per timestep.

    Expected per-file structure:
        - AMR format: timestep_XXXX/coordinates (N_t, 2), timestep_XXXX/fields (N_t, C)
        - Uniform format: input_fields (T, H, W, C), optional space_grid (H, W, 2)
        - Z-pinch format: t0_fields/density (T,H,W), t0_fields/pressure (T,H,W), etc.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        split: str = "train",
        n_input_timesteps: int = 4,
        n_pushforward_timesteps: int = 0,
        graph_mode: str = 'radius_graph_with_supernodes',
        radius_graph_r: float = 0.1,
        radius_graph_max_num_neighbors: int = 32,
        n_supernodes: Optional[int] = 2048,
        num_points_range: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
        pos_scale: float = 200.0,
        test_mode: Optional[str] = None,
        **kwargs,
    ):
        if "test_mode" in kwargs:
            kwargs.pop("test_mode", None)
        super().__init__(**kwargs)
        assert split in ["train", "valid", "test"], f"invalid split {split}"
        assert n_input_timesteps > 1, "n_input_timesteps must be > 1"
        assert graph_mode in ['knn', 'radius_graph', 'radius_graph_with_supernodes']

        self.root = Path(root)
        self.split = split
        self.n_input_timesteps = int(n_input_timesteps)
        self.n_pushforward_timesteps = int(n_pushforward_timesteps)
        self.graph_mode = graph_mode
        self.radius_graph_r = float(radius_graph_r)
        self.radius_graph_max_num_neighbors = int(radius_graph_max_num_neighbors)
        self.n_supernodes = int(n_supernodes) if n_supernodes else None
        self.num_points_range = num_points_range
        self.seed = seed
        self.pos_scale = float(pos_scale)
        self.test_mode = test_mode or ("full_traj" if split in ("valid", "test") else "parts_traj")

        # Prefer explicit subdirectories for splits if present
        split_dir = self.root / split
        if split_dir.exists() and split_dir.is_dir():
            files_all = _list_h5(split_dir)
        else:
            # fallback: gather all files and hash-split deterministically
            files_all = _list_h5(self.root)
            if len(files_all) == 0:
                raise FileNotFoundError(f"No HDF5 files found under {self.root}")

            def _split_bucket(p: Path) -> str:
                h = abs(hash(p.as_posix())) % 100
                if h < 80:
                    return "train"
                if h < 90:
                    return "valid"
                return "test"

            files_all = [p for p in files_all if _split_bucket(p) == split]

        # Detect usable files and record format/dims
        self._file_info: Dict[int, Dict] = {}
        files: List[Path] = []
        for p in files_all:
            fmt = None
            info: Dict = {}
            try:
                with h5py.File(p, "r") as f:
                    steps = _list_timesteps(f)
                    if len(steps) > 0:
                        # AMR-style
                        ds = f[steps[0]].get("fields")
                        if isinstance(ds, h5py.Dataset):
                            n_points, C = ds.shape
                            info = {"format": "amr", "C": int(C)}
                            fmt = "amr"
                        else:
                            fmt = None
                    elif "input_fields" in f:
                        ds = f["input_fields"]
                        # expect (T, H, W, C)
                        if isinstance(ds, h5py.Dataset) and ds.ndim == 4:
                            T, H, W, C = ds.shape
                            info = {"format": "uniform", "C": int(C), "H": int(H), "W": int(W), "T": int(T)}
                            fmt = "uniform"
                    else:
                        # Check for z-pinch style: t0_fields/, t1_fields/ groups
                        t_groups = sorted([k for k in f.keys() if k.startswith('t') and k.endswith('_fields') and k[1:-7].isdigit()])
                        if len(t_groups) > 0:
                            # Read first group to get field names and shape
                            grp = f[t_groups[0]]
                            if hasattr(grp, 'keys'):
                                field_names = list(grp.keys())
                                if len(field_names) > 0:
                                    first_field = grp[field_names[0]]
                                    if isinstance(first_field, h5py.Dataset) and first_field.ndim == 3:
                                        T, H, W = first_field.shape
                                        C = len(field_names)
                                        info = {"format": "zpinch", "C": int(C), "H": int(H), "W": int(W), "T": int(T), "field_names": field_names, "n_time_groups": len(t_groups)}
                                        fmt = "zpinch"
            except Exception:
                fmt = None
            if fmt is not None:
                idx = len(files)
                files.append(p)
                self._file_info[idx] = info
        if len(files) == 0:
            raise FileNotFoundError(f"No files remain for split '{split}' under {self.root}")
        self.files = files

        # Probe first usable file for channels
        if len(self.files) == 0:
            raise FileNotFoundError(f"No usable HDF5 files found under {self.root}/{split}")
        info0 = self._file_info[0]
        self.num_channels = int(info0.get("C", 0))
        assert self.num_channels > 0, "could not infer channel count"

        # Normalization running stats placeholders (compute optionally)
        self.vel_mean = torch.zeros(2, dtype=torch.float32)
        self.vel_std = torch.ones(2, dtype=torch.float32)
        self.acc_mean = torch.zeros(2, dtype=torch.float32)
        self.acc_std = torch.ones(2, dtype=torch.float32)

        # Bounds/box unknown -> scale via min/max later when needed
        self.pos_offset = torch.zeros(2, dtype=torch.float32)
        self.pos_scale_vec = torch.ones(2, dtype=torch.float32) * (self.pos_scale / 200.0)

        # Minimal metadata expected by rollout callbacks
        # Bounds in a unit square; dt/write_every/dx set to 1; 2D domain
        self.metadata: Dict[str, Union[float, int, List[List[float]]]] = {
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
            "dt": 1.0,
            "write_every": 1,
            "dx": 1.0,
            "dim": 2,
            "sequence_length_train": 1024,
            "sequence_length_test": 1024,
        }
        self.box = torch.tensor(self.metadata['bounds'])[:, 1] - torch.tensor(self.metadata['bounds'])[:, 0]

        # derive sequence sampling lengths for train vs eval
        self._init_sequence_lengths()
        self.n_seq = min(self.file_lengths.values()) if self.file_lengths else 0

        # input is velocities (n_input_timesteps-1)*2, output is next velocity 2
        self._shape_x = (None, (self.n_input_timesteps - 1) * 2)
        # target is next-step velocity (2); decoder expects (num_channels, None)
        self._shape_y = (2, None)

    # ---------- sequence bookkeeping ----------
    def _init_sequence_lengths(self):
        # compute per-file sequence lengths
        self.file_lengths: Dict[int, int] = {}
        for i, p in enumerate(self.files):
            try:
                info = self._file_info[i]
                if info.get("format") == "amr":
                    with h5py.File(p, "r") as f:
                        T = len(_list_timesteps(f))
                elif info.get("format") in ("uniform", "zpinch"):
                    T = int(info.get("T", 0))
                else:
                    T = 0
            except Exception:
                T = 0
            self.file_lengths[i] = int(T)

        # conservative common sequence length across files
        min_T = min(self.file_lengths.values()) if len(self.file_lengths) > 0 else 0
        self.metadata["sequence_length_train"] = int(min_T)
        self.metadata["sequence_length_test"] = int(min_T)

    def _locate_train_index(self, idx: int) -> Tuple[int, int]:
        # map a single integer idx to (file_idx, timestep within file)
        # use windowed logic for training
        for i, p in enumerate(self.files):
            L = self.file_lengths[i]
            if L < (self.n_input_timesteps + self.n_pushforward_timesteps + 1):
                continue
            nwin = L - (self.n_input_timesteps + self.n_pushforward_timesteps)
            if idx < nwin:
                return i, idx
            idx -= nwin
        return 0, 0

    def __len__(self):
        # total training windows across all files
        if self.split == "train":
            total = 0
            for i, L in self.file_lengths.items():
                nwin = max(0, L - (self.n_input_timesteps + self.n_pushforward_timesteps))
                total += nwin
            return total
        else:
            # test/valid: one sample per file (full trajectory)
            return len(self.files)

    def _read_window(self, file_idx: int, start_idx: int, end_idx: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return positions[T, N_t, 2], fields[T, N_t, C] lists for a window."""
        p = self.files[file_idx]
        info = self._file_info[file_idx]
        Ts: List[np.ndarray] = []
        Fs: List[np.ndarray] = []
        with h5py.File(p, "r") as f:
            if info.get("format") == "amr":
                steps = _list_timesteps(f)
                steps = steps[start_idx:end_idx]
                for s in steps:
                    coords = f[s]["coordinates"][:]
                    fields = f[s]["fields"][:]
                    if coords.size == 0:
                        n = fields.shape[0]
                        side = int(np.sqrt(n))
                        if side * side != n:
                            coords = np.zeros((0, 2), dtype=np.float32)
                            fields = np.zeros((0, fields.shape[1]), dtype=np.float32)
                        else:
                            xs = np.linspace(0.0, 1.0, side, dtype=np.float32)
                            ys = np.linspace(0.0, 1.0, side, dtype=np.float32)
                            X, Y = np.meshgrid(xs, ys, indexing="xy")
                            coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
                    Ts.append(coords.astype(np.float32))
                    Fs.append(fields.astype(np.float32))
            elif info.get("format") == "uniform":
                # uniform grid: input_fields (T,H,W,C), optional space_grid (H,W,2)
                ds = f["input_fields"]
                Ttot = ds.shape[0]
                s0 = max(0, start_idx)
                s1 = min(end_idx, Ttot)
                frames = ds[s0:s1]  # (t, H, W, C)
                if "space_grid" in f:
                    sg = f["space_grid"][:]  # (H,W,2)
                    coords_base = sg.reshape(-1, 2).astype(np.float32)
                else:
                    _, H, W, _ = ds.shape
                    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
                    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
                    X, Y = np.meshgrid(xs, ys, indexing="xy")
                    coords_base = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
                for t in range(frames.shape[0]):
                    f2 = frames[t].reshape(-1, frames.shape[-1]).astype(np.float32)
                    Ts.append(coords_base)
                    Fs.append(f2)
            elif info.get("format") == "zpinch":
                # z-pinch style: t0_fields/density, t0_fields/pressure, etc. each (T,H,W)
                field_names = info.get("field_names", [])
                n_time_groups = info.get("n_time_groups", 1)
                H, W = info.get("H", 32), info.get("W", 32)
                
                # Generate coordinates grid once
                xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
                ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
                X, Y = np.meshgrid(xs, ys, indexing="xy")
                coords_base = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1).astype(np.float32)
                
                # Read from appropriate time groups
                # Each t*_fields group contains multiple fields with shape (T,H,W)
                # We'll read from t0_fields by default (contains the primary timeseries)
                grp = f["t0_fields"]
                first_field = grp[field_names[0]]
                Ttot = first_field.shape[0]
                s0 = max(0, start_idx)
                s1 = min(end_idx, Ttot)
                
                # Stack all fields into (T, H, W, C)
                for t in range(s0, s1):
                    fields_at_t = []
                    for fname in field_names:
                        field_data = grp[fname][t, :, :]  # (H, W)
                        fields_at_t.append(field_data.reshape(-1))  # (H*W,)
                    f2 = np.stack(fields_at_t, axis=1).astype(np.float32)  # (H*W, C)
                    Ts.append(coords_base)
                    Fs.append(f2)
        # Ragged; return as lists -> callers index per timestep
        # Convert to torch using torch.tensor (creates independent storage, unlike from_numpy)
        T_list = [torch.tensor(t, dtype=torch.float32) for t in Ts]
        F_list = [torch.tensor(f, dtype=torch.float32) for f in Fs]
        return T_list, F_list

    # ---------- normalization helpers ----------
    def scale_pos(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos - self.pos_offset.to(pos.device)) * self.pos_scale_vec.to(pos.device)

    def _vel(self, positions: torch.Tensor) -> torch.Tensor:
        v = positions[1:, :, :] - positions[:-1, :, :]
        v = (v - self.vel_mean.to(v.device)) / self.vel_std.to(v.device)
        return v

    def _acc(self, positions: torch.Tensor) -> torch.Tensor:
        next_v = positions[2:, :, :] - positions[1:-1, :, :]
        cur_v = positions[1:-1, :, :] - positions[0:-2, :, :]
        a = next_v - cur_v
        a = (a - self.acc_mean.to(a.device)) / self.acc_std.to(a.device)
        return a

    def unnormalize_vel(self, vel: torch.Tensor) -> torch.Tensor:
        return vel * self.vel_std.to(vel.device) + self.vel_mean.to(vel.device)

    # ---------- Kappa interface ----------
    def getitem_timestep(self, idx, ctx=None):
        if self.split in ("valid", "test"):
            # use last input index by default
            return self.n_input_timesteps - 1
        # train: position within window
        assert ctx is not None and "time_idx" in ctx
        return ctx["time_idx"][-2]

    def getshape_timestep(self):
        # loose upper bound; trainers only need a 1-D shape
        return (1024,)

    def getshape_x(self):
        return self._shape_x

    def getshape_target(self):
        return self._shape_y

    def __getitem__(self, idx):
        ctx: Dict = {}
        if self.split == "train":
            file_idx, start = self._locate_train_index(int(idx))
            end = start + self.n_input_timesteps + self.n_pushforward_timesteps + 1
            T_list, F_list = self._read_window(file_idx, start, end)
            # pack ragged by aligning to last step count
            maxN = max((t.shape[0] for t in T_list), default=0)
            # if any is empty, bail to an empty sample
            if maxN == 0 or len(T_list) < (self.n_input_timesteps + 1):
                positions = torch.zeros(self.n_input_timesteps + 1, 0, 2)
                fields = torch.zeros(self.n_input_timesteps + 1, 0, self.num_channels)
            else:
                # pad each timestep to maxN (last dim) and ensure contiguous storage
                pos_pad = []
                fld_pad = []
                for t, f in zip(T_list, F_list):
                    n = t.shape[0]
                    if n < maxN:
                        t = torch.cat([t, t.new_zeros((maxN - n, t.shape[1]))], dim=0).contiguous()
                        f = torch.cat([f, f.new_zeros((maxN - n, f.shape[1]))], dim=0).contiguous()
                    else:
                        t = t.contiguous()
                        f = f.contiguous()
                    pos_pad.append(t)
                    fld_pad.append(f)
                positions = torch.stack(pos_pad, dim=0).contiguous()
                fields = torch.stack(fld_pad, dim=0).contiguous()
            ctx['time_idx'] = torch.arange(start, end)
            ctx['traj_idx'] = file_idx
        else:
            file_idx = int(idx)
            # Use _read_window for all formats
            T_total = self.file_lengths.get(file_idx, 0)
            T_list, F_list = self._read_window(file_idx, 0, T_total)
            # pad to maxN across all steps
            maxN = max((t.shape[0] for t in T_list), default=0)
            if maxN == 0:
                positions = torch.zeros(1, 0, 2)
            pos_pad = []
            fld_pad = []
            for t, f in zip(T_list, F_list):
                n = t.shape[0]
                if n < maxN:
                    t = torch.cat([t, t.new_zeros((maxN - n, t.shape[1]))], dim=0).contiguous()
                    f = torch.cat([f, f.new_zeros((maxN - n, f.shape[1]))], dim=0).contiguous()
                else:
                    t = t.contiguous()
                    f = f.contiguous()
                pos_pad.append(t)
                fld_pad.append(f)
            positions = torch.stack(pos_pad, dim=0).contiguous()
            fields = torch.stack(fld_pad, dim=0).contiguous()
            ctx['time_idx'] = torch.arange(0, positions.shape[0])
            ctx['traj_idx'] = file_idx

        return (positions.clone(), fields.clone()), ctx

    # ---------- getters required by trainer/collator ----------
    def _get_positions_fields(self, idx, ctx=None, downsample=True):
        # Ensure ctx exists
        if ctx is None:
            ctx = {}

        # If ctx is missing indices, derive them deterministically from idx
        if 'traj_idx' not in ctx or 'time_idx' not in ctx:
            if self.split == "train":
                file_idx, start = self._locate_train_index(int(idx))
                end = start + self.n_input_timesteps + self.n_pushforward_timesteps + 1
            else:
                file_idx = int(idx)
                # use full trajectory length from cached lengths
                T = self.file_lengths.get(file_idx, None)
                if T is None:
                    # fallback: read once
                    try:
                        with h5py.File(self.files[file_idx], "r") as f:
                            T = len(_list_timesteps(f))
                    except Exception:
                        T = 0
                start, end = 0, int(T)
            ctx['traj_idx'] = int(file_idx)
            ctx['time_idx'] = torch.arange(start, end)

        positions = None
        fields = None
        # If positions/fields missing, read window from HDF5
        if positions is None or fields is None:
            file_idx = int(ctx['traj_idx'])
            t_idx = ctx['time_idx']
            start = int(t_idx[0])
            end = int(t_idx[-1]) + 1 if t_idx.numel() > 0 else start
            T_list, F_list = self._read_window(file_idx, start, end)
            maxN = max((t.shape[0] for t in T_list), default=0)
            pos_pad, fld_pad = [], []
            for t, f in zip(T_list, F_list):
                n = t.shape[0]
                if n < maxN:
                    t = torch.cat([t, t.new_zeros((maxN - n, t.shape[1]))], dim=0).contiguous()
                    f = torch.cat([f, f.new_zeros((maxN - n, f.shape[1]))], dim=0).contiguous()
                else:
                    t = t.contiguous()
                    f = f.contiguous()
                pos_pad.append(t)
                fld_pad.append(f)
            positions = torch.stack(pos_pad, dim=0).contiguous() if len(pos_pad) > 0 else torch.zeros(1, 0, 2)
            fields = torch.stack(fld_pad, dim=0).contiguous() if len(fld_pad) > 0 else torch.zeros(1, 0, self.num_channels)

        # optional subsample/pad to fixed size for consistent batching
        if downsample and self.num_points_range is not None and positions.shape[1] > 0:
            lb, ub = self.num_points_range
            if lb == ub:
                target_n = lb
            else:
                target_n = int(torch.rand((), generator=None).item() * (ub - lb) + lb)
            
            current_n = positions.shape[1]
            if current_n > target_n:
                # Downsample
                perm = torch.randperm(current_n)[:target_n]
                ctx['perm'] = perm
                positions = positions[:, perm, :]
                fields = fields[:, perm, :]
            elif current_n < target_n:
                # Pad with zeros to target size
                T = positions.shape[0]
                pad_n = target_n - current_n
                pos_pad = torch.zeros(T, pad_n, positions.shape[2], dtype=positions.dtype, device=positions.device)
                fld_pad = torch.zeros(T, pad_n, fields.shape[2], dtype=fields.dtype, device=fields.device)
                positions = torch.cat([positions, pos_pad], dim=1).contiguous()
                fields = torch.cat([fields, fld_pad], dim=1).contiguous()
        return positions, fields, ctx

    def getitem_curr_pos(self, idx, ctx=None):
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        inp = positions[:self.n_input_timesteps, :, :]
        cur = inp[-1, :, :]
        return self.scale_pos(cur).contiguous()

    def getitem_curr_pos_full(self, idx, ctx=None):
        # For batching consistency, apply fixed sizing even for "full" data
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        inp = positions[:self.n_input_timesteps, :, :]
        cur = inp[-1, :, :]
        return self.scale_pos(cur).contiguous()

    def getitem_x(self, idx, ctx=None):
        positions, fields, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        # Use velocities as input features (T_in-1, N, 2), reshape to (T_in-1, C_in, N)
        v = self._vel(positions[:self.n_input_timesteps, :, :])
        v = v.permute(0, 2, 1)  # (t, dim, n)
        return v.contiguous()

    def getitem_edge_index(self, idx, ctx=None, downsample=True):
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        inp = positions[:self.n_input_timesteps, :, :]
        cur = inp[-1, :, :]
        if self.graph_mode == 'radius_graph':
            ei = radius_graph(x=cur, r=self.radius_graph_r, max_num_neighbors=self.radius_graph_max_num_neighbors, loop=True).T
        elif self.graph_mode == 'radius_graph_with_supernodes':
            n = cur.shape[0]
            gen = torch.Generator().manual_seed(int(idx)) if self.seed is not None else None
            perm_super = torch.randperm(n, generator=gen)[: self.n_supernodes or n]
            super_pos = cur[perm_super]
            ei = radius(x=cur, y=super_pos, r=self.radius_graph_r, max_num_neighbors=self.radius_graph_max_num_neighbors)
            ei[0] = perm_super[ei[0]]
            ei = ei.T
        else:
            # default radius_graph
            ei = radius_graph(x=cur, r=self.radius_graph_r, max_num_neighbors=self.radius_graph_max_num_neighbors, loop=True).T
        return ei.contiguous()

    def getitem_edge_index_target(self, idx, ctx=None, downsample=True):
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        tgt = positions[-1, :, :]
        ei = radius_graph(x=tgt, r=self.radius_graph_r, max_num_neighbors=self.radius_graph_max_num_neighbors, loop=True).T
        return ei.contiguous()

    def getitem_target_vel_large_t(self, idx, ctx=None):
        # Use downsample=True for batching consistency
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        v = self._vel(positions[-2:, :, :]).squeeze(0)  # (N,2)
        return v.contiguous()

    def getitem_target_acc(self, idx, ctx=None):
        # Use downsample=True for batching consistency
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        a = self._acc(positions[-3:, :, :]).squeeze(0)
        return a.contiguous()

    def getitem_target_pos(self, idx, ctx=None):
        # Use downsample=True for batching consistency
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        if self.split in ("valid", "test"):
            cur = positions[self.n_input_timesteps:, :, :]
        else:
            cur = positions[-1, :, :]
            cur = self.scale_pos(cur)
        return cur.contiguous()

    def getitem_target_pos_encode(self, idx, ctx=None):
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        if self.split in ("valid", "test"):
            cur = positions[self.n_input_timesteps:, :, :]
        else:
            cur = positions[-1, :, :]
            cur = self.scale_pos(cur)
        return cur.contiguous()

    def getitem_all_pos(self, idx, ctx=None):
        # Use downsample=True for batching consistency
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        return self.scale_pos(positions).contiguous()

    def getitem_all_vel(self, idx, ctx=None):
        # Use downsample=True for batching consistency
        positions, _, ctx = self._get_positions_fields(idx, ctx, downsample=True)
        return self._vel(positions).contiguous()

    def getitem_perm(self, idx, ctx=None):
        if ctx and 'perm' in ctx:
            # return permutation and an upper bound on particles
            return ctx['perm'], max(int(self.n_supernodes or 0), int(ctx['perm'].numel()))
        # identity mapping
        (positions, _), _ = self[idx]
        n = positions.shape[1]
        return torch.arange(n), n


# factory alias
amr_pointcloud_dataset = AmrPointCloudDataset
__all__ = ["AmrPointCloudDataset", "amr_pointcloud_dataset"]
