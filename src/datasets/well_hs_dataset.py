# src/datasets/well_trl2d_dataset.py
from pathlib import Path
import hashlib
import torch
import einops
import numpy as np

from the_well.data import WellDataset
from .base.dataset_base import DatasetBase


STATS_DIR = Path(__file__).resolve().parent / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


class WellHsDataset(DatasetBase):
    """
    Adapter around The WELL 'turbulent_radiative_layer_2D' for the SimFormer pipeline.

    Train mode (num_input_timesteps = k):
        x:       (N_pts, k*F)
        target:  (N_pts, F)

    Rollout mode (num_input_timesteps = .inf, with max_num_timesteps = T_total):
        x:             (N_pts, F)           # t0 only
        target:        (N_pts, F, T_out)    # stacked t=1..T_total-1 along last dim
        target_t0:     (N_pts, F)           # convenience (t=0 ground truth), if needed
    """
    def __init__(
        self,
        # WELL dataset location & naming
        well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
        well_dataset_name="helmholtz_staircase",
        split="train",                      # "train" | "valid" | "test"
        # SimFormer timing
        num_input_timesteps=4,              # k or .inf for rollout
        num_output_timesteps=1,             # keep 1 for training (next step)
        max_num_timesteps=50,              # needed for rollout (.inf): total snapshots per trajectory
        # normalization / clamping
        norm="mean0std1_auto",
        clamp=None,
        clamp_mode="log",
        # grid scaling to match CFD convention ([0,200]x[0,300])
        grid_scaling=(200.0, 300.0),
        # optional: limit number of sequences for a split
        max_num_sequences=None,
        # NOTE: any other SimFormer kwargs may appear in YAML; we swallow them below
        **kwargs,
    ):
        # --- swallow SimFormer/CFD-only keys so DatasetBase doesn't choke
        for k in [
            "num_input_points", "num_input_points_ratio", "num_input_points_mode",
            "num_query_points", "num_query_points_mode", "couple_query_with_input",
            "radius_graph_r", "radius_graph_max_num_neighbors", "num_supernodes",
            "grid_resolution", "standardize_query_pos",
            "version",  # CFD-only
        ]:
            kwargs.pop(k, None)
        kwargs.pop("root", None)
        super().__init__(**kwargs)

        # config
        self.split = split
        self.max_num_sequences = max_num_sequences
        self.max_num_timesteps = int(max_num_timesteps) if max_num_timesteps is not None else None
        self._rollout = (num_input_timesteps == float("inf"))

        # WELL expects explicit input/output window lengths
        if self._rollout:
            assert self.max_num_timesteps is not None, "Set max_num_timesteps when num_input_timesteps=.inf (rollout)."
            n_steps_input = 1                          # feed t0 only
            n_steps_output = self.max_num_timesteps - 1  # predict t=1..T-1
        else:
            n_steps_input = int(num_input_timesteps)
            n_steps_output = int(num_output_timesteps)

        self._stats_mode = norm
        base_path = Path(well_base_path)
        self._well_dataset_name = well_dataset_name
        self._stats_id = self._build_stats_id(base_path, well_dataset_name)

        # build WELL without internal normalization
        self.well = WellDataset(
            well_base_path=base_path,
            well_dataset_name=well_dataset_name,
            well_split_name=split,  # "train" | "valid" | "test" (WELL naming)
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=False,
        )
        
        self.clamp = None if clamp in (None, 0, 0.0) else float(clamp)
        self.clamp_mode = clamp_mode
        # optional truncation of sequences in a split
        self._indices = list(range(len(self.well)))
        if self.max_num_sequences is not None:
            self._indices = self._indices[: self.max_num_sequences]

        # metadata
        sample0 = self.well[self._indices[0]]
        # tensors are (T, H, W, F)
        T_in, H, W, F = sample0["input_fields"].shape
        self.H, self.W, self.F = int(H), int(W), int(F)
        constant_fields = sample0.get("constant_fields")
        if constant_fields is not None and constant_fields.numel() > 0:
            self._constant_field_channels = int(constant_fields.shape[-1])
        else:
            self._constant_field_channels = 0

        # positions (scaled to [0,200]x[0,300])
        max_x, max_y = grid_scaling
        xs = torch.linspace(0.0, max_x, self.W)   # width
        ys = torch.linspace(0.0, max_y, self.H)   # height
        X, Y = torch.meshgrid(xs, ys, indexing="xy")  # (W, H)
        self._mesh_pos_xy = torch.stack([X.T.reshape(-1), Y.T.reshape(-1)], dim=1).contiguous()  # (N,2)

        # geometry2d dummy (no obstacles)
        self._geometry2d = torch.zeros(2, self.H, self.W, dtype=torch.float32)

        # timestep/velocity dummies
        self._timestep_dummy = torch.tensor(0, dtype=torch.long)
        self._velocity_dummy = torch.tensor(0.0, dtype=torch.float32)

        self._num_total_snapshots = 50
        self._num_transitions = self._num_total_snapshots - 1

        # declared shapes for trainer
        if self._rollout:
            self._shape_x = (None, self.F)                     # x0
            self._shape_y = (None, self.F, n_steps_output)     # stacked over time
        else:
            self._shape_x = (None, n_steps_input * self.F)
            self._shape_y = (None, self.F if n_steps_output == 1 else n_steps_output * self.F)

        self.eps = 1e-6
        stats_path = STATS_DIR / f"{self._stats_id}_train.pt"
        stats_mode = norm
        if norm == "mean0std1_auto":
            if self.split == "train":
                if stats_path.exists():
                    data = torch.load(stats_path)
                    self.mean = data["mean"].to(torch.float32)
                    self.std = data["std"].to(torch.float32)
                else:
                    self.mean, self.std = self._compute_dataset_stats()
                    self.std = self.std.clamp_min(self.eps)
                    torch.save({"mean": self.mean, "std": self.std}, stats_path)
            else:
                if not stats_path.exists():
                    raise FileNotFoundError(
                        f"Normalization stats not found at {stats_path}. Instantiate the training split first."
                    )
                data = torch.load(stats_path)
                self.mean = data["mean"].to(torch.float32)
                self.std = data["std"].to(torch.float32)
            stats_mode = "mean0std1"

        if stats_mode == "mean0std1":
            if getattr(self, "mean", None) is None or getattr(self, "std", None) is None:
                raise ValueError(
                    "mean/std statistics not provided. Use 'mean0std1_auto' so the dataset can compute them."
                )
            self.std = self.std.clamp_min(self.eps)
        else:
            self.mean = torch.zeros(self.F, dtype=torch.float32)
            self.std = torch.ones(self.F, dtype=torch.float32)

        self._stats_mode = stats_mode

    @staticmethod
    def _build_stats_id(base_path: Path, dataset_name: str) -> str:
        root = (base_path / dataset_name).resolve()
        root_id = hashlib.md5(str(root).encode()).hexdigest()[:12]
        return f"well_hs_{root_id}"

    def _compute_dataset_stats(self):
        mean = torch.zeros(self.F, dtype=torch.float64)
        M2 = torch.zeros(self.F, dtype=torch.float64)
        count = 0
        for idx in self._indices:
            sample = self._item(idx)
            tensors = [torch.as_tensor(sample["input_fields"], dtype=torch.float64)]
            out = torch.as_tensor(sample["output_fields"], dtype=torch.float64)
            if out.numel() > 0:
                tensors.append(out)
            data = torch.cat(tensors, dim=0)
            flat = data.reshape(-1, self.F)
            n = flat.shape[0]
            if n == 0:
                continue
            batch_mean = flat.mean(dim=0)
            batch_M2 = ((flat - batch_mean).pow(2)).sum(dim=0)
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
        std = var.sqrt().clamp_min(self.eps)
        return mean.to(torch.float32), std.to(torch.float32)

    def _apply_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._stats_mode != "mean0std1":
            return tensor
        shape = [1] * tensor.ndim
        shape[-1] = self.F
        return tensor.sub(self.mean.view(*shape)).div(self.std.view(*shape))

    # ---- plumbing ----
    def __len__(self):
        return len(self._indices)

    def _item(self, idx):
        return self.well[self._indices[int(idx)]]

    def getshape_timestep(self):
        # KappaData expects a 1-D shape tuple here
        return (self._num_transitions,)

    # def getitem_timestep(self, idx, ctx=None):
    #     # Match CfdDataset behavior: index in [0, _num_transitions-1]
    #     if self._n_in == float("inf"):
    #         # rollout datasets don't use this key; any int is fine
    #         return torch.tensor(0, dtype=torch.long)
    #     return torch.tensor(idx % self._num_transitions, dtype=torch.long)

    def getitem_timestep(self, idx, ctx=None):
        # Kappa expects an int tensor; for rollout we don’t really use it
        if self._rollout:
            return torch.tensor(0, dtype=torch.long)
        return torch.tensor(idx % self._num_transitions, dtype=torch.long)

    # ---- mandatory interface for SimFormer collator/trainer ----
    def getshape_x(self):       return self._shape_x
    def getshape_target(self):  return self._shape_y

    # positions
    def getitem_all_pos(self, idx, ctx=None):   return self._mesh_pos_xy
    def getitem_mesh_pos(self, idx, ctx=None):  return self._mesh_pos_xy
    def getitem_query_pos(self, idx, ctx=None): return self._mesh_pos_xy

    # edges -> None (created on GPU by trainer)
    def getitem_mesh_edges(self, idx, ctx=None): return None

    # geometry2d (2,H,W)
    def getitem_geometry2d(self, idx, ctx=None): return self._geometry2d
    def getshape_geometry2d(self):               return (2, self.H, self.W)

    # conditioner inputs (we give stable dummies; you can wire in real values later)
    # def getitem_timestep(self, idx, ctx=None):   return self._timestep_dummy
    # def getshape_timestep(self):                 return ()
    def getitem_velocity(self, idx, ctx=None):   return self._velocity_dummy
    def getshape_velocity(self):                 return ()

    # data
    def getitem_x(self, idx, ctx=None):
        it = self._item(idx)
        x = torch.as_tensor(it["input_fields"], dtype=torch.float32)
        x = self._apply_norm(x)
        if self._rollout:
            x0 = x[0]
            x0 = einops.rearrange(x0, "h w c -> (h w) c")
            return self._clamp_if_needed(x0)
        x = einops.rearrange(x, "t h w c -> (h w) (t c)")
        return self._clamp_if_needed(x)

    def getitem_target(self, idx, ctx=None):
        it = self._item(idx)
        y = torch.as_tensor(it["output_fields"], dtype=torch.float32)
        y = self._apply_norm(y)
        if self._rollout:
            y = einops.rearrange(y, "t h w c -> (h w) c t")
            return self._clamp_if_needed(y)
        if y.shape[0] == 1:
            y = y[0]
            y = einops.rearrange(y, "h w c -> (h w) c")
        else:
            y = einops.rearrange(y, "t h w c -> (h w) (t c)")
        return self._clamp_if_needed(y)

    # convenient alias (sometimes used by callbacks)
    def getitem_target_t0(self, idx, ctx=None):
        it = self._item(idx)
        x = torch.as_tensor(it["input_fields"][0], dtype=torch.float32)
        x = self._apply_norm(x)
        x = einops.rearrange(x, "h w c -> (h w) c")
        return self._clamp_if_needed(x)

    # constant fields (e.g. staircase mask)
    def getitem_constant_fields(self, idx, ctx=None):
        const = self._item(idx).get("constant_fields")
        if const is None or const.numel() == 0:
            return torch.zeros(self.H, self.W, 0, dtype=torch.float32)
        return torch.as_tensor(const, dtype=torch.float32)

    def getshape_constant_fields(self):
        if self._constant_field_channels == 0:
            return (self.H, self.W, 0)
        return (self.H, self.W, self._constant_field_channels)

    # (optional) reconstruction aliases
    def getitem_reconstruction_input(self, idx, ctx=None):  return self.getitem_target(idx, ctx)
    def getitem_reconstruction_output(self, idx, ctx=None): return self.getitem_target(idx, ctx)

    # clamping identical to CFD utility
    def _clamp_if_needed(self, data: torch.Tensor):
        if self.clamp is None:
            return data
        if self.clamp_mode == "hard":
            return data.clamp(-self.clamp, self.clamp)
        if self.clamp_mode == "log":
            mask = data.abs() > self.clamp
            if mask.any():
                vals = data[mask]
                out = data.clone()
                out[mask] = torch.sign(vals) * (
                    self.clamp + torch.log1p(vals.abs()) - np.log1p(self.clamp)
                )
                return out
            return data
        raise NotImplementedError


# UPT factory alias (instantiate by 'kind: well_trl2d_dataset')
well_hs_dataset = WellHsDataset
__all__ = ["WellHsDataset", "well_hs_dataset"]
