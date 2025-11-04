# src/datasets/well_trl2d_dataset.py
from pathlib import Path
import torch
import einops
import numpy as np

from the_well.data import WellDataset
from .base.dataset_base import DatasetBase


class WellTrl2dDataset(DatasetBase):
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
        well_dataset_name="turbulent_radiative_layer_2D",
        split="train",                      # "train" | "valid" | "test"
        # SimFormer timing
        num_input_timesteps=4,              # k or .inf for rollout
        num_output_timesteps=1,             # keep 1 for training (next step)
        max_num_timesteps=101,              # needed for rollout (.inf): total snapshots per trajectory
        # normalization / clamping
        norm="mean0std1",                   # map to use_normalization
        normalization_type=None,            # Allow override from YAML
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
        # normalization_type is handled as explicit parameter, not passed to super
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

        use_normalization = (norm != "none")
        
        # Handle normalization_type: convert from dict if needed
        if use_normalization:
            if normalization_type is None:
                # Default to Z-score
                from the_well.data.normalization import ZScoreNormalization
                normalization_type = ZScoreNormalization
            elif isinstance(normalization_type, dict):
                # Convert from YAML config dict to actual class
                norm_kind = normalization_type.get("kind", "ZScoreNormalization")
                if norm_kind == "RMSNormalization":
                    from the_well.data.normalization import RMSNormalization
                    normalization_type = RMSNormalization
                elif norm_kind == "ZScoreNormalization":
                    from the_well.data.normalization import ZScoreNormalization
                    normalization_type = ZScoreNormalization
                else:
                    raise ValueError(f"Unknown normalization type: {norm_kind}")

        # build WELL
        self.well = WellDataset(
            well_base_path=Path(well_base_path),
            well_dataset_name=well_dataset_name,
            well_split_name=split,  # "train" | "valid" | "test" (WELL naming)
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
        )
        
        self.clamp = None if clamp in (None, 0, 0.0) else float(clamp)
        self.clamp_mode = clamp_mode
        # optional truncation of sequences in a split
        self._indices = list(range(len(self.well)))
        # self.max_num_sequences = int(len(self.well)/4)
        if self.max_num_sequences is not None:
            self._indices = self._indices[: self.max_num_sequences]

        # metadata
        sample0 = self.well[self._indices[0]]
        # tensors are (T, H, W, F)
        T_in, H, W, F = sample0["input_fields"].shape
        self.H, self.W, self.F = int(H), int(W), int(F)

        # positions (scaled to [0,200]x[0,300])
        max_x, max_y = grid_scaling
        xs = torch.linspace(0.0, max_x, self.W)   # width
        ys = torch.linspace(0.0, max_y, self.H)   # height
        # xs = torch.linspace(0.0, int(max_x/2), int(self.W/2))   # width # downsample by 2
        # ys = torch.linspace(0.0, int(max_y/2), int(self.H/2))   # height # downsample by 2
        X, Y = torch.meshgrid(xs, ys, indexing="xy")  # (W, H)
        self._mesh_pos_xy = torch.stack([X.T.reshape(-1), Y.T.reshape(-1)], dim=1).contiguous()  # (N,2)

        # geometry2d dummy (no obstacles)
        self._geometry2d = torch.zeros(2, self.H, self.W, dtype=torch.float32)

        # timestep/velocity dummies
        self._timestep_dummy = torch.tensor(0, dtype=torch.long)
        self._velocity_dummy = torch.tensor(0.0, dtype=torch.float32)

        self._num_total_snapshots = 101
        self._num_transitions = self._num_total_snapshots - 1

        # declared shapes for trainer
        if self._rollout:
            self._shape_x = (None, self.F)                     # x0
            self._shape_y = (None, self.F, n_steps_output)     # stacked over time
        else:
            self._shape_x = (None, n_steps_input * self.F)
            self._shape_y = (None, self.F if n_steps_output == 1 else n_steps_output * self.F)

    # ---- plumbing ----
    def __len__(self):
        return len(self._indices)

    def _item(self, idx):
        actual_idx = self._indices[int(idx)]
        if idx <= 2:
            print(f"[DATASET] _item called with idx={idx}, using well[{actual_idx}]")
        return self.well[actual_idx]

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
        x = it["input_fields"]  # (T_in, H, W, F)
        
        # DEBUG - check per-channel statistics and verify same sample
        if idx <= 2:
            import hashlib
            data_hash = hashlib.md5(x.cpu().numpy().tobytes()).hexdigest()[:8]
            print(f"\n[TRL2D DEBUG] getitem_x idx={idx}, hash={data_hash}")
            print(f"  input_fields shape: {x.shape}")
            for ch in range(x.shape[-1]):
                ch_data = x[..., ch]
                print(f"  Ch{ch}: mean={ch_data.mean():.6f}, std={ch_data.std():.6f}, range=[{ch_data.min():.3f}, {ch_data.max():.3f}]")
        
        # x = x[:, ::2, ::2, :]  # downsample by 2
        if self._rollout:
            x0 = x[0]  # (H, W, F)
            x0 = einops.rearrange(x0, "h w c -> (h w) c")  # (N, F)
            return self._clamp_if_needed(x0.float())
        # training: flatten time
        x = einops.rearrange(x, "t h w c -> (h w) (t c)")  # (N, T_in*F)
        
        if idx == 0:
            print(f"  After reshape x: {x.shape}, range: [{x.min():.6f}, {x.max():.6f}]")
        
        return self._clamp_if_needed(x.float())

    def getitem_target(self, idx, ctx=None):
        it = self._item(idx)
        y = it["output_fields"]  # (T_out, H, W, F)
        
        # DEBUG - check per-channel statistics and verify same sample
        if idx <= 2:
            import hashlib
            data_hash = hashlib.md5(y.cpu().numpy().tobytes()).hexdigest()[:8]
            print(f"[TRL2D DEBUG] getitem_target idx={idx}, hash={data_hash}")
            print(f"  output_fields shape: {y.shape}")
            for ch in range(y.shape[-1]):
                ch_data = y[..., ch]
                print(f"  Ch{ch}: mean={ch_data.mean():.6f}, std={ch_data.std():.6f}, range=[{ch_data.min():.3f}, {ch_data.max():.3f}]")
        
        # y = y[:, ::2, ::2, :]  # downsample by 2
        if self._rollout:
            # return (N, F, T_out) for offline rollout evaluation
            y = einops.rearrange(y, "t h w c -> (h w) c t")
            return self._clamp_if_needed(y.float())
        # training (T_out=1): next frame (N, F)
        if y.shape[0] == 1:
            y = y[0]
            y = einops.rearrange(y, "h w c -> (h w) c")
        else:
            y = einops.rearrange(y, "t h w c -> (h w) (t c)")
        
        if idx == 0:
            print(f"  After reshape target: {y.shape}, range: [{y.min():.6f}, {y.max():.6f}]")
        
        return self._clamp_if_needed(y.float())

    # convenient alias (sometimes used by callbacks)
    def getitem_target_t0(self, idx, ctx=None):
        it = self._item(idx)
        x = it["input_fields"][0]                 # (H, W, F)
        # x = x[::2, ::2, :]  # downsample by 2
        x = einops.rearrange(x, "h w c -> (h w) c")
        return self._clamp_if_needed(x.float())

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
well_trl2d_dataset = WellTrl2dDataset
__all__ = ["WellTrl2dDataset", "well_trl2d_dataset"]
