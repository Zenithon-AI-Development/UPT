#!/usr/bin/env python
"""
Create proper autoregressive rollout animation for ZPinch.
Shows: GT[0:4] as input, then GT vs Pred for timesteps 4-100
"""
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import einops
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from torch_geometric.nn import radius_graph
from utils.plotting import create_sequential_animation


def load_full_zpinch_trajectory(root, split):
    """Load complete ZPinch trajectory from HDF5."""
    root = Path(root)
    files = sorted((root / split).glob("*.hdf5"))
    
    if not files:
        raise FileNotFoundError(f"No files found in {root / split}")
    
    first_file = files[0]
    print(f"Loading full trajectory from: {first_file.name}")
    
    with h5py.File(first_file, 'r') as f:
        # ZPinch 7-channel format
        if 't0_fields' in f and 't1_fields' in f and 'forcing_fields' in f:
            density = torch.from_numpy(f['t0_fields/density'][:]).float()
            pressure = torch.from_numpy(f['t0_fields/pressure'][:]).float()
            velocity = torch.from_numpy(f['t1_fields/velocity'][:]).float()
            b_field = torch.from_numpy(f['t1_fields/b_field'][:]).float()
            current_drive = torch.from_numpy(f['forcing_fields/current_drive'][:]).float()
            
            # Stack channels
            density = density.unsqueeze(-1)
            pressure = pressure.unsqueeze(-1)
            current_drive = current_drive.unsqueeze(-1)
            
            full_traj = torch.cat([density, pressure, velocity, b_field, current_drive], dim=-1)
        else:
            raise ValueError(f"Unknown format in {first_file}")
    
    T, H, W, C = full_traj.shape
    print(f"  Shape: (T={T}, H={H}, W={W}, C={C})")
    
    return full_traj, H, W, C


def create_rollout(stage_id: str, output_path: str = 'zpinch_rollout.gif'):
    """Create autoregressive rollout animation."""
    
    print(f"Loading from stage: {stage_id}")
    
    # Load config
    save_dir = Path("/home/workspace/projects/transformer/UPT/benchmarking/save")
    config_path = save_dir / "stage1" / stage_id / "hp_resolved.yaml"
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    # Load full trajectory
    train_config = hp["datasets"]["train"]
    root = train_config["root"]
    split = train_config["split"]
    
    full_traj, H, W, C = load_full_zpinch_trajectory(root, split)
    T = full_traj.shape[0]
    
    # Apply normalization
    if "mean" in train_config and "std" in train_config:
        mean = torch.tensor(train_config["mean"]).view(1, 1, 1, -1)
        std = torch.tensor(train_config["std"]).view(1, 1, 1, -1)
        full_traj = (full_traj - mean) / std
        print(f"  Applied normalization")
    
    # Flatten spatial: (T, H, W, C) -> (T, N, C)
    full_traj = einops.rearrange(full_traj, "t h w c -> t (h w) c")
    N = full_traj.shape[1]
    print(f"  Flattened to: (T={T}, N={N}, C={C})")
    
    # Create model (simplified - just need input/output shapes)
    from models import model_from_kwargs
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    num_input_timesteps = train_config["num_input_timesteps"]
    input_shape = (N, num_input_timesteps * C)
    output_shape = (N, C)
    
    print(f"\nCreating model...")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    
    class FakeDataset:
        def __init__(self, input_shape, output_shape):
            self._shape_x = input_shape
            self._shape_y = output_shape
        def getshape_x(self):
            return self._shape_x
        def getshape_target(self):
            return self._shape_y
        def getdim_timestep(self):
            return T  # Total timesteps
    
    class MiniDataContainer:
        def __init__(self, input_shape, output_shape):
            self.fake_dataset = FakeDataset(input_shape, output_shape)
        def get_dataset(self, key=None):
            return self.fake_dataset
    
    model = model_from_kwargs(
        **hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=None,
        data_container=MiniDataContainer(input_shape, output_shape),
    )
    
    # Load checkpoints
    ckpt_dir = save_dir / "stage1" / stage_id / "checkpoints"
    checkpoint = "latest"
    
    for name, stem in {
        "conditioner": "cfd_simformer_model.conditioner",
        "encoder": "cfd_simformer_model.encoder",
        "latent": "cfd_simformer_model.latent",
        "decoder": "cfd_simformer_model.decoder",
    }.items():
        candidates = list(ckpt_dir.glob(f"{stem}*cp={checkpoint}*model.th"))
        if not candidates:
            candidates = list(ckpt_dir.glob(f"{stem}*model.th"))
        if candidates:
            ckpt_file = sorted(candidates)[-1]
            sd = torch.load(ckpt_file, map_location=device, weights_only=False)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            getattr(model, name).load_state_dict(sd, strict=False)
            print(f"  Loaded {name}: {ckpt_file.name}")
    
    model = model.to(device)
    model.eval()
    
    # Setup geometry and positions
    grid_scaling = train_config.get("grid_scaling", [200.0, 300.0])
    xs = torch.linspace(0, grid_scaling[0], W)
    ys = torch.linspace(0, grid_scaling[1], H)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    mesh_pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1).to(device)  # (N, 2)
    query_pos = mesh_pos.clone()
    geometry2d = None  # ZPinch doesn't use geometry2d
    
    # Get graph params
    radius_r = hp['trainer']['radius_graph_r']
    radius_max_nn = hp['trainer']['radius_graph_max_num_neighbors']
    num_supernodes = train_config['collators'][0].get('num_supernodes', None)
    
    print(f"\n🔄 Starting rollout with GT inputs (teacher forcing, like GAOT)...")
    print(f"  Total timesteps: {T}")
    print(f"  Predictions: timesteps {num_input_timesteps}-{T-1} ({T-num_input_timesteps} total)")
    print(f"  Each prediction uses GT inputs (NOT previous predictions)")
    
    pred_sequence = []
    
    with torch.no_grad():
        for t in range(num_input_timesteps, T):
            # Build window from GROUND TRUTH only (teacher forcing, like GAOT)
            # t=4: use GT[0:4] → predict GT[4]
            # t=5: use GT[1:5] → predict GT[5]  (uses GT[4], NOT Pred[4]!)
            # t=6: use GT[2:6] → predict GT[6]
            start_idx = t - num_input_timesteps
            
            # Get GT window: (4, N, C)
            window = full_traj[start_idx:t]  # (4, N, C)
            
            # Stack and flatten: (4, N, C) -> (N, 4, C) -> (N, 4*C)
            x_window = einops.rearrange(window, "t n c -> n t c")  # (N, 4, C)
            x_flat = x_window.reshape(N, -1).to(device)  # (N, 4*C)
            
            # Create batch indices
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            unbatch_idx = torch.zeros(N, dtype=torch.long, device=device)
            unbatch_select = torch.zeros(1, dtype=torch.long, device=device)
            
            # Create radius graph
            flow = "target_to_source" if num_supernodes is not None else "source_to_target"
            edge_index = radius_graph(
                x=mesh_pos, r=radius_r, batch=batch_idx, loop=True,
                max_num_neighbors=radius_max_nn, flow=flow
            )
            mesh_edges = edge_index.T
            
            # Forward pass
            timestep = torch.tensor([t], dtype=torch.long, device=device)
            velocity = torch.tensor([0.0], dtype=torch.float32, device=device)  # ZPinch uses 0.0
            query_pos_batch = query_pos.unsqueeze(0)  # (1, N, 2)
            
            outputs = model(
                x=x_flat,
                mesh_pos=mesh_pos,
                query_pos=query_pos_batch,
                mesh_edges=mesh_edges,
                geometry2d=geometry2d,
                timestep=timestep,
                velocity=velocity,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select
            )
            
            # Get prediction (still normalized)
            pred = outputs['x_hat'].cpu()  # (N, C)
            pred_sequence.append(pred)
            
            if (t - num_input_timesteps + 1) % 10 == 0:
                print(f"  Predicted timestep {t}/{T-1}")
    
    print(f"\n✓ Rollout complete: {len(pred_sequence)} predictions")
    
    # DENORMALIZE everything back to physical units
    if "mean" in train_config and "std" in train_config:
        mean_np = train_config["mean"]
        std_np = train_config["std"]
        mean_t = torch.tensor(mean_np).view(1, 1, -1)  # (1, 1, C)
        std_t = torch.tensor(std_np).view(1, 1, -1)
        
        # Denormalize GT
        full_traj_denorm = full_traj * std_t + mean_t  # (T, N, C)
        gt_sequence = full_traj_denorm.cpu().numpy()
        
        # Denormalize predictions
        pred_sequence_denorm = []
        for pred in pred_sequence:
            pred_denorm = pred * std_t.squeeze(0) + mean_t.squeeze(0)  # (N, C)
            pred_sequence_denorm.append(pred_denorm.numpy())
        pred_sequence = pred_sequence_denorm
        print(f"  Denormalized all predictions and GT")
    else:
        gt_sequence = full_traj.cpu().numpy()  # (T, N, C) - all ground truth
    
    # Pred sequence: first 4 are GT, rest are predictions
    pred_seq_list = []
    for t in range(T):
        if t < num_input_timesteps:
            pred_seq_list.append(gt_sequence[t])  # Use GT for initial frames
        else:
            pred_idx = t - num_input_timesteps
            pred_seq_list.append(pred_sequence[pred_idx])  # Use prediction
    
    pred_seq_array = np.array(pred_seq_list)  # (T, N, C)
    
    print(f"\nSequence shapes:")
    print(f"  GT: {gt_sequence.shape}")
    print(f"  Pred: {pred_seq_array.shape}")
    
    # Get coordinates
    coords = mesh_pos.cpu().numpy()
    
    # Channel names
    names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']
    
    # Create animation
    print(f"\nCreating animation with {T} frames...")
    print(f"  Frames 0-{num_input_timesteps-1}: Ground Truth (shown in Input column)")
    print(f"  Frames {num_input_timesteps}-{T-1}: Predictions vs Ground Truth")
    
    # Use last GT input frame (timestep 3) as "input" reference for the Input column
    # input_data should be (n_points, n_channels), not (n_input_timesteps, n_points, n_channels)
    input_data = gt_sequence[num_input_timesteps - 1]  # Last input frame: (N, C)
    
    # Limit to first 101 frames for reasonable animation size
    max_frames = min(101, T)
    gt_seq_anim = gt_sequence[:max_frames]
    pred_seq_anim = pred_seq_array[:max_frames]
    
    print(f"  Animation will show frames 0-{max_frames-1}")
    
    create_sequential_animation(
        gt_sequence=gt_seq_anim,
        pred_sequence=pred_seq_anim,
        coords=coords,
        save_path=output_path,
        input_data=input_data,
        interval=100,  # 100ms = 10 fps
        symmetric=True,
        names=names,
        colorbar_type="light",
        show_error=True,
        dynamic_colorscale=True,  # ADAPTIVE SCALES LIKE GAOT!
    )
    
    print(f"\n✓ Animation saved: {output_path}")
    print(f"  Total frames: {T}")
    print(f"  GT initial: frames 0-{num_input_timesteps-1}")
    print(f"  Rollout: frames {num_input_timesteps}-{T-1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_id", required=True, help="Stage ID")
    parser.add_argument("--output", default="zpinch_rollout.gif", help="Output path")
    
    args = parser.parse_args()
    create_rollout(args.stage_id, args.output)

