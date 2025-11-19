#!/usr/bin/env python
"""
Create proper autoregressive rollout animation for ZPinch models.
Shows all timesteps with first 4 as GT, rest as predictions.
Like GAOT: GT[0], GT[1], GT[2], GT[3], Pred(GT[0:4]), Pred(GT[1:5]), ...
"""
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import einops

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.factory import instantiate
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from providers.path_provider import PathProvider
from providers.dataset_config_provider import DatasetConfigProvider
from torch_geometric.nn import radius_graph
from utils.plotting import create_sequential_animation
from configs.static_config import StaticConfig


def create_rollout(
    stage_id: str,
    checkpoint: str = "latest",
    output_path: str = 'zpinch_rollout.gif',
):
    """
    Create proper autoregressive rollout animation.
    
    Loads full trajectory (101 timesteps) and does autoregressive rollout:
    - Frames 0-3: Ground truth (initial condition)
    - Frame 4: Predicted from GT[0:4]
    - Frame 5: Predicted from [GT[1:4], Pred[4]]
    - Frame 6: Predicted from [GT[2:4], Pred[4:6]]
    - etc.
    """
    print(f"Loading from stage: {stage_id}")
    
    # Load config
    output_path_obj = Path("/home/workspace/projects/transformer/UPT/benchmarking/save")
    config_path = output_path_obj / "stage1" / stage_id / "hp_resolved.yaml"
    print(f"Config: {config_path}")
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    # Setup providers
    static_config = StaticConfig(Path(__file__).parent / "src" / "static_config.yaml")
    model_path = static_config.model_path or static_config.output_path
    path_provider = PathProvider(
        model_path=model_path,
        output_path=static_config.output_path,
        stage_name=hp['stage_name'],
        stage_id=stage_id,
        temp_path=static_config.temp_path,
    )
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static_config.get_global_dataset_paths(),
        local_dataset_path=static_config.get_local_dataset_path(),
        data_source_modes=static_config.get_data_source_modes(),
    )
    
    # Load FULL trajectory dataset (use train split, single sequence)
    train_kwargs = hp["datasets"]["train"].copy()
    train_kwargs["max_num_sequences"] = 1  # Just one trajectory
    # Keep num_input_timesteps=4 for model, but we'll load full trajectory manually
    
    dataset = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **train_kwargs,
    )
    
    print(f"Dataset: {len(dataset)} samples (windows)")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    # Load the FULL trajectory directly from the underlying dataset
    # Get first file
    if hasattr(dataset, 'well'):
        # WellZpinchDataset / WellTrl2dDataset
        well_dataset = dataset.well
        full_sample = well_dataset[0]  # Get first trajectory
        
        # input_fields: (T_in, H, W, C) or target_fields: (T_out, H, W, C)
        # For training data, input_fields has first k timesteps
        # We need all timesteps - load from file directly
        import h5py
        
        # Get root and files
        root = Path(train_kwargs['root'])
        split = train_kwargs['split']
        file_pattern = "*.hdf5"
        files = sorted((root / split).glob(file_pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found in {root / split}")
        
        first_file = files[0]
        print(f"Loading full trajectory from: {first_file}")
        
        # Load full trajectory
        with h5py.File(first_file, 'r') as f:
            # ZPinch format: t0_fields, t1_fields, forcing_fields
            if 't0_fields' in f and 't1_fields' in f and 'forcing_fields' in f:
                density = torch.from_numpy(f['t0_fields/density'][:]).float()  # (T,H,W)
                pressure = torch.from_numpy(f['t0_fields/pressure'][:]).float()  # (T,H,W)
                velocity = torch.from_numpy(f['t1_fields/velocity'][:]).float()  # (T,H,W,2)
                b_field = torch.from_numpy(f['t1_fields/b_field'][:]).float()  # (T,H,W,2)
                current_drive = torch.from_numpy(f['forcing_fields/current_drive'][:]).float()  # (T,H,W)
                
                # Stack: (T,H,W) -> (T,H,W,1)
                density = density.unsqueeze(-1)
                pressure = pressure.unsqueeze(-1)
                current_drive = current_drive.unsqueeze(-1)
                
                # Concatenate all 7 channels
                full_traj = torch.cat([density, pressure, velocity, b_field, current_drive], dim=-1)  # (T,H,W,7)
            else:
                raise ValueError(f"Unknown format in {first_file}")
        
        T, H, W, C = full_traj.shape
        print(f"  Full trajectory shape: (T={T}, H={H}, W={W}, C={C})")
        
        # Apply normalization if needed
        if hasattr(dataset, '_norm_inplace'):
            dataset._norm_inplace(full_traj)
        elif hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
            mean = dataset.mean.view(1, 1, 1, -1)
            std = dataset.std.view(1, 1, 1, -1)
            full_traj = (full_traj - mean) / std
        
        # Flatten spatial dimensions: (T, H, W, C) -> (T, N, C)
        full_traj = einops.rearrange(full_traj, "t h w c -> t (h w) c")
        N = full_traj.shape[1]
        
        print(f"  Flattened: (T={T}, N={N}, C={C})")
        
    elif hasattr(dataset, "_load_traj"):
        full_traj = dataset._load_traj(0)  # (T, H, W, C)
        T, H, W, C = full_traj.shape
        print(f"  Full trajectory shape (via dataset): (T={T}, H={H}, W={W}, C={C})")
        if hasattr(dataset, "_norm_inplace"):
            dataset._norm_inplace(full_traj)
        elif hasattr(dataset, "mean") and hasattr(dataset, "std"):
            mean = dataset.mean.view(1, 1, 1, -1)
            std = dataset.std.view(1, 1, 1, -1)
            full_traj = (full_traj - mean) / std
        full_traj = einops.rearrange(full_traj, "t h w c -> t (h w) c")
        N = full_traj.shape[1]
        print(f"  Flattened: (T={T}, N={N}, C={C})")

    else:
        raise NotImplementedError("Dataset doesn't have 'well' attribute")
    
    # Load model
    checkpoint_dir = output_path_obj / "stage1" / stage_id / "checkpoints"
    print(f"\nLoading model from: {checkpoint_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_kwargs = hp["model"].copy()
    model_kind = model_kwargs.pop("kind")
    
    # Create mini data container
    class MiniDataContainer:
        def __init__(self, dataset):
            self.dataset = dataset
        def get_dataset(self, key=None, mode=None, **kwargs):
            if mode is not None:
                return self.dataset, None
            return self.dataset
    
    data_container = MiniDataContainer(dataset)
    model = model_from_kwargs(
        kind=model_kind,
        data_container=data_container,
        input_shape=input_shape,
        output_shape=output_shape,
        **model_kwargs,
    )
    model = model.to(device)
    model.eval()
    
    # Load checkpoints
    for component_name in ['conditioner', 'encoder', 'latent', 'decoder']:
        full_name = f"cfd_simformer_model.{component_name}"
        ckpt_pattern = f"{full_name} cp=*model.th"
        ckpt_files = list(checkpoint_dir.glob(ckpt_pattern))
        if ckpt_files:
            ckpt_file = sorted(ckpt_files)[-1]
            ckpt_obj = torch.load(ckpt_file, map_location=device, weights_only=False)
            state_dict = ckpt_obj.get("state_dict", ckpt_obj)
            getattr(model, component_name).load_state_dict(state_dict)
            print(f"  Loaded {component_name}: {ckpt_file.name}")
    
    # Setup for forward passes
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0)
    geometry2d = dataset.getitem_geometry2d(0) if hasattr(dataset, 'getitem_geometry2d') else None
    if geometry2d is not None:
        geometry2d = geometry2d.unsqueeze(0).to(device)
    if hasattr(dataset, "getitem_velocity"):
        velocity_base = dataset.getitem_velocity(0).view(1).to(device)
    else:
        velocity_base = torch.zeros(1, device=device)
    
    radius_r = hp['trainer']['radius_graph_r']
    radius_max_nn = hp['trainer']['radius_graph_max_num_neighbors']
    num_supernodes = hp['datasets']['train']['collators'][0].get('num_supernodes', None)
    
    # Autoregressive rollout
    print(f"\n🔄 Starting autoregressive rollout...")
    print(f"  Initial GT timesteps: 0-3 (4 frames)")
    print(f"  Rollout: Predicting timesteps 4-{T-1} ({T-4} frames)")
    
    pred_sequence = []
    num_input_timesteps = 4
    
    with torch.no_grad():
        for t in range(num_input_timesteps, T):
            # Get window of previous 4 timesteps
            # For t=4: use GT[0:4]
            # For t=5: use [GT[1:4], Pred[4]]
            # For t=6: use [GT[2:4], Pred[4:6]]
            # For t=7: use [GT[3], Pred[4:7]]
            # For t=8+: use Pred[t-4:t]
            
            start_idx = t - num_input_timesteps
            window = []
            
            for i in range(start_idx, t):
                if i < num_input_timesteps:
                    # Use ground truth
                    window.append(full_traj[i])  # (N, C)
                else:
                    # Use prediction
                    pred_idx = i - num_input_timesteps
                    window.append(torch.from_numpy(pred_sequence[pred_idx]))  # (N, C)
            
            # Stack window: list of (N, C) -> (N, 4, C) -> (N, 4*C)
            x_window = torch.stack(window, dim=1)  # (N, 4, C)
            x_flat = x_window.reshape(N, -1)  # (N, 4*C)
            x_flat = x_flat.to(device)
            
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
            
            # Prepare inputs
            timestep = torch.tensor([t], dtype=torch.long, device=device)
            velocity = velocity_base
            query_pos_dev = query_pos.unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(
                x=x_flat,
                mesh_pos=mesh_pos,
                query_pos=query_pos_dev,
                mesh_edges=mesh_edges,
                geometry2d=geometry2d,
                timestep=timestep,
                velocity=velocity,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select
            )
            
            # Get prediction
            pred = outputs['x_hat'].cpu().numpy()  # (N, C)
            pred_sequence.append(pred)
            
            if (t - num_input_timesteps + 1) % 10 == 0:
                print(f"  Predicted timestep {t}/{T-1}")
    
    print(f"\n✓ Rollout complete!")
    print(f"  GT frames: {num_input_timesteps}")
    print(f"  Predicted frames: {len(pred_sequence)}")
    
    # Build final sequences for animation
    # gt_sequence: all timesteps from ground truth
    # pred_sequence: first 4 are GT, rest are predictions
    gt_sequence = full_traj.cpu().numpy()  # (T, N, C)
    
    # Build predicted sequence
    pred_seq_full = []
    for t in range(T):
        if t < num_input_timesteps:
            # Use ground truth for initial frames
            pred_seq_full.append(gt_sequence[t])
        else:
            # Use prediction
            pred_idx = t - num_input_timesteps
            pred_seq_full.append(pred_sequence[pred_idx])
    
    pred_seq_full = np.array(pred_seq_full)  # (T, N, C)
    
    print(f"\nFinal sequence shapes:")
    print(f"  GT: {gt_sequence.shape}")
    print(f"  Pred: {pred_seq_full.shape}")
    
    # Get coordinates
    coords = query_pos.cpu().numpy()
    
    # Channel names
    dataset_kind = hp['datasets']['train'].get('kind', '')
    if 'zpinch' in dataset_kind.lower():
        if C == 2:
            names = ['Density', 'Pressure']
        elif C == 6:
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi']
        elif C == 7:
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']
        else:
            names = [f'Variable {i}' for i in range(C)]
    else:
        names = [f'Variable {i}' for i in range(C)]
    
    # Create animation
    print(f"\nCreating animation with {T} total frames...")
    print(f"  Frames 0-{num_input_timesteps-1}: Ground Truth (initial)")
    print(f"  Frames {num_input_timesteps}-{T-1}: Autoregressive Predictions")
    
    # Input data: show first 4 GT frames as "input" panel
    input_data = gt_sequence[0]
    
    create_sequential_animation(
        gt_sequence=gt_sequence,
        pred_sequence=pred_seq_full,
        coords=coords,
        save_path=output_path,
        input_data=input_data,
        interval=100,  # 100ms per frame = 10 fps
        symmetric=True,
        names=names,
        colorbar_type="light",
        show_error=True,
        dynamic_colorscale=False,  # Use fixed colorscale
    )
    
    print(f"\n✓ Animation saved: {output_path}")
    print(f"  Total frames: {T}")
    print(f"  Resolution: {N} points ({int(np.sqrt(N))}x{int(np.sqrt(N))} grid)")
    print(f"  Channels: {C}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create autoregressive rollout animation")
    parser.add_argument("--stage_id", type=str, required=True, help="Stage ID of training run")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint to load")
    parser.add_argument("--output", type=str, default="zpinch_rollout.gif", help="Output path")
    
    args = parser.parse_args()
    
    create_rollout(
        stage_id=args.stage_id,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )

