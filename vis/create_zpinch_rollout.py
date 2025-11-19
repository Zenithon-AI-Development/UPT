#!/usr/bin/env python
"""
Create autoregressive rollout animation for ZPinch models.
Shows: GT[0:4] | GT[4] | Pred[4] | ... (like GAOT)
"""
import sys
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.kappaconfig.util import get_stage_hp
from utils.factory import instantiate
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from providers.path_provider import PathProvider
from providers.dataset_config_provider import DatasetConfigProvider
from torch_geometric.nn import radius_graph
from utils.plotting import create_sequential_animation


def create_rollout_animation(
    stage_id: str,
    checkpoint: str = "latest",
    output_path: str = 'zpinch_rollout.gif',
    n_rollout_steps: int = 97,  # Rollout from timestep 4 to 101
    num_input_timesteps: int = 4,
):
    """
    Create autoregressive rollout animation.
    
    Args:
        stage_id: Training run ID
        checkpoint: Which checkpoint to load
        output_path: Where to save the animation
        n_rollout_steps: Number of autoregressive steps (101 - num_input_timesteps)
        num_input_timesteps: Number of initial GT timesteps
    """
    print(f"Loading from stage: {stage_id}")
    
    # Load config
    output_path_obj = Path("/home/workspace/projects/transformer/UPT/benchmarking/save")
    config_path = output_path_obj / "stage1" / stage_id / "hp_resolved.yaml"
    print(f"Config: {config_path}")
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    # Get dataset
    path_provider = PathProvider(output_path=str(output_path_obj))
    dataset_config_provider = DatasetConfigProvider()
    
    # Use train split for rollout (has longer sequences)
    test_kwargs = hp["datasets"].get("train", hp["datasets"]["train"])
    test_kwargs["split"] = "train"
    test_kwargs["max_num_sequences"] = 1  # Just one trajectory for visualization
    test_kwargs["num_input_timesteps"] = num_input_timesteps  # 4 input timesteps
    
    dataset = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **test_kwargs,
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Get model
    checkpoint_dir = output_path_obj / "stage1" / stage_id / "checkpoints"
    print(f"Checkpoint dir: {checkpoint_dir}")
    
    # Find checkpoint files
    if checkpoint == "latest":
        ckpt_files = sorted(checkpoint_dir.glob("*.th"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Create model
    print("Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_kwargs = hp["model"].copy()
    model_kind = model_kwargs.pop("kind")
    
    # Create mini data container
    class MiniDataContainer:
        def __init__(self, dataset):
            self.dataset = dataset
        def get_dataset(self, key):
            return self.dataset
    
    data_container = MiniDataContainer(dataset)
    model = model_from_kwargs(kind=model_kind, data_container=data_container, **model_kwargs)
    model = model.to(device)
    model.eval()
    
    # Load checkpoints for each component
    for component_name in ['conditioner', 'encoder', 'latent', 'decoder']:
        full_name = f"cfd_simformer_model.{component_name}"
        ckpt_pattern = f"{full_name} cp=*model.th"
        ckpt_files = list(checkpoint_dir.glob(ckpt_pattern))
        if ckpt_files:
            ckpt_file = sorted(ckpt_files)[-1]
            state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)
            getattr(model, component_name).load_state_dict(state_dict)
            print(f"  Loaded {component_name}: {ckpt_file.name}")
    
    print(f"\nModel on {device}")
    
    # Get a single trajectory (first sample)
    print(f"\nPerforming autoregressive rollout...")
    
    # Get all data for first sample
    sample = dataset[0]
    
    # Extract components
    x = sample['x']  # Input features (flattened timesteps)
    target = sample['target']
    mesh_pos = sample['mesh_pos']
    query_pos = sample['query_pos']
    geometry2d = sample.get('geometry2d', None)
    timestep = sample.get('timestep', 0.0)
    velocity = sample.get('velocity', None)
    
    # Get dimensions
    n_channels = target.shape[-1]
    n_points = query_pos.shape[0]
    
    # Reshape x to get initial timesteps: (N, T*C) -> (N, T, C)
    x_reshaped = x.reshape(n_points, num_input_timesteps, n_channels)
    
    # Initialize sequences
    # GT sequence: need to load full trajectory ground truth
    # For now, we'll create rollout from first 4 timesteps
    
    # Start with initial GT frames
    gt_frames = [x_reshaped[:, i, :].cpu().numpy() for i in range(num_input_timesteps)]
    pred_frames = []
    
    # Current state for rollout
    current_window = x_reshaped.clone()  # (N, 4, C)
    
    print(f"  Initial GT frames: {num_input_timesteps}")
    print(f"  Rollout steps: {n_rollout_steps}")
    
    # Autoregressive rollout
    with torch.no_grad():
        for step in range(n_rollout_steps):
            # Flatten current window: (N, 4, C) -> (N, 4*C)
            x_flat = current_window.reshape(n_points, -1)
            
            # Setup for forward pass
            N = x_flat.shape[0]
            Nq = query_pos.shape[0]
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
            unbatch_idx = torch.zeros(Nq, dtype=torch.long, device=device)
            unbatch_select = torch.zeros(1, dtype=torch.long, device=device)
            
            # Move to device
            x_input = x_flat.to(device)
            mesh_pos_dev = mesh_pos.to(device)
            query_pos_dev = query_pos.unsqueeze(0).to(device)
            
            # Create radius graph
            radius_r = hp['trainer']['radius_graph_r']
            radius_max_nn = hp['trainer']['radius_graph_max_num_neighbors']
            num_supernodes = hp['datasets']['train']['collators'][0].get('num_supernodes', None)
            
            flow = "target_to_source" if num_supernodes is not None else "source_to_target"
            edge_index = radius_graph(
                x=mesh_pos_dev, r=radius_r, batch=batch_idx, loop=True,
                max_num_neighbors=radius_max_nn, flow=flow
            )
            mesh_edges = edge_index.T
            
            # Prepare inputs
            geom_dev = geometry2d.to(device) if geometry2d is not None else None
            timestep_dev = torch.as_tensor(timestep, dtype=torch.float32, device=device).view(1)
            velocity_dev = torch.as_tensor(velocity, dtype=torch.float32, device=device).view(1) if velocity is not None else None
            
            # Forward pass
            inputs = (x_input, geom_dev, timestep_dev, velocity_dev, mesh_pos_dev, query_pos_dev,
                     mesh_edges, batch_idx, unbatch_idx, unbatch_select)
            outputs = model(*inputs)
            
            # Get prediction
            pred = outputs['x_hat'].cpu()  # (N, C)
            pred_frames.append(pred.numpy())
            
            # Update window: shift left and add prediction
            # current_window[:, 0:3, :] = current_window[:, 1:4, :]
            # current_window[:, 3, :] = pred
            current_window = torch.cat([current_window[:, 1:, :], pred.unsqueeze(1)], dim=1)
            
            if (step + 1) % 10 == 0:
                print(f"  Rollout step {step+1}/{n_rollout_steps}")
    
    # Combine GT and predictions
    # Frames: GT[0], GT[1], GT[2], GT[3], Pred[4], Pred[5], ...
    all_gt = gt_frames + [gt_frames[-1]] * n_rollout_steps  # Pad with last GT
    all_pred = [None] * num_input_timesteps + pred_frames  # First 4 are GT, rest are predictions
    
    # Stack into sequences
    gt_sequence = np.array(all_gt)  # (T, N, C)
    pred_sequence_list = []
    for i, p in enumerate(all_pred):
        if p is None:
            # For initial frames, use GT
            pred_sequence_list.append(gt_sequence[i])
        else:
            pred_sequence_list.append(p)
    pred_sequence = np.array(pred_sequence_list)  # (T, N, C)
    
    print(f"\nSequence shapes:")
    print(f"  GT: {gt_sequence.shape}")
    print(f"  Pred: {pred_sequence.shape}")
    
    # Get coordinates
    coords = query_pos.cpu().numpy()
    
    # Channel names
    dataset_kind = hp['datasets']['train'].get('kind', '')
    if 'zpinch' in dataset_kind.lower():
        if n_channels == 2:
            names = ['Density', 'Pressure']
        elif n_channels == 6:
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi']
        elif n_channels == 7:
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']
        else:
            names = [f'Variable {i}' for i in range(n_channels)]
    else:
        names = [f'Variable {i}' for i in range(n_channels)]
    
    # Create animation
    print(f"\nCreating animation with {len(gt_sequence)} frames...")
    print(f"  First {num_input_timesteps} frames: Ground Truth")
    print(f"  Frames {num_input_timesteps}+: Autoregressive Predictions")
    
    # Add input data (show first 4 GT frames as "input")
    input_data = gt_sequence[:num_input_timesteps]
    
    create_sequential_animation(
        gt_sequence=gt_sequence,
        pred_sequence=pred_sequence,
        coords=coords,
        save_path=output_path,
        input_data=input_data,
        interval=200,  # 200ms per frame = 5 fps
        symmetric=True,
        names=names,
        colorbar_type="light",
        show_error=True,
        dynamic_colorscale=False,  # Use fixed colorscale for stability
    )
    
    print(f"\n✓ Animation saved: {output_path}")
    print(f"  Total frames: {len(gt_sequence)}")
    print(f"  GT initial: frames 0-{num_input_timesteps-1}")
    print(f"  Predictions: frames {num_input_timesteps}-{len(gt_sequence)-1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create autoregressive rollout animation for ZPinch")
    parser.add_argument("--stage_id", type=str, required=True, help="Stage ID of training run")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint to load")
    parser.add_argument("--output", type=str, default="zpinch_rollout.gif", help="Output path")
    parser.add_argument("--n_rollout_steps", type=int, default=97, help="Number of rollout steps")
    parser.add_argument("--num_input_timesteps", type=int, default=4, help="Number of initial GT timesteps")
    
    args = parser.parse_args()
    
    create_rollout_animation(
        stage_id=args.stage_id,
        checkpoint=args.checkpoint,
        output_path=args.output,
        n_rollout_steps=args.n_rollout_steps,
        num_input_timesteps=args.num_input_timesteps,
    )


