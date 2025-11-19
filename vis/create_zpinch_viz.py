#!/usr/bin/env python
"""
Create visualization from zpinch checkpoint using UPT infrastructure.
Based on time_upt_per_sample.py's model loading approach.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from utils.plotting import create_sequential_animation
from torch_geometric.nn import radius_graph


class MiniDataContainer:
    """Minimal DataContainer for model initialization."""
    def __init__(self, ds):
        self._ds = ds
    def get_dataset(self, *_, **__):
        return self._ds


def load_zpinch_model(stage_id: str, checkpoint: str = "E1000_U1000_S1000"):
    """Load zpinch model from checkpoint."""
    
    repo_root = Path(__file__).resolve().parent
    hp_path = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "hp_resolved.yaml"
    ckpt_dir = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "checkpoints"
    
    print(f"Loading from stage: {stage_id}")
    print(f"Config: {hp_path}")
    print(f"Checkpoint dir: {ckpt_dir}")
    
    # Load config
    with open(hp_path) as f:
        hp = yaml.safe_load(f)
    
    static_config_path = repo_root / "src" / "static_config.yaml"
    static = StaticConfig(uri=str(static_config_path))
    
    # Build dataset providers
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name=f"viz_{stage_id}",
        stage_id=stage_id,
        temp_path=static.temp_path,
    )
    
    # Get test dataset (use train if test is too small)
    test_kwargs = hp["datasets"].get("test", hp["datasets"]["train"])
    if "split" not in test_kwargs:
        test_kwargs["split"] = "test"
    
    dataset = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **test_kwargs,
    )
    
    print(f"Test dataset: {len(dataset)} samples")
    
    # If test set is too small, use train instead
    if len(dataset) < 5:
        print(f"Test set too small, using train split...")
        train_test_kwargs = hp["datasets"]["train"].copy()
        train_test_kwargs["split"] = "train"
        dataset = dataset_from_kwargs(
            dataset_config_provider=dataset_config_provider,
            path_provider=path_provider,
            **train_test_kwargs,
        )
        print(f"Train dataset: {len(dataset)} samples")
    
    # Build model with train dataset for shape inference
    train_kwargs = hp["datasets"]["train"]
    train_ds = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **train_kwargs,
    )
    
    input_shape = train_ds.getshape_x()
    output_shape = train_ds.getshape_target()
    if len(output_shape) == 2:
        output_shape = (output_shape[0], output_shape[1])
    
    print(f"Creating model...")
    model = model_from_kwargs(
        **hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDataContainer(train_ds),
    )
    
    # Load checkpoints for each component
    def _load_sd(path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "weights")):
            for k in ("state_dict", "model", "weights"):
                if k in sd:
                    sd = sd[k]
                    break
        return sd
    
    loaded = []
    for name, stem in {
        "conditioner": "cfd_simformer_model.conditioner",
        "encoder": "cfd_simformer_model.encoder",
        "latent": "cfd_simformer_model.latent",
        "decoder": "cfd_simformer_model.decoder",
    }.items():
        candidates = list(ckpt_dir.glob(f"{stem}*cp={checkpoint}*model.th"))
        if candidates and hasattr(model, name):
            path = candidates[0]
            sd = _load_sd(str(path))
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            getattr(model, name).load_state_dict(sd, strict=False)
            loaded.append(path.name)
            print(f"  Loaded {name}: {path.name}")
    
    if not loaded:
        raise RuntimeError(f"No checkpoint files found for {checkpoint}")
    
    # Get params
    radius_r = hp.get("trainer", {}).get("radius_graph_r", 5.0)
    radius_max_nn = hp.get("trainer", {}).get("radius_graph_max_num_neighbors", 32)
    num_supernodes = hp.get("vars", {}).get("num_supernodes", 1024)
    
    return model, dataset, radius_r, radius_max_nn, num_supernodes, hp


def create_animation(stage_id: str, checkpoint: str = "latest", output_path: str = 'zpinch_animation.gif', n_frames: int = 15):
    """Create animation from zpinch model."""
    
    # Load model and dataset
    model, dataset, radius_r, radius_max_nn, num_supernodes, hp = load_zpinch_model(stage_id, checkpoint)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    print(f"\nModel on {device}")
    print(f"Generating predictions for {min(n_frames, len(dataset))} samples...")
    
    gt_sequence = []
    pred_sequence = []
    
    with torch.no_grad():
        for i in range(min(n_frames, len(dataset))):
            # Get data items
            x = dataset.getitem_x(i)
            target = dataset.getitem_target(i)
            geometry2d = dataset.getitem_geometry2d(i) if hasattr(dataset, 'getitem_geometry2d') else None
            timestep = dataset.getitem_timestep(i)
            velocity = dataset.getitem_velocity(i) if hasattr(dataset, 'getitem_velocity') else None
            mesh_pos = dataset.getitem_mesh_pos(i)
            query_pos = dataset.getitem_query_pos(i)
            
            # Convert timestep and velocity to tensors
            if not torch.is_tensor(timestep):
                timestep = torch.as_tensor(timestep, dtype=torch.long).view(1)
            else:
                timestep = timestep.view(1)
            
            if velocity is not None and not torch.is_tensor(velocity):
                velocity = torch.as_tensor(velocity, dtype=torch.float32).view(1)
            elif velocity is not None:
                velocity = velocity.view(1)
            
            # Create batch indices
            N = x.shape[0]
            Nq = query_pos.shape[0]
            batch_idx = torch.zeros(N, dtype=torch.long)
            unbatch_idx = torch.zeros(Nq, dtype=torch.long)
            unbatch_select = torch.zeros(1, dtype=torch.long)
            
            # Move to device
            x = x.to(device)
            target = target.to(device)
            mesh_pos = mesh_pos.to(device)
            query_pos = query_pos.unsqueeze(0).to(device)  # Add batch dim ONLY to query_pos
            if geometry2d is not None:
                geometry2d = geometry2d.to(device)
            timestep = timestep.to(device)
            if velocity is not None:
                velocity = velocity.to(device)
            batch_idx = batch_idx.to(device)
            unbatch_idx = unbatch_idx.to(device)
            unbatch_select = unbatch_select.to(device)
            
            # Create radius graph
            flow = "target_to_source" if num_supernodes is not None else "source_to_target"
            edge_index = radius_graph(
                x=mesh_pos, r=radius_r, batch=batch_idx, loop=True,
                max_num_neighbors=radius_max_nn, flow=flow
            )
            mesh_edges = edge_index.T
            
            # Forward pass - pass as tuple like in time_upt_per_sample
            inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
                     mesh_edges, batch_idx, unbatch_idx, unbatch_select)
            outputs = model(*inputs)
            
            # Extract predictions
            pred = outputs['x_hat'].cpu().numpy()
            gt = target.cpu().numpy()
            
            gt_sequence.append(gt)
            pred_sequence.append(pred)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{min(n_frames, len(dataset))}")
    
    gt_sequence = np.array(gt_sequence)
    pred_sequence = np.array(pred_sequence)
    
    print(f"Sequence shape: {gt_sequence.shape}")
    
    # Get coordinates for first sample
    coords = dataset.getitem_query_pos(0).cpu().numpy()
    
    # Get input
    x0 = dataset.getitem_x(0).cpu().numpy()
    n_channels = gt_sequence.shape[-1]
    n_timesteps = x0.shape[-1] // n_channels
    x_reshaped = x0.reshape(-1, n_timesteps, n_channels)
    u_inp = x_reshaped[:, -1, :]
    
    # Get variable names from config
    dataset_kind = hp['datasets']['test'].get('kind', '')
    if 'zpinch' in dataset_kind.lower():
        if n_channels == 2:
            names = ['Density', 'Pressure']
        elif n_channels == 6:
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi']
        elif n_channels == 7:
            # 7 channels: density, pressure, velx, vely, magz, magp, current_drive
            names = ['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']
        else:
            names = [f'Variable {i}' for i in range(n_channels)]
    else:
        names = [f'Variable {i}' for i in range(n_channels)]
    
    print(f"\nCreating animation...")
    print(f"  Channels: {n_channels}")
    print(f"  Names: {names}")
    
    create_sequential_animation(
        gt_sequence=gt_sequence,
        pred_sequence=pred_sequence,
        coords=coords,
        save_path=output_path,
        input_data=u_inp,
        interval=300,
        symmetric=False,
        names=names,
        colorbar_type="light",
        show_error=True,
        dynamic_colorscale=False
    )
    
    print(f"\n✓ Animation saved: {output_path}")
    print(f"  Frames: {len(gt_sequence)}")
    print(f"  Resolution: {coords.shape[0]}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage_id', type=str, required=True,
                       help='Stage ID (e.g., zv7z5fzc)')
    parser.add_argument('--checkpoint', type=str, default='E1000_U1000_S1000',
                       help='Checkpoint name')
    parser.add_argument('--output', type=str, default='zpinch_animation.gif')
    parser.add_argument('--n_frames', type=int, default=15)
    
    args = parser.parse_args()
    
    create_animation(
        stage_id=args.stage_id,
        checkpoint=args.checkpoint,
        output_path=args.output,
        n_frames=args.n_frames
    )

