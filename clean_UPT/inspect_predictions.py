#!/usr/bin/env python3
"""
Inspect model predictions vs targets to diagnose normalization issues.
"""
import torch
import yaml
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider  
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from torch_geometric.nn import radius_graph


class MiniDC:
    def __init__(self, ds):
        self._ds = ds
    def get_dataset(self, *_, **__):
        return self._ds


def inspect_single_prediction(stage_id, sample_idx=0):
    """Inspect a single prediction in detail."""
    # Load model and dataset
    repo_root = Path(__file__).resolve().parents[1]
    hp_path = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "hp_resolved.yaml"
    ckpt_dir = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "checkpoints"
    
    with open(hp_path) as f:
        hp = yaml.safe_load(f)
    
    static = StaticConfig(uri=str(repo_root / "src" / "static_config.yaml"))
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name=f"inspect_{stage_id}",
        stage_id=stage_id,
        temp_path=static.temp_path,
    )
    
    # Load dataset
    test_kwargs = hp["datasets"].get("test", hp["datasets"]["train"])
    dataset = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **test_kwargs,
    )
    
    # Load model
    train_kwargs = hp["datasets"]["train"]
    train_ds = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **train_kwargs,
    )
    
    input_shape = train_ds.getshape_x()
    output_shape = train_ds.getshape_target()
    
    model = model_from_kwargs(
        **hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(train_ds),
    )
    
    # Load checkpoints
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_pattern = "E1000_U1000_S1000"
    for component in ["conditioner", "encoder", "latent", "decoder"]:
        ckpt_files = list(ckpt_dir.glob(f"*{component}*cp={checkpoint_pattern}*model.th"))
        if ckpt_files and hasattr(model, component):
            state = torch.load(str(ckpt_files[0]), map_location="cpu")
            sd = state.get("state_dict", state)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            getattr(model, component).load_state_dict(sd, strict=False)
    
    model = model.to(device).eval()
    
    # Get sample
    x = dataset.getitem_x(sample_idx).to(device)
    target = dataset.getitem_target(sample_idx).to(device)
    geometry2d = dataset.getitem_geometry2d(sample_idx).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(sample_idx), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(sample_idx), dtype=torch.float32).view(1).to(device)
    mesh_pos = dataset.getitem_mesh_pos(sample_idx).to(device)
    query_pos = dataset.getitem_query_pos(sample_idx).to(device)
    
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    query_pos = query_pos.unsqueeze(0)
    
    # Build graph
    radius_r = hp.get("trainer", {}).get("radius_graph_r", 5.0)
    radius_max_nn = hp.get("trainer", {}).get("radius_graph_max_num_neighbors", 32)
    num_supernodes = hp.get("vars", {}).get("num_supernodes", 2048)
    
    flow = "target_to_source" if num_supernodes else "source_to_target"
    edge_index = radius_graph(x=mesh_pos, r=radius_r, batch=batch_idx, loop=True,
                              max_num_neighbors=radius_max_nn, flow=flow)
    mesh_edges = edge_index.T
    
    # Forward pass
    with torch.no_grad():
        outputs = model(x, geometry2d, timestep, velocity, mesh_pos, query_pos,
                       mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    
    pred = outputs["x_hat"]
    
    # Detailed analysis
    print("\n" + "="*70)
    print("DETAILED PREDICTION ANALYSIS")
    print("="*70)
    
    print(f"\nShapes:")
    print(f"  Input (x): {x.shape}")
    print(f"  Target: {target.shape}")
    print(f"  Prediction: {pred.shape}")
    
    print(f"\nInput statistics:")
    print(f"  Range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"  Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
    
    print(f"\nTarget statistics:")
    print(f"  Range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    print(f"  Mean: {target.mean().item():.4f}, Std: {target.std().item():.4f}")
    
    print(f"\nPrediction statistics:")
    print(f"  Range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"  Mean: {pred.mean().item():.4f}, Std: {pred.std().item():.4f}")
    
    # Per-channel analysis
    num_channels = target.shape[1]
    print(f"\nPer-channel analysis ({num_channels} channels):")
    for c in range(num_channels):
        target_c = target[:, c]
        pred_c = pred[:, c]
        mse_c = ((pred_c - target_c) ** 2).mean().item()
        mae_c = (pred_c - target_c).abs().mean().item()
        rel_l1_c = (pred_c - target_c).abs().sum().item() / (target_c.abs().sum().item() + 1e-12)
        
        print(f"  Channel {c}:")
        print(f"    Target: mean={target_c.mean().item():.4f}, std={target_c.std().item():.4f}")
        print(f"    Pred:   mean={pred_c.mean().item():.4f}, std={pred_c.std().item():.4f}")
        print(f"    MSE: {mse_c:.6f}, MAE: {mae_c:.6f}, Rel L1: {rel_l1_c:.6f}")
    
    # Overall metrics
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    rel_l1 = (pred - target).abs().sum().item() / (target.abs().sum().item() + 1e-12)
    
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    rel_l2 = (
        torch.linalg.vector_norm(pred_flat - target_flat, ord=2, dim=1).sum() /
        (torch.linalg.vector_norm(target_flat, ord=2, dim=1).sum() + 1e-12)
    ).item()
    
    print(f"\nOverall metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Rel L1: {rel_l1:.6f} ({rel_l1*100:.2f}%)")
    print(f"  Rel L2: {rel_l2:.6f} ({rel_l2*100:.2f}%)")
    
    # Check if there's a systematic bias
    bias = (pred - target).mean(dim=0)
    print(f"\nSystematic bias (mean error per channel):")
    for c in range(num_channels):
        print(f"  Channel {c}: {bias[c].item():.6f}")
    
    # Correlation
    print(f"\nPer-channel correlation:")
    for c in range(num_channels):
        target_c = target[:, c].flatten()
        pred_c = pred[:, c].flatten()
        corr = torch.corrcoef(torch.stack([target_c, pred_c]))[0, 1].item()
        print(f"  Channel {c}: {corr:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage_id", default="qii7e18l")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()
    
    inspect_single_prediction(args.stage_id, args.sample_idx)



