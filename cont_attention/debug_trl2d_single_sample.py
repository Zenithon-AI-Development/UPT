#!/usr/bin/env python3
"""
Debug script to diagnose issues with TRL2D dataset training.
Checks data loading, normalization, model forward pass, and loss computation.
"""
import torch
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


def check_data_statistics(dataset, num_samples=5):
    """Check data statistics and ranges."""
    print("\n" + "="*70)
    print("DATA STATISTICS CHECK")
    print("="*70)
    
    x_vals, target_vals = [], []
    
    for idx in range(min(num_samples, len(dataset))):
        x = dataset.getitem_x(idx)
        target = dataset.getitem_target(idx)
        x_vals.append(x)
        target_vals.append(target)
        
        print(f"\nSample {idx}:")
        print(f"  x shape: {x.shape}, target shape: {target.shape}")
        print(f"  x range: [{x.min():.4f}, {x.max():.4f}], mean: {x.mean():.4f}, std: {x.std():.4f}")
        print(f"  target range: [{target.min():.4f}, {target.max():.4f}], mean: {target.mean():.4f}, std: {target.std():.4f}")
    
    # Check if data looks normalized
    x_all = torch.cat([x.flatten() for x in x_vals])
    target_all = torch.cat([t.flatten() for t in target_vals])
    
    print(f"\nOverall statistics:")
    print(f"  x - mean: {x_all.mean():.4f}, std: {x_all.std():.4f}")
    print(f"  target - mean: {target_all.mean():.4f}, std: {target_all.std():.4f}")
    
    if abs(x_all.mean()) < 0.1 and abs(x_all.std() - 1.0) < 0.2:
        print("  ✓ Input data appears to be normalized (mean~0, std~1)")
    else:
        print("  ⚠ Input data may not be properly normalized!")
    
    if abs(target_all.mean()) < 0.1 and abs(target_all.std() - 1.0) < 0.2:
        print("  ✓ Target data appears to be normalized (mean~0, std~1)")
    else:
        print("  ⚠ Target data may not be properly normalized!")


def check_model_forward(model, dataset, device="cuda"):
    """Check model forward pass."""
    print("\n" + "="*70)
    print("MODEL FORWARD PASS CHECK")
    print("="*70)
    
    model = model.to(device).eval()
    
    # Get first sample
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    geometry2d = dataset.getitem_geometry2d(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32).view(1).to(device)
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    
    # Create batch indices
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    
    # Add batch dimension to query_pos
    query_pos = query_pos.unsqueeze(0)
    
    # Build mesh edges
    edge_index = radius_graph(
        x=mesh_pos, r=5.0, batch=batch_idx, loop=True,
        max_num_neighbors=32, flow="target_to_source"
    )
    mesh_edges = edge_index.T
    
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  target: {target.shape}")
    print(f"  mesh_pos: {mesh_pos.shape}")
    print(f"  query_pos: {query_pos.shape}")
    print(f"  mesh_edges: {mesh_edges.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            x, geometry2d, timestep, velocity, mesh_pos, query_pos,
            mesh_edges, batch_idx, unbatch_idx, unbatch_select
        )
    
    pred = outputs["x_hat"]
    print(f"\nOutput shape: {pred.shape}")
    print(f"Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"Prediction mean: {pred.mean():.4f}, std: {pred.std():.4f}")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"Target mean: {target.mean():.4f}, std: {target.std():.4f}")
    
    # Compute errors
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rel_l1 = (torch.abs(pred - target).sum() / (torch.abs(target).sum() + 1e-12)).item()
    rel_l2 = (torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)).item()
    
    print(f"\nInitial errors (random init):")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Rel L1: {rel_l1:.6f}")
    print(f"  Rel L2: {rel_l2:.6f}")
    
    return pred, target, mse


def check_loss_computation(pred, target):
    """Check loss computation."""
    print("\n" + "="*70)
    print("LOSS COMPUTATION CHECK")
    print("="*70)
    
    # MSE loss
    mse = torch.mean((pred - target) ** 2)
    print(f"MSE Loss: {mse.item():.6f}")
    
    # Check if loss is reasonable
    if mse.item() > 1e6:
        print("  ⚠ Loss is very large! May indicate normalization issues.")
    elif mse.item() < 1e-6:
        print("  ⚠ Loss is very small! May indicate data or model issues.")
    else:
        print("  ✓ Loss magnitude seems reasonable")
    
    # Check gradients
    pred_with_grad = pred.clone().requires_grad_(True)
    loss = torch.mean((pred_with_grad - target) ** 2)
    loss.backward()
    
    if pred_with_grad.grad is not None:
        grad_norm = pred_with_grad.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.6f}")
        if grad_norm > 1e6:
            print("  ⚠ Gradient is very large! May cause instability.")
        elif grad_norm < 1e-8:
            print("  ⚠ Gradient is very small! May cause slow learning.")
        else:
            print("  ✓ Gradient magnitude seems reasonable")


def main():
    print("="*70)
    print("TRL2D SINGLE SAMPLE DEBUGGING")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    static_config_path = Path(__file__).parent / "static_config.yaml"
    static = StaticConfig(uri=str(static_config_path))
    
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name="debug",
        stage_id="debug",
        temp_path=static.temp_path,
    )
    
    # Load dataset (1 sample)
    print("\nLoading dataset...")
    dataset = dataset_from_kwargs(
        kind="well_trl2d_dataset",
        split="train",
        well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
        num_input_timesteps=4,
        norm="mean0std1",
        clamp=0,
        clamp_mode="log",
        max_num_timesteps=101,
        max_num_sequences=1,
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Check data
    check_data_statistics(dataset)
    
    # Create model
    print("\nCreating model...")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    model = model_from_kwargs(
        kind="cfd_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=192),
        ),
        encoder=dict(
            kind="encoders.cfd_pool_transformer_perceiver",
            num_latent_tokens=128,
            enc_depth=2,
            kwargs=dict(gnn_dim=96, enc_dim=96, perc_dim=192, enc_num_attn_heads=2, perc_num_attn_heads=3),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=2,
            kwargs=dict(dim=192, num_attn_heads=3),
        ),
        decoder=dict(
            kind="decoders.cfd_transformer_perceiver",
            depth=2,
            use_last_norm=True,
            clamp=0,
            clamp_mode="log",
            kwargs=dict(dim=192, perc_dim=96, num_attn_heads=3, perc_num_attn_heads=2),
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(dataset),
    )
    
    print(f"Model created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check forward pass
    pred, target, mse = check_model_forward(model, dataset, device)
    
    # Check loss computation
    check_loss_computation(pred, target)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. If data looks normalized -> proceed with overfitting test")
    print("2. If loss/gradients are unstable -> check normalization or learning rate")
    print("3. Run: python main_train.py --hp yamls/trl2d/trl2d_overfit_single.yaml")


if __name__ == "__main__":
    main()



