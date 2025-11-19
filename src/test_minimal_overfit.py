#!/usr/bin/env python3
"""
Minimal test: can we overfit a single sample with minimal model?
"""
import torch
import torch.nn as nn
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


def test_overfit_single_sample():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()  # Clear memory
    
    print("="*70)
    print("MINIMAL OVERFITTING TEST - SINGLE SAMPLE")
    print("="*70)
    
    # Setup
    static = StaticConfig(uri=str(Path(__file__).parent / "static_config.yaml"))
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name="test",
        stage_id="test",
        temp_path=static.temp_path,
    )
    
    # Load dataset (1 sample)
    print("\nLoading dataset (max_num_sequences=1)...")
    dataset = dataset_from_kwargs(
        kind="well_trl2d_dataset",
        split="train",
        well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
        num_input_timesteps=4,
        norm="mean0std1",
        clamp=0,
        clamp_mode="log",
        max_num_timesteps=101,
        max_num_sequences=1,  # ONE SAMPLE ONLY
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
    )
    
    print(f"  Dataset length: {len(dataset)} (should be 1)")
    assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
    
    # Get sample
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    geometry2d = dataset.getitem_geometry2d(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32).view(1).to(device)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Target stats: mean={target.mean():.6f}, std={target.std():.6f}")
    
    # Setup batch
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    query_pos = query_pos.unsqueeze(0)
    
    # Build graph
    edge_index = radius_graph(x=mesh_pos, r=5.0, batch=batch_idx, loop=True,
                              max_num_neighbors=32, flow="target_to_source")
    mesh_edges = edge_index.T
    
    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    
    # Create SMALL model for overfitting
    print("\nCreating minimal model...")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    model = model_from_kwargs(
        kind="cfd_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=64),
        ),
        encoder=dict(
            kind="encoders.cfd_pool_transformer_perceiver",
            num_latent_tokens=32,  # Very small
            enc_depth=1,
            kwargs=dict(gnn_dim=32, enc_dim=32, perc_dim=64, enc_num_attn_heads=2, perc_num_attn_heads=2),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=1,
            kwargs=dict(dim=64, num_attn_heads=2),
        ),
        decoder=dict(
            kind="decoders.cfd_transformer_perceiver",
            depth=1,
            use_last_norm=False,
            clamp=0,
            clamp_mode="log",
            kwargs=dict(dim=64, perc_dim=32, num_attn_heads=2, perc_num_attn_heads=2),
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(dataset),
    )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Train with high LR
    print("\nTraining with LR=0.01 for 1000 steps...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for step in range(1000):
        optimizer.zero_grad()
        
        outputs = model(*inputs)
        pred = outputs["x_hat"]
        
        # Loss
        loss = ((pred - target) ** 2).mean()
        
        # Metrics (using evaluation formula)
        rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
        rel_l2 = torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)
        
        loss.backward()
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        if step in [0, 1, 5, 10, 50, 100, 500, 999]:
            print(f"  Step {step:4d}: Loss={loss.item():.8f}, "
                  f"Rel L1={rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%), "
                  f"Rel L2={rel_l2.item():.8f} ({rel_l2.item()*100:.4f}%)")
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Final Loss: {loss.item():.8f}")
    print(f"Final Rel L1: {rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%)")
    print(f"Final Rel L2: {rel_l2.item():.8f} ({rel_l2.item()*100:.4f}%)")
    print(f"\nTarget: Rel L1 < 0.001 (0.1%)")
    
    if rel_l1.item() < 0.001:
        print("✓✓✓ SUCCESS! Achieved < 0.1% rel_l1! ✓✓✓")
    else:
        print(f"⚠ Not reached yet. Current: {rel_l1.item()*100:.4f}%")
        print(f"  Need {(rel_l1.item() / 0.001):.1f}x improvement")
        
        # Check if loss is still decreasing
        print(f"\n  Loss trend: Still decreasing? {loss.item() < 0.1}")
        if loss.item() > 0.1:
            print("  ⚠ Loss still high - model may have capacity or training issues")


if __name__ == "__main__":
    test_overfit_single_sample()



