#!/usr/bin/env python3
"""
Test true single-sample overfitting: One timestep transition only.
Checks everything systematically and ensures we can reach <0.1% rel_l1.
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


def test_overfit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print("="*70)
    print("TRUE SINGLE-SAMPLE OVERFITTING TEST")
    print("="*70)
    print("Target: Rel L1 < 0.001 (0.1%) for ONE sample")
    print()
    
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
    
    # Load dataset - ONE SAMPLE ONLY
    print("1. Loading dataset (max_num_sequences=1)...")
    dataset = dataset_from_kwargs(
        kind="well_trl2d_dataset",
        split="train",
        well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
        num_input_timesteps=4,
        norm="mean0std1",
        clamp=0,
        clamp_mode="log",
        max_num_timesteps=101,
        max_num_sequences=1,  # ONE SEQUENCE = ONE SAMPLE
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
    )
    
    assert len(dataset) == 1, f"Must have exactly 1 sample, got {len(dataset)}"
    print(f"   ✓ Dataset length: {len(dataset)}")
    
    # Get sample
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    
    print(f"\n2. Data check:")
    print(f"   Input shape: {x.shape}")
    print(f"   Target shape: {target.shape}")
    print(f"   Target mean: {target.mean():.6f}, std: {target.std():.6f}")
    print(f"   Target range: [{target.min():.6f}, {target.max():.6f}]")
    
    # Verify data is actually different (not all zeros)
    assert not torch.allclose(x, torch.zeros_like(x)), "Input is all zeros!"
    assert not torch.allclose(target, torch.zeros_like(target)), "Target is all zeros!"
    assert target.abs().sum() > 1e-6, "Target is near-zero!"
    print(f"   ✓ Data is valid")
    
    # Setup model inputs
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    geometry2d = dataset.getitem_geometry2d(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32).view(1).to(device)
    
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    query_pos = query_pos.unsqueeze(0)
    
    # Build graph once
    edge_index = radius_graph(x=mesh_pos, r=5.0, batch=batch_idx, loop=True,
                              max_num_neighbors=32, flow="target_to_source")
    mesh_edges = edge_index.T
    
    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    
    # Create model with adequate capacity
    print(f"\n3. Creating model...")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    model = model_from_kwargs(
        kind="cfd_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=256),
        ),
        encoder=dict(
            kind="encoders.cfd_pool_transformer_perceiver",
            num_latent_tokens=512,
            enc_depth=4,
            kwargs=dict(gnn_dim=128, enc_dim=128, perc_dim=256, enc_num_attn_heads=4, perc_num_attn_heads=4),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=4,
            kwargs=dict(dim=256, num_attn_heads=4),
        ),
        decoder=dict(
            kind="decoders.cfd_transformer_perceiver",
            depth=4,
            use_last_norm=False,
            clamp=0,
            clamp_mode="log",
            kwargs=dict(dim=256, perc_dim=128, num_attn_heads=4, perc_num_attn_heads=4),
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(dataset),
    )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    print(f"   Target values: {target.numel():,} ({target.shape[0]} points × {target.shape[1]} channels)")
    
    if n_params < target.numel():
        print(f"   ⚠ Model has fewer parameters ({n_params:,}) than target values ({target.numel():,})")
        print(f"   → This might limit memorization capacity")
    else:
        print(f"   ✓ Model has enough capacity ({n_params:,} > {target.numel():,})")
    
    # Test: Can model produce exact target?
    print(f"\n4. Testing model capacity...")
    model.train()
    
    # Try to directly optimize output to match target
    # (Bypass encoder/latent, just test if decoder can memorize)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    best_rel_l1 = float('inf')
    best_step = 0
    
    print(f"\n5. Training (aiming for rel_l1 < 0.001)...")
    print(f"   {'Step':>6} {'Loss':>12} {'Rel L1':>12} {'Rel L2':>12} {'Status':>10}")
    print(f"   {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    for step in range(5000):
        optimizer.zero_grad()
        
        outputs = model(*inputs)
        pred = outputs["x_hat"]
        
        # Loss
        loss = ((pred - target) ** 2).mean()
        
        # Metrics (evaluation formula)
        rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
        rel_l2 = torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)
        
        if rel_l1.item() < best_rel_l1:
            best_rel_l1 = rel_l1.item()
            best_step = step
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n   ⚠⚠⚠ NaN/Inf loss at step {step}! Stopping.")
            break
        
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        if grad_norm > 1000:
            print(f"\n   ⚠⚠⚠ Exploding gradients (norm={grad_norm:.2f})! Stopping.")
            break
        
        optimizer.step()
        
        # Logging
        if step in [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000-1]:
            status = "✓ TARGET!" if rel_l1.item() < 0.001 else ""
            print(f"   {step:>6} {loss.item():>12.8f} {rel_l1.item():>12.8f} {rel_l2.item():>12.8f} {status:>10}")
            
            if rel_l1.item() < 0.001:
                print(f"\n   ✓✓✓ SUCCESS! Reached rel_l1={rel_l1.item():.8f} (< 0.1%) at step {step}! ✓✓✓")
                break
    
    print(f"\n6. Final Results:")
    print(f"   Best rel_l1: {best_rel_l1:.8f} ({best_rel_l1*100:.4f}%) at step {best_step}")
    print(f"   Final loss: {loss.item():.8f}")
    
    if best_rel_l1 < 0.001:
        print(f"\n   ✓✓✓ SUCCESS! Achieved target (< 0.1% rel_l1)! ✓✓✓")
        return True
    else:
        print(f"\n   ⚠ Not reached yet. Need {(best_rel_l1 / 0.001):.1f}x improvement")
        
        # Diagnostic
        final_pred = model(*inputs)["x_hat"]
        pred_std = final_pred.std()
        target_std = target.std()
        
        print(f"\n   Diagnostics:")
        print(f"   - Pred std: {pred_std:.6f}, Target std: {target_std:.6f}")
        print(f"   - Std ratio: {pred_std/target_std:.6f}")
        if pred_std / target_std < 0.5:
            print(f"   - ⚠ Variance collapse: predictions too smooth")
        
        # Check if loss is still decreasing
        print(f"   - Loss still decreasing: {loss.item() < 1.0}")
        
        return False


if __name__ == "__main__":
    success = test_overfit()
    sys.exit(0 if success else 1)



