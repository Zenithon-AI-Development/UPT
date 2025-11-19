#!/usr/bin/env python3
"""
Debug the entire pipeline for a single sample to find issues.
Checks: data loading, normalization, model forward, loss, backward.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
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


def check_gradients(model, loss):
    """Check gradient flow."""
    print("\n" + "="*70)
    print("GRADIENT CHECK")
    print("="*70)
    
    loss.backward(retain_graph=True)
    
    grad_norms = {}
    param_counts = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
        else:
            print(f"  ⚠ NO GRADIENT: {name}")
        param_counts[name] = param.numel()
    
    # Summary by component
    for component in ["encoder", "latent", "decoder", "conditioner"]:
        comp_grads = [v for k, v in grad_norms.items() if component in k]
        comp_params = [v for k, v in param_counts.items() if component in k]
        
        if comp_grads:
            print(f"\n  {component.upper()}:")
            print(f"    Total params: {sum(comp_params):,}")
            print(f"    Avg grad norm: {np.mean(comp_grads):.6e}")
            print(f"    Max grad norm: {np.max(comp_grads):.6e}")
            print(f"    Min grad norm: {np.min(comp_grads):.6e}")
            
            # Check for vanishing gradients
            if np.max(comp_grads) < 1e-8:
                print(f"    ⚠⚠⚠ VANISHING GRADIENTS! ⚠⚠⚠")
            if any(g > 1000 for g in comp_grads):
                print(f"    ⚠⚠⚠ EXPLODING GRADIENTS! ⚠⚠⚠")


def check_data_consistency(dataset, idx=0):
    """Check if data is consistent across calls."""
    print("\n" + "="*70)
    print("DATA CONSISTENCY CHECK")
    print("="*70)
    
    x1 = dataset.getitem_x(idx).clone()
    x2 = dataset.getitem_x(idx).clone()
    
    print(f"  Input consistency: {torch.allclose(x1, x2)}")
    print(f"  Input shape: {x1.shape}")
    print(f"  Input stats: mean={x1.mean():.6f}, std={x1.std():.6f}")
    print(f"  Input range: [{x1.min():.6f}, {x1.max():.6f}]")
    
    t1 = dataset.getitem_target(idx).clone()
    t2 = dataset.getitem_target(idx).clone()
    
    print(f"  Target consistency: {torch.allclose(t1, t2)}")
    print(f"  Target shape: {t1.shape}")
    print(f"  Target stats: mean={t1.mean():.6f}, std={t1.std():.6f}")
    print(f"  Target range: [{t1.min():.6f}, {t1.max():.6f}]")
    
    # Check for NaN/Inf
    print(f"  Input has NaN: {torch.isnan(x1).any()}, Inf: {torch.isinf(x1).any()}")
    print(f"  Target has NaN: {torch.isnan(t1).any()}, Inf: {torch.isinf(t1).any()}")


def check_model_output(model, inputs):
    """Check model output consistency."""
    print("\n" + "="*70)
    print("MODEL OUTPUT CHECK")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        out1 = model(*inputs)
        out2 = model(*inputs)
    
    pred1 = out1["x_hat"]
    pred2 = out2["x_hat"]
    
    print(f"  Output consistency: {torch.allclose(pred1, pred2, atol=1e-6)}")
    print(f"  Prediction shape: {pred1.shape}")
    print(f"  Prediction stats: mean={pred1.mean():.6f}, std={pred1.std():.6f}")
    print(f"  Prediction range: [{pred1.min():.6f}, {pred1.max():.6f}]")
    print(f"  Has NaN: {torch.isnan(pred1).any()}, Inf: {torch.isinf(pred1).any()}")
    
    return pred1


def manual_training_step(model, dataset, device, idx=0, lr=1e-3):
    """Manual training step with full debugging."""
    print("\n" + "="*70)
    print("MANUAL TRAINING STEP")
    print("="*70)
    
    # Get data
    x = dataset.getitem_x(idx).to(device)
    target = dataset.getitem_target(idx).to(device)
    geometry2d = dataset.getitem_geometry2d(idx).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(idx), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(idx), dtype=torch.float32).view(1).to(device)
    mesh_pos = dataset.getitem_mesh_pos(idx).to(device)
    query_pos = dataset.getitem_query_pos(idx).to(device)
    
    print(f"  Input x shape: {x.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Mesh pos shape: {mesh_pos.shape}")
    print(f"  Query pos shape: {query_pos.shape}")
    
    # Setup batch
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    query_pos = query_pos.unsqueeze(0)  # Add batch dimension
    
    # Build graph
    edge_index = radius_graph(x=mesh_pos, r=5.0, batch=batch_idx, loop=True,
                              max_num_neighbors=32, flow="target_to_source")
    mesh_edges = edge_index.T
    
    print(f"  Mesh edges shape: {mesh_edges.shape}")
    
    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    
    # Forward
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    optimizer.zero_grad()
    outputs = model(*inputs)
    pred = outputs["x_hat"]
    
    print(f"\n  Prediction shape: {pred.shape}")
    print(f"  Prediction stats: mean={pred.mean():.6f}, std={pred.std():.6f}")
    
    # Loss
    mse_loss = ((pred - target) ** 2).mean()
    mae_loss = (pred - target).abs().mean()
    rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
    rel_l2 = torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)
    
    print(f"\n  MSE Loss: {mse_loss.item():.8f}")
    print(f"  MAE Loss: {mae_loss.item():.8f}")
    print(f"  Rel L1: {rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%)")
    print(f"  Rel L2: {rel_l2.item():.8f} ({rel_l2.item()*100:.4f}%)")
    
    # Check loss sanity
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        print(f"  ⚠⚠⚠ LOSS IS NaN/Inf! ⚠⚠⚠")
        return None, None, None
    
    # Backward
    mse_loss.backward()
    
    # Check gradients
    check_gradients(model, mse_loss)
    
    # Optimizer step
    optimizer.step()
    
    # Second forward to see improvement
    optimizer.zero_grad()
    outputs2 = model(*inputs)
    pred2 = outputs2["x_hat"]
    mse_loss2 = ((pred2 - target) ** 2).mean()
    rel_l1_2 = (pred2 - target).abs().sum() / (target.abs().sum() + 1e-12)
    
    print(f"\n  After 1 step:")
    print(f"    MSE Loss: {mse_loss2.item():.8f} (changed by {mse_loss2.item() - mse_loss.item():.8f})")
    print(f"    Rel L1: {rel_l1_2.item():.8f} ({rel_l1_2.item()*100:.4f}%)")
    
    if mse_loss2.item() >= mse_loss.item():
        print(f"    ⚠ Loss did NOT decrease! Model may not be learning.")
    
    return pred, target, mse_loss.item()


def main():
    print("="*70)
    print("SINGLE SAMPLE PIPELINE DEBUG")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Setup
    repo_root = Path(__file__).parent.parent
    static = StaticConfig(uri=str(Path(__file__).parent / "static_config.yaml"))
    
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
    
    # Load dataset (1 sample exactly)
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
        max_num_sequences=1,  # Only 1 sequence
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
    )
    
    print(f"Dataset length: {len(dataset)} (should be 1)")
    
    if len(dataset) != 1:
        print(f"⚠ WARNING: Dataset length is {len(dataset)}, expected 1!")
    
    # Check data
    check_data_consistency(dataset, idx=0)
    
    # Create model
    print("\nCreating model...")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    
    model = model_from_kwargs(
        kind="cfd_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=128),
        ),
        encoder=dict(
            kind="encoders.cfd_pool_transformer_perceiver",
            num_latent_tokens=64,
            enc_depth=2,
            kwargs=dict(gnn_dim=64, enc_dim=64, perc_dim=128, enc_num_attn_heads=2, perc_num_attn_heads=2),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=2,
            kwargs=dict(dim=128, num_attn_heads=2),
        ),
        decoder=dict(
            kind="decoders.cfd_transformer_perceiver",
            depth=2,
            use_last_norm=False,
            clamp=0,
            clamp_mode="log",
            kwargs=dict(dim=128, perc_dim=64, num_attn_heads=2, perc_num_attn_heads=2),
        ),
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDC(dataset),
    )
    
    model = model.to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check model output
    print("\n" + "="*70)
    print("SETTING UP MODEL INPUTS FOR CHECK")
    print("="*70)
    
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    geometry2d = dataset.getitem_geometry2d(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32).view(1).to(device)
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long).to(device)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long).to(device)
    unbatch_select = torch.zeros(1, dtype=torch.long).to(device)
    query_pos = query_pos.unsqueeze(0)
    
    edge_index = radius_graph(x=mesh_pos, r=5.0, batch=batch_idx, loop=True,
                              max_num_neighbors=32, flow="target_to_source")
    mesh_edges = edge_index.T
    
    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    
    check_model_output(model, inputs)
    
    # Manual training
    pred, target, loss = manual_training_step(model, dataset, device, idx=0, lr=5e-3)
    
    if pred is not None:
        print("\n" + "="*70)
        print("FINAL CHECK")
        print("="*70)
        print(f"Initial loss: {loss:.8f}")
        print(f"Target range: [{target.min():.6f}, {target.max():.6f}]")
        print(f"Pred range: [{pred.min():.6f}, {pred.max():.6f}]")
        
        if loss > 0.1:
            print("\n⚠ High initial loss - model may have issues")
        if torch.isclose(pred, target, atol=1e-4).all():
            print("✓ Model can produce exact target (good for overfitting)")


if __name__ == "__main__":
    main()



