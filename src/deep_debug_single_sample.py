#!/usr/bin/env python3
"""
Deep debugging to find why overfitting doesn't reach 0.1% rel_l1.
"""
import torch
import torch.nn as nn
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


def check_gradient_flow(model):
    """Check if gradients are flowing to all parameters."""
    print("\n" + "="*70)
    print("GRADIENT FLOW CHECK")
    print("="*70)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name:60s} grad_norm={grad_norm:.6e}")
        else:
            print(f"  {name:60s} NO GRADIENT!")


def check_activations(model, inputs):
    """Check for NaN or saturation in activations."""
    print("\n" + "="*70)
    print("ACTIVATION CHECK")
    print("="*70)
    
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(*inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Check activations
    for name, act in activations.items():
        if act.numel() > 0:
            has_nan = torch.isnan(act).any().item()
            has_inf = torch.isinf(act).any().item()
            mean = act.mean().item()
            std = act.std().item()
            
            if has_nan or has_inf:
                print(f"  ⚠ {name}: NaN={has_nan}, Inf={has_inf}")
            elif abs(mean) > 100 or std > 100:
                print(f"  ⚠ {name}: mean={mean:.2e}, std={std:.2e} (LARGE VALUES)")


def manual_training_loop(model, dataset, device, num_epochs=100, lr=1e-3):
    """Manual training loop with detailed debugging."""
    print("\n" + "="*70)
    print("MANUAL TRAINING WITH DETAILED DEBUGGING")
    print("="*70)
    
    # Get sample
    x = dataset.getitem_x(0).to(device)
    target = dataset.getitem_target(0).to(device)
    geometry2d = dataset.getitem_geometry2d(0).to(device)
    timestep = torch.as_tensor(dataset.getitem_timestep(0), dtype=torch.long).view(1).to(device)
    velocity = torch.as_tensor(dataset.getitem_velocity(0), dtype=torch.float32).view(1).to(device)
    mesh_pos = dataset.getitem_mesh_pos(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    
    # Setup batch
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
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Print initial state
    print(f"\nTarget statistics:")
    print(f"  Shape: {target.shape}")
    print(f"  Range: [{target.min():.4f}, {target.max():.4f}]")
    print(f"  Mean: {target.mean():.4f}, Std: {target.std():.4f}")
    print(f"  Per-channel std: {[target[:, i].std().item() for i in range(target.shape[1])]}")
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        outputs = model(*inputs)
        pred = outputs["x_hat"]
        
        # Compute loss
        loss = ((pred - target) ** 2).mean()
        
        # Compute metrics
        rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
        rel_l2 = torch.linalg.vector_norm(pred - target) / (torch.linalg.vector_norm(target) + 1e-12)
        
        # Backward
        loss.backward()
        
        # Check for gradient issues
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        optimizer.step()
        
        # Detailed logging
        if epoch in [0, 1, 2, 5, 10, 20, 50, 99] or epoch % 50 == 0:
            print(f"\nEpoch {epoch:4d}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Rel L1: {rel_l1.item():.6f} ({rel_l1.item()*100:.2f}%)")
            print(f"  Rel L2: {rel_l2.item():.6f} ({rel_l2.item()*100:.2f}%)")
            print(f"  Pred mean: {pred.mean().item():.6f}, std: {pred.std().item():.6f}")
            print(f"  Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"  Total grad norm: {total_grad_norm:.6e}")
            
            # Per-channel pred std
            pred_stds = [pred[:, i].std().item() for i in range(pred.shape[1])]
            target_stds = [target[:, i].std().item() for i in range(target.shape[1])]
            print(f"  Pred std by channel: {[f'{s:.4f}' for s in pred_stds]}")
            print(f"  Target std by channel: {[f'{s:.4f}' for s in target_stds]}")
            
            if epoch in [0, 1, 2]:
                # Check gradient flow on first few epochs
                print(f"  Gradient norms by component:")
                for name, module in [("encoder", model.encoder), ("latent", model.latent), 
                                      ("decoder", model.decoder), ("conditioner", model.conditioner)]:
                    if module is not None:
                        grad_norms = []
                        for p in module.parameters():
                            if p.grad is not None:
                                grad_norms.append(p.grad.norm().item())
                        if grad_norms:
                            print(f"    {name}: mean={np.mean(grad_norms):.2e}, max={np.max(grad_norms):.2e}")
    
    return pred, target, loss.item(), rel_l1.item(), rel_l2.item()


def main():
    print("="*70)
    print("DEEP DEBUGGING - SINGLE SAMPLE OVERFITTING")
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
        stage_name="deep_debug",
        stage_id="deep_debug",
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
    
    # Create small model for debugging
    print("Creating model...")
    input_shape = dataset.getshape_x()
    output_shape = dataset.getshape_target()
    
    model = model_from_kwargs(
        kind="cfd_simformer_model",
        conditioner=dict(
            kind="conditioners.timestep_velocity_conditioner_pdearena",
            kwargs=dict(dim=128),
        ),
        encoder=dict(
            kind="encoders.cfd_pool_transformer_perceiver",
            num_latent_tokens=64,  # Small for debugging
            enc_depth=1,  # Single layer for debugging
            kwargs=dict(gnn_dim=64, enc_dim=64, perc_dim=128, enc_num_attn_heads=2, perc_num_attn_heads=2),
        ),
        latent=dict(
            kind="latent.transformer_model",
            depth=1,  # Single layer for debugging
            kwargs=dict(dim=128, num_attn_heads=2),
        ),
        decoder=dict(
            kind="decoders.cfd_transformer_perceiver",
            depth=1,  # Single layer for debugging
            use_last_norm=False,  # Try without final norm
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Manual training
    final_pred, target, final_loss, final_rel_l1, final_rel_l2 = manual_training_loop(
        model, dataset, device, num_epochs=100, lr=5e-3  # Higher LR
    )
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Final Rel L1: {final_rel_l1:.6f} ({final_rel_l1*100:.2f}%)")
    print(f"Final Rel L2: {final_rel_l2:.6f} ({final_rel_l2*100:.2f}%)")
    
    if final_rel_l1 > 0.01:
        print("\n⚠ WARNING: Overfitting failed! Rel L1 should be < 1%")
        print("Investigating further...")
        
        # Check if target has any special structure
        print(f"\nTarget analysis:")
        print(f"  Number of unique values: {len(torch.unique(target))}")
        print(f"  Zero elements: {(target == 0).sum().item()} / {target.numel()}")
        print(f"  Near-zero (< 0.01): {(target.abs() < 0.01).sum().item()} / {target.numel()}")
    else:
        print("\n✓ SUCCESS: Achieved proper overfitting!")


if __name__ == "__main__":
    main()



