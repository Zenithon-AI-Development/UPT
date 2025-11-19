#!/usr/bin/env python3
"""
Test pure memorization with aggressive optimization to reach 0.1% error.
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


def test_memorization():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
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
        stage_name="memo_test",
        stage_id="memo_test",
        temp_path=static.temp_path,
    )
    
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
    
    target = dataset.getitem_target(0).to(device)
    
    print(f"Target shape: {target.shape}")
    print(f"Target stats: mean={target.mean():.6f}, std={target.std():.6f}")
    
    # Test 1: Direct copy (should be perfect immediately)
    print("\n" + "="*70)
    print("TEST 1: Direct assignment (sanity check)")
    print("="*70)
    
    lookup = target.clone()
    rel_l1 = (lookup - target).abs().sum() / (target.abs().sum() + 1e-12)
    print(f"Direct copy rel_l1: {rel_l1.item():.15e}")
    print(f"Should be ≈ 0: {rel_l1.item() < 1e-10}")
    
    # Test 2: Learnable parameter with high LR
    print("\n" + "="*70)
    print("TEST 2: Learnable parameter (high LR, many epochs)")
    print("="*70)
    
    # Initialize close to target for faster convergence
    lookup_param = nn.Parameter(target.clone() + torch.randn_like(target) * 0.1)
    
    # Try different optimizers and LRs
    for lr in [0.1, 0.5, 1.0]:
        print(f"\n--- Learning rate: {lr} ---")
        lookup_param.data = target.clone() + torch.randn_like(target) * 0.1
        optimizer = torch.optim.Adam([lookup_param], lr=lr)
        
        for epoch in range(1000):
            optimizer.zero_grad()
            loss = ((lookup_param - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            
            if epoch in [0, 1, 5, 10, 50, 100, 500, 999]:
                rel_l1 = (lookup_param - target).abs().sum() / (target.abs().sum() + 1e-12)
                rel_l2 = torch.linalg.vector_norm(lookup_param - target) / (torch.linalg.vector_norm(target) + 1e-12)
                max_diff = (lookup_param - target).abs().max()
                print(f"  Epoch {epoch:4d}: Loss={loss.item():.8e}, Rel L1={rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%), "
                      f"Max diff={max_diff.item():.6e}")
                
                if rel_l1.item() < 0.001:
                    print(f"  ✓ Reached target! (rel_l1 < 0.1%)")
                    break
    
    # Test 3: Check if there's numerical precision issue
    print("\n" + "="*70)
    print("TEST 3: Numerical precision check")
    print("="*70)
    
    # Create prediction with tiny random noise
    pred_tiny_noise = target + torch.randn_like(target) * 1e-6
    rel_l1_tiny = (pred_tiny_noise - target).abs().sum() / (target.abs().sum() + 1e-12)
    mse_tiny = ((pred_tiny_noise - target) ** 2).mean()
    
    print(f"Prediction with 1e-6 noise:")
    print(f"  MSE: {mse_tiny.item():.15e}")
    print(f"  Rel L1: {rel_l1_tiny.item():.15e}")
    
    # Check denormalization
    print("\n" + "="*70)
    print("TEST 4: Check if we need to denormalize")
    print("="*70)
    
    # Check if dataset has denormalization method
    if hasattr(dataset, 'well') and hasattr(dataset.well, 'norm'):
        print(f"Dataset has normalization object: {dataset.well.norm is not None}")
        if dataset.well.norm is not None:
            print("Normalization is active!")
            print(f"  Type: {type(dataset.well.norm)}")
            
            # Try to denormalize
            if hasattr(dataset.well.norm, 'denormalize'):
                print("\n  Attempting denormalization...")
                try:
                    # This might not work directly, but let's see
                    print("  (Denormalization would need field names and proper shape)")
                except Exception as e:
                    print(f"  Error: {e}")
    
    return rel_l1_perfect.item()


if __name__ == "__main__":
    test_memorization()



