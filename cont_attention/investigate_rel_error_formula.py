#!/usr/bin/env python3
"""
Investigate why even perfect memorization has high rel_l1.
Check if the formula or data has issues.
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


def investigate():
    print("="*70)
    print("INVESTIGATING RELATIVE ERROR FORMULA")
    print("="*70)
    
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
        stage_name="investigate",
        stage_id="investigate",
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
    
    # Get target
    target = dataset.getitem_target(0)
    
    print(f"\nTarget shape: {target.shape}")
    print(f"Target dtype: {target.dtype}")
    
    # Create perfect prediction (should give rel_l1 = 0)
    pred_perfect = target.clone()
    
    # Compute rel_l1 with perfect prediction
    rel_l1_perfect = (pred_perfect - target).abs().sum() / (target.abs().sum() + 1e-12)
    print(f"\nPerfect prediction rel_l1: {rel_l1_perfect.item():.15f}")
    
    if rel_l1_perfect.item() > 1e-10:
        print("⚠ WARNING: Perfect prediction doesn't give rel_l1 ≈ 0!")
        print("  Something is wrong with the formula or data!")
    else:
        print("✓ Perfect prediction gives rel_l1 ≈ 0 (as expected)")
    
    # Check if target has special structure
    print(f"\nTarget statistics:")
    print(f"  Mean: {target.mean():.6f}")
    print(f"  Std: {target.std():.6f}")
    print(f"  Min: {target.min():.6f}")
    print(f"  Max: {target.max():.6f}")
    print(f"  Abs sum: {target.abs().sum():.6f}")
    
    # Check per-channel
    print(f"\nPer-channel analysis:")
    for ch in range(target.shape[1]):
        t_ch = target[:, ch]
        print(f"  Channel {ch}:")
        print(f"    Mean: {t_ch.mean():.6f}, Std: {t_ch.std():.6f}")
        print(f"    Min: {t_ch.min():.6f}, Max: {t_ch.max():.6f}")
        print(f"    Abs sum: {t_ch.abs().sum():.6f}")
        print(f"    Num positive: {(t_ch > 0).sum()}, Num negative: {(t_ch < 0).sum()}")
        
        # Check if channel has sign-balanced values
        pos_sum = t_ch[t_ch > 0].sum().item()
        neg_sum = t_ch[t_ch < 0].abs().sum().item()
        print(f"    Positive sum: {pos_sum:.6f}, Negative sum: {neg_sum:.6f}")
        if abs(pos_sum - neg_sum) / (pos_sum + neg_sum + 1e-12) > 0.5:
            print(f"    ⚠ Highly imbalanced! This makes abs().sum() small!")
    
    # Test with mean prediction
    pred_mean = torch.full_like(target, target.mean())
    rel_l1_mean = (pred_mean - target).abs().sum() / (target.abs().sum() + 1e-12)
    mse_mean = ((pred_mean - target) ** 2).mean()
    
    print(f"\nMean prediction (baseline):")
    print(f"  Rel L1: {rel_l1_mean.item():.6f}")
    print(f"  MSE: {mse_mean.item():.6f}")
    
    # Test if sign-cancellation is the issue
    print(f"\nChecking for sign cancellation issue:")
    total_abs_sum = target.abs().sum().item()
    total_sum = target.sum().item()
    print(f"  abs().sum(): {total_abs_sum:.6f}")
    print(f"  sum(): {total_sum:.6f}")
    print(f"  Ratio: {abs(total_sum) / total_abs_sum:.6f}")
    
    if abs(total_sum) / total_abs_sum < 0.1:
        print("  ⚠ Values cancel out! abs().sum() is misleading for rel_l1!")
        print("  → Using abs().sum() in denominator is problematic")
        print("  → Should use RMS or L2 norm instead")
        
        # Recompute with better formula
        print(f"\nAlternative formulas:")
        # Using RMS
        target_rms = torch.sqrt((target ** 2).mean())
        rel_l1_rms = (pred_perfect - target).abs().mean() / (target_rms + 1e-12)
        print(f"  Rel L1 (RMS norm): {rel_l1_rms.item():.15f}")
        
        # Using L2 norm
        rel_l1_l2norm = (pred_perfect - target).abs().sum() / (torch.linalg.vector_norm(target) + 1e-12)
        print(f"  Rel L1 (L2 norm): {rel_l1_l2norm.item():.15f}")


if __name__ == "__main__":
    investigate()



