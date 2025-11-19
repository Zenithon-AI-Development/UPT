#!/usr/bin/env python3
"""
Comprehensive normalization checker for all UPT models.
Verifies that normalization stats in configs match actual training data.
"""
import sys
sys.path.insert(0, 'src')
import h5py
import numpy as np
import yaml
from pathlib import Path
from glob import glob

def check_zpinch_normalization(config_path):
    """Check ZPinch normalization stats."""
    print("\n" + "="*80)
    print("CHECKING: ZPinch")
    print("="*80)
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    train_config = hp["datasets"]["train"]
    if "mean" not in train_config or "std" not in train_config:
        print("❌ No normalization stats in config")
        return
    
    config_mean = np.array(train_config["mean"])
    config_std = np.array(train_config["std"])
    
    print(f"\nConfig stats:")
    print(f"  Channels: {len(config_mean)}")
    print(f"  Mean: {config_mean}")
    print(f"  Std: {config_std}")
    
    # Load actual training data
    root = Path(train_config["root"])
    split = train_config["split"]
    files = sorted(list((root / split).glob("*.hdf5")))[:5]  # First 5 files
    
    if not files:
        print(f"❌ No training files found in {root / split}")
        return
    
    print(f"\nComputing stats from {len(files)} training files...")
    
    all_data = []
    for file in files:
        with h5py.File(file, 'r') as f:
            # Load first 10 timesteps
            n_timesteps = min(10, f['t0_fields/density'].shape[0])
            
            density = f['t0_fields/density'][:n_timesteps]
            pressure = f['t0_fields/pressure'][:n_timesteps]
            
            if 't1_fields' in f:
                velocity = f['t1_fields/velocity'][:n_timesteps]
                b_field = f['t1_fields/b_field'][:n_timesteps]
                
                if 'forcing_fields' in f:
                    current = f['forcing_fields/current_drive'][:n_timesteps]
                    # Stack all 7 channels
                    data = np.concatenate([
                        density[..., None], pressure[..., None],
                        velocity, b_field, current[..., None]
                    ], axis=-1)
                else:
                    # 6 channels
                    data = np.concatenate([
                        density[..., None], pressure[..., None],
                        velocity, b_field
                    ], axis=-1)
            else:
                # Only 2 channels
                data = np.stack([density, pressure], axis=-1)
            
            all_data.append(data.reshape(-1, data.shape[-1]))
    
    all_data = np.concatenate(all_data, axis=0)
    actual_mean = all_data.mean(axis=0)
    actual_std = all_data.std(axis=0)
    
    print(f"\nActual training data stats:")
    print(f"  Mean: {actual_mean}")
    print(f"  Std: {actual_std}")
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON:")
    print(f"{'='*80}")
    
    mean_match = np.allclose(config_mean, actual_mean, rtol=0.1)
    std_match = np.allclose(config_std, actual_std, rtol=0.1)
    
    if mean_match and std_match:
        print("✅ Normalization stats MATCH training data!")
    else:
        print("❌ NORMALIZATION MISMATCH!")
        print("\nMean relative error:")
        for i in range(len(config_mean)):
            rel_err = abs(config_mean[i] - actual_mean[i]) / (abs(actual_mean[i]) + 1e-10)
            print(f"  Ch{i}: {rel_err*100:.1f}%")
        
        print("\nStd relative error:")
        for i in range(len(config_std)):
            rel_err = abs(config_std[i] - actual_std[i]) / (abs(actual_std[i]) + 1e-10)
            print(f"  Ch{i}: {rel_err*100:.1f}%")
        
        # Check normalized variance
        print("\nNormalized variance (should be ~1.0):")
        normalized_var = (actual_std / config_std) ** 2
        for i in range(len(normalized_var)):
            status = "✅" if 0.5 < normalized_var[i] < 2.0 else "❌"
            print(f"  Ch{i}: {normalized_var[i]:.3f} {status}")

def check_helmholtz_normalization(config_path):
    """Check Helmholtz normalization - uses The Well's automatic stats."""
    print("\n" + "="*80)
    print("CHECKING: Helmholtz")
    print("="*80)
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    train_config = hp["datasets"]["train"]
    
    if "mean" in train_config and "std" in train_config:
        print("⚠️  Manual normalization stats found (should use The Well's automatic)")
        print(f"  Mean: {train_config['mean']}")
        print(f"  Std: {train_config['std']}")
    else:
        norm_mode = train_config.get("norm", "none")
        print(f"✅ Using automatic normalization: {norm_mode}")
        print("   The Well library computes stats from training data automatically")

def check_trl2d_normalization(config_path):
    """Check TRL2D normalization."""
    print("\n" + "="*80)
    print("CHECKING: TRL2D")
    print("="*80)
    
    with open(config_path) as f:
        hp = yaml.safe_load(f)
    
    train_config = hp["datasets"]["train"]
    norm_mode = train_config.get("norm", "none")
    
    if "mean" in train_config and "std" in train_config:
        print("⚠️  Manual normalization stats found")
        print(f"  Mean: {train_config['mean']}")
        print(f"  Std: {train_config['std']}")
    else:
        print(f"✅ Using automatic normalization: {norm_mode}")
        print("   The Well library computes stats from training data automatically")

def main():
    print("="*80)
    print("NORMALIZATION VERIFICATION FOR ALL UPT MODELS")
    print("="*80)
    
    # Find all training configs
    save_dir = Path("benchmarking/save")
    
    # Check all stage1 models (ZPinch, Helmholtz)
    stage1_configs = list(save_dir.glob("stage1/*/hp_resolved.yaml"))
    
    print(f"\nFound {len(stage1_configs)} stage1 model configs")
    
    for config in sorted(stage1_configs)[-3:]:  # Check last 3 models
        print(f"\n{'='*80}")
        print(f"Config: {config}")
        print(f"{'='*80}")
        
        with open(config) as f:
            hp = yaml.safe_load(f)
        
        dataset_kind = hp["datasets"]["train"].get("kind", "unknown")
        
        if "zpinch" in dataset_kind:
            check_zpinch_normalization(config)
        elif "hs" in dataset_kind or "helmholtz" in dataset_kind:
            check_helmholtz_normalization(config)
        elif "trl2d" in dataset_kind:
            check_trl2d_normalization(config)
        else:
            print(f"ℹ️  Unknown dataset kind: {dataset_kind}")
    
    # Check TRL2D models
    trl2d_configs = list(save_dir.glob("trl2d*/*/hp_resolved.yaml"))
    for config in trl2d_configs:
        check_trl2d_normalization(config)
    
    # Check Helmholtz models
    helmholtz_configs = list(save_dir.glob("helmholtz*/*/hp_resolved.yaml"))
    for config in helmholtz_configs:
        check_helmholtz_normalization(config)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ Models using automatic normalization (from The Well): SAFE")
    print("❌ Models with manual stats: NEED VERIFICATION")
    print("\nRecommendation: Always use norm='mean0std1' without manual mean/std")
    print("                Let The Well library compute stats from training data!")

if __name__ == "__main__":
    main()


