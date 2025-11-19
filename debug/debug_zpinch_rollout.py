#!/usr/bin/env python3
"""Debug script to verify zpinch rollout predictions match training"""
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from pathlib import Path
import yaml

# Load config
stage_id = 'k9931ybq'
config_path = Path(f"benchmarking/save/stage1/{stage_id}/hp_resolved.yaml")
with open(config_path) as f:
    hp = yaml.safe_load(f)

# Get normalization stats
train_config = hp["datasets"]["train"]
mean = np.array(train_config["mean"])
std = np.array(train_config["std"])

print("=== ZPinch Training Config ===")
print(f"num_input_timesteps: {train_config['num_input_timesteps']}")
print(f"Channels: {len(mean)}")
print(f"Mean: {mean}")
print(f"Std: {std}")
print()

# Load one sample through the dataset
from datasets import dataset_from_kwargs
from kappadata.wrappers import ModeWrapper

ds_cfg = train_config.copy()
ds_cfg.pop('collators', None)
ds_cfg.pop('dataset_wrappers', None)
dataset = dataset_from_kwargs(**ds_cfg)
dataset = ModeWrapper(dataset=dataset, mode='x target')

print(f"Dataset length: {len(dataset)}")
sample = dataset[0]
x, target = sample

print(f"\nSample 0 (from dataset):")
print(f"  x shape: {x.shape}")  # Should be (N, 4*C) for 4 input timesteps
print(f"  target shape: {target.shape}")  # Should be (N, C)
print(f"  x range (normalized): [{x.min():.3f}, {x.max():.3f}]")
print(f"  target range (normalized): [{target.min():.3f}, {target.max():.3f}]")
print()

# Denormalize and check physical ranges
N, input_dim = x.shape
C = len(mean)
num_input_timesteps = input_dim // C

# Reshape x to (N, num_input_timesteps, C)
x_reshaped = x.reshape(N, num_input_timesteps, C)

# Denormalize last input timestep
x_last_norm = x_reshaped[:, -1, :]  # (N, C)
x_last_phys = x_last_norm * torch.tensor(std) + torch.tensor(mean)

# Denormalize target
target_phys = target * torch.tensor(std) + torch.tensor(mean)

print("Physical units (after denormalization):")
print(f"  Last input timestep:")
for i, name in enumerate(['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']):
    print(f"    {name}: [{x_last_phys[:, i].min():.3e}, {x_last_phys[:, i].max():.3e}]")

print(f"\n  Target:")
for i, name in enumerate(['Density', 'Pressure', 'Vel-X', 'Vel-Y', 'Mag-Z', 'Mag-Phi', 'Current']):
    print(f"    {name}: [{target_phys[:, i].min():.3e}, {target_phys[:, i].max():.3e}]")

print("\n✓ If physical ranges look reasonable (e.g., density ~1, pressure ~1e14, etc.),")
print("  then the normalization/denormalization is working correctly.")
print("  If they're wildly off, there's a bug in the normalization pipeline.")


