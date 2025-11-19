"""
Quick script to verify TRL2D grid dimensions for FANO config.

Checks the actual H×W dimensions of TRL2D data to ensure
YAML configs have correct grid_h and grid_w values.
"""

from datasets.well_trl2d_dataset import WellTrl2dDataset

# Create dataset
dataset = WellTrl2dDataset(
    well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
    split="train",
    num_input_timesteps=4,
    norm="mean0std1",
    max_num_sequences=1,  # Just load one for checking
)

print("="*70)
print("TRL2D DATASET DIMENSIONS")
print("="*70)
print(f"Grid Height (H): {dataset.H}")
print(f"Grid Width (W):  {dataset.W}")
print(f"Total points:    {dataset.H * dataset.W}")
print(f"Num channels (F): {dataset.F}")
print()

# Suggest patch configurations
print("="*70)
print("SUGGESTED PATCH CONFIGURATIONS:")
print("="*70)

for num_patches_h, num_patches_w in [(4, 4), (8, 8), (16, 16)]:
    if dataset.H % num_patches_h == 0 and dataset.W % num_patches_w == 0:
        patch_h = dataset.H // num_patches_h
        patch_w = dataset.W // num_patches_w
        num_patches = num_patches_h * num_patches_w
        print(f"✓ {num_patches_h}×{num_patches_w} patches = {num_patches} total, "
              f"each {patch_h}×{patch_w}")
    else:
        print(f"✗ {num_patches_h}×{num_patches_w} - dimensions not divisible")

print()
print("="*70)
print("UPDATE YOUR YAML CONFIG:")
print("="*70)
print(f"grid_h: {dataset.H}")
print(f"grid_w: {dataset.W}")
print(f"num_patches_h: 8  # Adjust based on suggestions above")
print(f"num_patches_w: 8")
print("="*70)

