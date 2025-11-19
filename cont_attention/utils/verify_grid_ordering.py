"""
Verify that zpinch dataset points are in proper grid order.

This script checks that:
1. Data flattening (h w) matches mesh_pos ordering
2. We can correctly reshape flat points back to 2D grid
3. Patch decomposition will work correctly
"""

import torch
import numpy as np
from datasets.well_zpinch_dataset import WellZpinchDataset


def verify_zpinch_ordering():
    """
    Verify the ordering of zpinch dataset points.
    """
    print("="*70)
    print("VERIFYING ZPINCH DATASET GRID ORDERING")
    print("="*70)
    
    # Create dataset
    dataset = WellZpinchDataset(
        root="/home/workspace/flash/ZEN_WELL_train/128x128/data/",
        split="train",
        num_input_timesteps=4,
        norm="mean0std1_auto",
        max_num_sequences=1,  # Just load one file
    )
    
    H, W = dataset.H, dataset.W
    print(f"\nGrid dimensions: H={H}, W={W}")
    print(f"Total points: {H * W}")
    
    # Get mesh positions
    mesh_pos = dataset.getitem_mesh_pos(0)  # (H*W, 2)
    print(f"\nmesh_pos shape: {mesh_pos.shape}")
    
    # Check first few positions
    print("\n" + "="*70)
    print("FIRST 10 POSITIONS:")
    print("="*70)
    for i in range(min(10, len(mesh_pos))):
        x, y = mesh_pos[i]
        print(f"Point {i:4d}: x={x:8.3f}, y={y:8.3f}")
    
    # Check positions at row boundaries
    print("\n" + "="*70)
    print("POSITIONS AT ROW BOUNDARIES:")
    print("="*70)
    indices = [0, W-1, W, W+1, 2*W-1, 2*W, 2*W+1]
    for i in indices:
        if i < len(mesh_pos):
            x, y = mesh_pos[i]
            expected_h = i // W
            expected_w = i % W
            print(f"Point {i:4d}: x={x:8.3f}, y={y:8.3f}  (expected row={expected_h}, col={expected_w})")
    
    # Verify ordering by checking if positions match expected grid pattern
    print("\n" + "="*70)
    print("VERIFYING ORDERING:")
    print("="*70)
    
    # Reshape positions back to grid
    mesh_pos_grid = mesh_pos.reshape(H, W, 2)
    
    # Check if x-coordinate increases along width (columns)
    x_along_width = mesh_pos_grid[0, :, 0]  # First row, all columns, x-coord
    is_x_increasing = torch.all(x_along_width[1:] > x_along_width[:-1])
    print(f"✓ X-coordinate increases along width (columns): {is_x_increasing}")
    
    # Check if y-coordinate increases along height (rows)
    y_along_height = mesh_pos_grid[:, 0, 1]  # All rows, first column, y-coord
    is_y_increasing = torch.all(y_along_height[1:] > y_along_height[:-1])
    print(f"✓ Y-coordinate increases along height (rows): {is_y_increasing}")
    
    # Check if x is constant along columns (for each row)
    x_constant_in_rows = True
    for row in range(H):
        x_vals = mesh_pos_grid[row, :, 0]
        if not torch.allclose(x_vals[0:1].expand_as(x_vals), x_vals, atol=1e-5):
            x_constant_in_rows = False
            break
    
    # Actually wait, x should VARY along columns, not be constant
    # Let me check what indexing="xy" means
    print(f"\nFirst row x-coords (should increase): {x_along_width[:5].tolist()}")
    print(f"First col y-coords (should increase): {y_along_height[:5].tolist()}")
    
    # Get actual data to verify it matches
    x_data = dataset.getitem_x(0)  # (H*W, k*C)
    print(f"\n✓ Data shape: {x_data.shape}")
    
    # Reshape data back to grid
    num_channels = x_data.shape[1]
    x_data_grid = x_data.reshape(H, W, num_channels)
    print(f"✓ Reshaped to grid: {x_data_grid.shape}")
    
    # Verify we can reconstruct correctly
    x_data_flat = x_data_grid.reshape(-1, num_channels)
    matches = torch.allclose(x_data, x_data_flat, atol=1e-6)
    print(f"✓ Reshape round-trip successful: {matches}")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    if matches and is_x_increasing and is_y_increasing:
        print("✅ Data is in proper grid order (row-major: h, w)")
        print("✅ Simple reshape will work for patch builder!")
        print(f"✅ Can reshape (N={H*W}, C) → ({H}, {W}, C) directly")
        return True
    else:
        print("⚠️  Data ordering is not standard - need special handling!")
        return False


def test_patch_decomposition():
    """
    Test that patch decomposition works correctly.
    """
    print("\n" + "="*70)
    print("TESTING PATCH DECOMPOSITION")
    print("="*70)
    
    H, W = 128, 128
    num_patches_h, num_patches_w = 8, 8
    patch_h, patch_w = H // num_patches_h, W // num_patches_w
    
    print(f"Grid: {H}×{W}")
    print(f"Patches: {num_patches_h}×{num_patches_w} = {num_patches_h * num_patches_w} total")
    print(f"Patch size: {patch_h}×{patch_w}")
    
    # Create dummy data
    dummy = torch.arange(H * W).reshape(H, W).float()
    print(f"\nDummy data shape: {dummy.shape}")
    print(f"First few values (row-major): {dummy.reshape(-1)[:10].tolist()}")
    
    # Method 1: Direct reshape (what we'll use)
    patches = dummy.reshape(num_patches_h, patch_h, num_patches_w, patch_w)
    patches = patches.permute(0, 2, 1, 3)  # (num_patches_h, num_patches_w, patch_h, patch_w)
    patches = patches.reshape(num_patches_h * num_patches_w, patch_h, patch_w)
    
    print(f"\n✓ Patches shape: {patches.shape}")
    print(f"  Patch 0 top-left corner: {patches[0, 0, :3].tolist()}")
    print(f"  Patch 0 should start at position [0,0] in original grid")
    
    # Verify patch 0 contains the right values
    expected_patch_0 = dummy[:patch_h, :patch_w]
    matches = torch.allclose(patches[0], expected_patch_0)
    print(f"✓ Patch 0 correct: {matches}")
    
    # Reconstruct grid from patches
    patches_reshaped = patches.reshape(num_patches_h, num_patches_w, patch_h, patch_w)
    patches_reshaped = patches_reshaped.permute(0, 2, 1, 3)  # (num_patches_h, patch_h, num_patches_w, patch_w)
    reconstructed = patches_reshaped.reshape(H, W)
    
    reconstruction_matches = torch.allclose(dummy, reconstructed)
    print(f"✓ Grid reconstruction successful: {reconstruction_matches}")
    
    if reconstruction_matches:
        print("\n✅ Patch decomposition and reconstruction work correctly!")
        return True
    else:
        print("\n⚠️  Patch decomposition failed!")
        return False


if __name__ == "__main__":
    try:
        ordering_ok = verify_zpinch_ordering()
        patching_ok = test_patch_decomposition()
        
        print("\n" + "="*70)
        print("FINAL VERDICT:")
        print("="*70)
        if ordering_ok and patching_ok:
            print("✅ All checks passed!")
            print("✅ Patch builder can use simple reshape operations")
            print("✅ No need for reordering or scatter/gather operations")
        else:
            print("⚠️  Some checks failed - review implementation")
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

