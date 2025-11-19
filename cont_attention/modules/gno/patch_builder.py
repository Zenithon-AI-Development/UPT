"""
Patch Builder for FANO Continuum Attention

NEW MODULE - Added for continuum attention experiments

Converts flat mesh points to patched 2D grids for FANO encoder.
Assumes data is in row-major grid order (verified for zpinch and trl2d datasets).

Classes:
    GridPatchBuilder: Convert regular grid mesh → patches
"""

import torch
import torch.nn as nn


class GridPatchBuilder(nn.Module):
    """
    Convert flat mesh points to 2D patches for FANO attention.
    
    Assumes:
    - Data is on a regular H×W grid
    - Points are ordered row-major: (h w) via einops.rearrange
    - This is verified for zpinch (128×128) and trl2d datasets
    
    Pipeline:
        Flat mesh (N_total, C) with batch_idx
          ↓ Split by batch
        Per-sample flat (H×W, C)
          ↓ Reshape to grid (verified row-major ordering)
        Grid (H, W, C)
          ↓ Chop into patches
        Patches (num_patches_h, num_patches_w, patch_h, patch_w, C)
          ↓ Flatten patch grid
        Output (num_patches, patch_h, patch_w, C)
    
    Args:
        H: Grid height (e.g., 128 for zpinch)
        W: Grid width (e.g., 128 for zpinch)
        num_patches_h: Number of patches in height dimension (e.g., 8)
        num_patches_w: Number of patches in width dimension (e.g., 8)
    
    Example:
        For zpinch 128×128 grid with 8×8 patches:
        - Input: (16384, 28) flat points [batch_size=4 → 4×16384]
        - Output: (4, 64, 16, 16, 28) patches
        - 64 = 8×8 patches, each patch is 16×16
    """
    
    def __init__(self, H, W, num_patches_h, num_patches_w):
        super().__init__()
        self.H = H
        self.W = W
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        # Compute patch dimensions
        assert H % num_patches_h == 0, f"H ({H}) must be divisible by num_patches_h ({num_patches_h})"
        assert W % num_patches_w == 0, f"W ({W}) must be divisible by num_patches_w ({num_patches_w})"
        
        self.patch_h = H // num_patches_h
        self.patch_w = W // num_patches_w
        self.num_patches = num_patches_h * num_patches_w
        
    def forward(self, x, mesh_pos, batch_idx):
        """
        Convert flat mesh points to patches.
        
        Args:
            x: (N_total, C) - Flat mesh points across batch
                Where N_total = batch_size × H × W
            mesh_pos: (N_total, 2) - Positions (not used, but kept for API compatibility)
            batch_idx: (N_total,) - Batch index for each point
        
        Returns:
            patches: (batch_size, num_patches, patch_h, patch_w, C)
        
        Process per sample:
            1. Extract points for this batch: (H×W, C)
            2. Reshape to grid (row-major): (H, W, C)
            3. Chop to patches: (num_patches_h, patch_h, num_patches_w, patch_w, C)
            4. Rearrange: (num_patches_h, num_patches_w, patch_h, patch_w, C)
            5. Flatten patch grid: (num_patches, patch_h, patch_w, C)
        """
        batch_size = int(batch_idx.max().item()) + 1
        C = x.size(1)
        N_per_sample = self.H * self.W
        
        # Verify expected input size
        expected_total = batch_size * N_per_sample
        assert x.size(0) == expected_total, \
            f"Expected {expected_total} points ({batch_size} × {N_per_sample}), got {x.size(0)}"
        
        patches = []
        
        for b in range(batch_size):
            # Step 1: Extract points for this batch
            mask = batch_idx == b
            x_b = x[mask]  # (H×W, C)
            
            assert x_b.size(0) == N_per_sample, \
                f"Batch {b}: Expected {N_per_sample} points, got {x_b.size(0)}"
            
            # Step 2: Reshape flat points to grid
            # Data is row-major ordered from einops.rearrange("h w c -> (h w) c")
            # So we can directly reshape: (H×W, C) → (H, W, C)
            grid = x_b.reshape(self.H, self.W, C)
            
            # Step 3: Chop grid into patches
            # Reshape to: (num_patches_h, patch_h, num_patches_w, patch_w, C)
            chopped = grid.reshape(
                self.num_patches_h, self.patch_h,
                self.num_patches_w, self.patch_w,
                C
            )
            
            # Step 4: Rearrange to group patches together
            # From: (num_patches_h, patch_h, num_patches_w, patch_w, C)
            # To:   (num_patches_h, num_patches_w, patch_h, patch_w, C)
            chopped = chopped.permute(0, 2, 1, 3, 4)
            
            # Step 5: Flatten patch grid dimension
            # From: (num_patches_h, num_patches_w, patch_h, patch_w, C)
            # To:   (num_patches, patch_h, patch_w, C)
            chopped = chopped.reshape(self.num_patches, self.patch_h, self.patch_w, C)
            
            patches.append(chopped)
        
        # Stack batch dimension: (batch_size, num_patches, patch_h, patch_w, C)
        patches = torch.stack(patches, dim=0)
        
        return patches
    
    def __repr__(self):
        return (f"GridPatchBuilder(H={self.H}, W={self.W}, "
                f"patches={self.num_patches_h}×{self.num_patches_w}={self.num_patches}, "
                f"patch_size={self.patch_h}×{self.patch_w})")

