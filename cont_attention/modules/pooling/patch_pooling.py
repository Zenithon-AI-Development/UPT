"""
Patch Pooling Mechanisms for FANO Encoder

After FANO blocks process patches with spatial structure (H×W), we need to
convert patches to tokens for the perceiver cross-attention stage.

Two main philosophies:
A) Direct Pooling: Pool patches directly to tokens (custom, not from FANO paper)
B) Grid Reconstruction: Reshape to full grid → sample → tokens (faithful to FANO)

Available strategies:
1. SpatialAvgPooling: Simple spatial average per patch (Direct)
2. LearnedPooling: Tiny MLP probe per patch (Direct)
3. AttentionPooling: Learnable weighted average (Direct)
4. GridUniformSampling: Reconstruct grid → uniform subsample (Faithful to FANO)
5. GridAdaptiveSampling: Reconstruct grid → learned sampling (Hybrid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAvgPooling(nn.Module):
    """
    Simple spatial average pooling per patch.
    
    Pools (H, W) spatial dimensions to a single vector per patch.
    This is the default choice - simple, fast, and effective.
    
    Args:
        None (stateless pooling)
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, patch_h, patch_w, d_model)
        Returns:
            (batch, num_patches, d_model)
        """
        # Average over spatial dimensions
        return x.mean(dim=(2, 3))


class LearnedPooling(nn.Module):
    """
    Learned pooling via a tiny probe MLP per patch.
    
    Uses a small MLP to aggregate spatial information, allowing the model
    to learn what spatial patterns are most important for each patch.
    
    Args:
        d_model: Model dimension
        hidden_dim: Hidden dimension of probe MLP (default: d_model * 2)
        dropout: Dropout probability
    """
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or (d_model * 2)
        
        # Tiny probe MLP: flatten spatial → hidden → output
        self.probe = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, patch_h, patch_w, d_model)
        Returns:
            (batch, num_patches, d_model)
        """
        batch, num_patches, patch_h, patch_w, d_model = x.shape
        
        # Flatten spatial dimensions: (batch, num_patches, patch_h * patch_w, d_model)
        x_flat = x.reshape(batch, num_patches, patch_h * patch_w, d_model)
        
        # Average pool first (for stability)
        x_avg = x_flat.mean(dim=2)  # (batch, num_patches, d_model)
        
        # Apply probe MLP
        return self.probe(x_avg)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling with learnable query vector.
    
    Uses a learnable query vector to compute attention weights over
    spatial locations, then performs weighted average.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads (default: 1)
        dropout: Dropout probability
    """
    def __init__(self, d_model, num_heads=1, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Learnable query vector for pooling
        self.query = nn.Parameter(torch.randn(1, 1, num_heads, self.d_k))
        
        # Key and Value projections
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Scale for attention
        self.scale = self.d_k ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, patch_h, patch_w, d_model)
        Returns:
            (batch, num_patches, d_model)
        """
        batch, num_patches, patch_h, patch_w, d_model = x.shape
        
        # Flatten spatial dimensions: (batch, num_patches, patch_h * patch_w, d_model)
        x_flat = x.reshape(batch, num_patches, patch_h * patch_w, d_model)
        
        # Project to keys and values
        # (batch, num_patches, spatial, num_heads, d_k)
        keys = self.key_proj(x_flat).reshape(batch, num_patches, -1, self.num_heads, self.d_k)
        values = self.value_proj(x_flat).reshape(batch, num_patches, -1, self.num_heads, self.d_k)
        
        # Expand query for batch and patches
        # (batch, num_patches, 1, num_heads, d_k)
        query = self.query.expand(batch, num_patches, -1, -1, -1)
        
        # Compute attention scores
        # (batch, num_patches, num_heads, 1, spatial)
        scores = torch.einsum('bpqhd,bpshd->bphs', query, keys) * self.scale
        scores = scores.transpose(2, 3)  # (batch, num_patches, 1, num_heads, spatial)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_patches, 1, num_heads, d_k)
        pooled = torch.einsum('bpqhs,bpshd->bpqhd', attn_weights, values)
        
        # Reshape and project: (batch, num_patches, d_model)
        pooled = pooled.squeeze(2).reshape(batch, num_patches, d_model)
        pooled = self.out_proj(pooled)
        
        return pooled


# ============================================================================
# GRID RECONSTRUCTION APPROACH (Faithful to FANO Paper)
# ============================================================================


class GridUniformSampling(nn.Module):
    """
    Grid reconstruction + uniform sampling (Option A1 - Faithful to FANO).
    
    Philosophy:
    1. Reshape patches back to full spatial grid (like original FANO)
    2. Uniformly subsample the grid to get tokens
    3. Feed sampled tokens to perceiver
    
    This is more faithful to FANO's design where the full field is reconstructed
    before any downstream processing.
    
    Args:
        H: Original grid height
        W: Original grid width
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
        num_samples: Number of points to sample (e.g., 64, 256)
                     If None, samples one point per patch (num_patches_h × num_patches_w)
    """
    def __init__(self, H, W, num_patches_h, num_patches_w, num_samples=None):
        super().__init__()
        self.H = H
        self.W = W
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.patch_h = H // num_patches_h
        self.patch_w = W // num_patches_w
        
        # Default: sample one point per patch
        if num_samples is None:
            num_samples = num_patches_h * num_patches_w
        self.num_samples = num_samples
        
        # Compute sampling stride
        # Want roughly sqrt(num_samples) points in each dimension
        samples_per_dim = int(num_samples ** 0.5)
        self.stride_h = max(1, H // samples_per_dim)
        self.stride_w = max(1, W // samples_per_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, patch_h, patch_w, d_model)
        Returns:
            (batch, num_samples, d_model)
        """
        batch, num_patches, patch_h, patch_w, d_model = x.shape
        
        # Step 1: Reshape patches back to full grid (FANO-style)
        # (batch, num_patches_h, num_patches_w, patch_h, patch_w, d_model)
        x = x.reshape(batch, self.num_patches_h, self.num_patches_w, patch_h, patch_w, d_model)
        
        # Rearrange to full grid: (batch, H, W, d_model)
        # Interleave patches: (batch, num_patches_h, patch_h, num_patches_w, patch_w, d_model)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(batch, self.H, self.W, d_model)
        
        # Step 2: Uniform grid subsampling
        # Sample every stride_h-th row and stride_w-th column
        x_sampled = x[:, ::self.stride_h, ::self.stride_w, :]
        
        # Flatten spatial dimensions: (batch, num_sampled_points, d_model)
        x_sampled = x_sampled.reshape(batch, -1, d_model)
        
        # If we got more/fewer samples than expected, adjust
        actual_samples = x_sampled.size(1)
        if actual_samples > self.num_samples:
            # Take first num_samples
            x_sampled = x_sampled[:, :self.num_samples, :]
        elif actual_samples < self.num_samples:
            # This shouldn't happen with correct stride calculation
            # But if it does, pad with zeros
            padding = torch.zeros(batch, self.num_samples - actual_samples, d_model, 
                                device=x_sampled.device, dtype=x_sampled.dtype)
            x_sampled = torch.cat([x_sampled, padding], dim=1)
        
        return x_sampled


class GridAdaptiveSampling(nn.Module):
    """
    Grid reconstruction + learned adaptive sampling (Hybrid approach).
    
    Philosophy:
    1. Reshape patches back to full spatial grid
    2. Learn importance scores for each grid point
    3. Sample top-K most important points
    4. Feed sampled tokens to perceiver
    
    This allows the model to learn which spatial locations are most informative.
    
    Args:
        H: Original grid height
        W: Original grid width
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
        num_samples: Number of points to sample
        d_model: Model dimension
    """
    def __init__(self, H, W, num_patches_h, num_patches_w, num_samples, d_model):
        super().__init__()
        self.H = H
        self.W = W
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.patch_h = H // num_patches_h
        self.patch_w = W // num_patches_w
        self.num_samples = num_samples
        
        # Learnable importance scoring
        self.importance_net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, patch_h, patch_w, d_model)
        Returns:
            (batch, num_samples, d_model)
        """
        batch, num_patches, patch_h, patch_w, d_model = x.shape
        
        # Step 1: Reshape patches back to full grid
        x = x.reshape(batch, self.num_patches_h, self.num_patches_w, patch_h, patch_w, d_model)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(batch, self.H, self.W, d_model)
        
        # Step 2: Compute importance scores for each grid point
        # (batch, H, W, 1)
        importance = self.importance_net(x)
        importance = importance.squeeze(-1)  # (batch, H, W)
        
        # Step 3: Top-K sampling based on importance
        # Flatten spatial: (batch, H*W)
        importance_flat = importance.reshape(batch, -1)
        x_flat = x.reshape(batch, -1, d_model)
        
        # Get top-K indices
        top_k_values, top_k_indices = torch.topk(importance_flat, self.num_samples, dim=1)
        
        # Gather top-K points: (batch, num_samples, d_model)
        x_sampled = torch.gather(
            x_flat, 
            1, 
            top_k_indices.unsqueeze(-1).expand(-1, -1, d_model)
        )
        
        return x_sampled

