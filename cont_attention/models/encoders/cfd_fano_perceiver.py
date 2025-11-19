"""
FANO Perceiver Encoder for Continuum Attention

NEW MODEL - Added for continuum attention experiments
Based on "Continuum Attention for Neural Operators" (Calvello et al., 2024)

Architecture:
    Flat mesh → GridPatchBuilder → Patches (B, P, H, W, C)
      → input_proj → (B, P, H, W, d_model)
      → FANO blocks → (B, P, H, W, d_model)
      → Pooling → (B, K, d_model)  [K = num tokens for perceiver]
      → Perceiver cross-attention → (B, num_latent_tokens, perc_dim)

Key features:
- Preserves 2D spatial structure via patches (unlike graph pooling)
- Uses Fourier spectral convolutions for Q/K/V (smooth, continuous)
- Two pooling strategies: direct (simple) or grid reconstruction (FANO-faithful)
- Fully configurable via YAML
"""

from functools import partial

import torch
from torch import nn
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, DitPerceiverPoolingBlock

from models.base.single_model_base import SingleModelBase
from modules.gno.patch_builder import GridPatchBuilder
from modules.attention.fano_attention import TransformerEncoderLayer_Conv, TransformerEncoder_Operator
from modules.pooling.patch_pooling import (
    SpatialAvgPooling,
    LearnedPooling,
    AttentionPooling,
    GridUniformSampling,
    GridAdaptiveSampling,
)
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class CfdFanoPerceiver(SingleModelBase):
    """
    FANO-based encoder with perceiver pooling for continuum attention.
    
    Replaces UPT's graph pooling + standard transformer with:
    - Patch-based representation (preserves 2D structure)
    - FANO continuum attention (Fourier Q/K/V + spatial attention)
    - Flexible pooling strategies
    
    Args:
        input_dim: Input feature dimension (e.g., 28 = 4 timesteps × 7 channels)
        d_model: Model dimension for FANO blocks (e.g., 192)
        num_fano_layers: Number of FANO encoder layers (e.g., 2 or 4)
        num_heads: Number of attention heads in FANO (e.g., 6)
        perc_dim: Perceiver dimension (e.g., 192)
        perc_num_heads: Perceiver attention heads (e.g., 3)
        num_latent_tokens: Number of latent tokens from perceiver (e.g., 256)
        H: Grid height (e.g., 128 for zpinch)
        W: Grid width (e.g., 128 for zpinch)
        num_patches_h: Number of patches in height (e.g., 8)
        num_patches_w: Number of patches in width (e.g., 8)
        fourier_modes: Fourier modes per dimension (e.g., 9 = patch_size//2 + 1)
        pooling_type: 'spatial_avg', 'learned', 'attention', 'grid_uniform', 'grid_adaptive'
        num_grid_samples: For grid_uniform/grid_adaptive - how many points to sample
        dropout: Dropout probability
        activation: Activation function for FANO ('relu' or 'gelu')
        dim_feedforward: FFN dimension in FANO (e.g., 768 = 4×d_model)
        init_weights: Weight initialization scheme
    """
    
    def __init__(
        self,
        input_dim,
        d_model,
        num_fano_layers,
        num_heads,
        perc_dim,
        perc_num_heads,
        num_latent_tokens,
        H,
        W,
        num_patches_h,
        num_patches_w,
        fourier_modes,
        pooling_type="spatial_avg",
        num_grid_samples=None,
        dropout=0.1,
        activation="gelu",
        dim_feedforward=None,
        init_weights="xavier_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Store configuration
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_fano_layers = num_fano_layers
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.H = H
        self.W = W
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.fourier_modes = fourier_modes
        self.pooling_type = pooling_type
        self.num_grid_samples = num_grid_samples
        self.dropout = dropout
        self.activation = activation
        self.init_weights = init_weights

        # FANO encoder operates on grid patches and does not require mesh edges
        self.requires_mesh_edges = False
        
        dim_feedforward = dim_feedforward or (d_model * 4)
        self.dim_feedforward = dim_feedforward
        
        # Computed properties
        patch_h = H // num_patches_h
        patch_w = W // num_patches_w
        num_patches = num_patches_h * num_patches_w
        
        # ================================================================
        # STEP 1: Patch Builder - Convert flat mesh to 2D patches
        # ================================================================
        self.patch_builder = GridPatchBuilder(
            H=H,
            W=W,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        
        # ================================================================
        # STEP 2: Input Projection - Lift input_dim → d_model
        # ================================================================
        # Pointwise projection over channels (like 1×1 conv)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # ================================================================
        # STEP 3: FANO Blocks - Continuum attention on patches
        # ================================================================
        # Create one FANO layer template
        fano_layer = TransformerEncoderLayer_Conv(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            activation=activation,
            norm_first=True,  # Pre-norm for stability
            do_layer_norm=True,
            dim_feedforward=dim_feedforward,
            modes=(fourier_modes, fourier_modes),  # Fourier modes for Q/K/V
            patch_size=patch_h,  # Used for auto-computing modes if needed
            im_size=patch_h,     # Scale factor for continuum attention
            batch_first=True,
        )
        
        # Stack multiple FANO layers
        self.fano_encoder = TransformerEncoder_Operator(fano_layer, num_fano_layers)
        
        # ================================================================
        # STEP 4: Pooling - Convert patches to tokens
        # ================================================================
        # Create pooling module based on strategy
        if pooling_type == "spatial_avg":
            # Direct pooling: Average each patch (B, P, H, W, d) → (B, P, d)
            self.pooling = SpatialAvgPooling()
            num_tokens_for_perceiver = num_patches
            
        elif pooling_type == "learned":
            # Direct pooling: Learned probe MLP per patch
            self.pooling = LearnedPooling(d_model=d_model, dropout=dropout)
            num_tokens_for_perceiver = num_patches
            
        elif pooling_type == "attention":
            # Direct pooling: Attention-based pooling per patch
            self.pooling = AttentionPooling(d_model=d_model, num_heads=1, dropout=dropout)
            num_tokens_for_perceiver = num_patches
            
        elif pooling_type == "grid_uniform":
            # Grid reconstruction: Reshape to full grid → uniform subsample
            # This is faithful to FANO paper design
            num_grid_samples = num_grid_samples or num_patches  # Default: one sample per patch
            self.pooling = GridUniformSampling(
                H=H, W=W,
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w,
                num_samples=num_grid_samples,
            )
            num_tokens_for_perceiver = num_grid_samples
            
        elif pooling_type == "grid_adaptive":
            # Grid reconstruction: Reshape to full grid → learned top-K sampling
            assert num_grid_samples is not None, "grid_adaptive requires num_grid_samples"
            self.pooling = GridAdaptiveSampling(
                H=H, W=W,
                num_patches_h=num_patches_h,
                num_patches_w=num_patches_w,
                num_samples=num_grid_samples,
                d_model=d_model,
            )
            num_tokens_for_perceiver = num_grid_samples
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        
        # ================================================================
        # STEP 5: Perceiver Cross-Attention - Compress to latent
        # ================================================================
        # Project pooled tokens to perceiver dimension if needed
        if d_model != perc_dim:
            self.perc_proj = LinearProjection(d_model, perc_dim, init_weights=init_weights)
        else:
            self.perc_proj = nn.Identity()
        
        # Perceiver pooling block (cross-attention)
        # Q: num_latent_tokens learned queries
        # K,V: num_tokens_for_perceiver from pooling
        if "condition_dim" in self.static_ctx:
            # Conditioned perceiver (timestep/velocity conditioning via AdaLN)
            perceiver_block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.static_ctx["condition_dim"],
                    init_weights=init_weights,
                ),
            )
        else:
            # Standard perceiver
            perceiver_block_ctor = partial(
                PerceiverPoolingBlock,
                perceiver_kwargs=dict(init_weights=init_weights),
            )
        
        self.perceiver = perceiver_block_ctor(
            dim=perc_dim,
            num_heads=perc_num_heads,
            num_query_tokens=num_latent_tokens,
        )
        
        # ================================================================
        # Output shape for downstream modules
        # ================================================================
        self.output_shape = (num_latent_tokens, perc_dim)
    
    def get_model_specific_param_group_modifiers(self):
        """Exclude perceiver query from weight decay (standard practice)."""
        return [ExcludeFromWdByNameModifier(name="perceiver.query")]
    
    def forward(self, x, mesh_pos, mesh_edges, batch_idx, condition=None, static_tokens=None):
        """
        Forward pass through FANO encoder.
        
        Args:
            x: (N_total, C) - Flat mesh points across batch
            mesh_pos: (N_total, 2) - Point positions
            mesh_edges: Not used for FANO (kept for API compatibility)
            batch_idx: (N_total,) - Batch index for each point
            condition: (batch, cond_dim) - Optional timestep/velocity conditioning
            static_tokens: Not used (kept for API compatibility)
        
        Returns:
            (batch, num_latent_tokens, perc_dim) - Latent representation
        
        Pipeline:
            (N_total, C) flat mesh
              ↓ GridPatchBuilder
            (batch, num_patches, patch_h, patch_w, C)
              ↓ input_proj
            (batch, num_patches, patch_h, patch_w, d_model)
              ↓ FANO blocks (continuum attention)
            (batch, num_patches, patch_h, patch_w, d_model)
              ↓ Pooling (strategy-dependent)
            (batch, num_tokens, d_model)
              ↓ perc_proj
            (batch, num_tokens, perc_dim)
              ↓ Perceiver cross-attention
            (batch, num_latent_tokens, perc_dim) ✓
        """
        
        # ================================================================
        # STEP 1: Build patches from flat mesh
        # ================================================================
        # Input: (N_total, C) where N_total = batch_size × H × W
        # Output: (batch, num_patches, patch_h, patch_w, C)
        x = self.patch_builder(x, mesh_pos, batch_idx)
        
        # ================================================================
        # STEP 2: Pointwise lift to d_model
        # ================================================================
        # Apply linear projection over channel dimension
        # Input: (batch, num_patches, patch_h, patch_w, input_dim)
        # Output: (batch, num_patches, patch_h, patch_w, d_model)
        x = self.input_proj(x)
        
        # ================================================================
        # STEP 3: FANO continuum attention
        # ================================================================
        # Apply stacked FANO encoder layers
        # Each layer does:
        #   - Fourier spectral conv for Q/K/V projection
        #   - Continuum attention: einsum("bnpxyd,bnqxyd->bnpq")
        #   - Feedforward network (pointwise MLP)
        # Input/Output: (batch, num_patches, patch_h, patch_w, d_model)
        x = self.fano_encoder(x, mask=None)
        
        # ================================================================
        # STEP 4: Pool patches to tokens
        # ================================================================
        # Strategy depends on pooling_type:
        #
        # Direct pooling (custom):
        #   - spatial_avg: Average (H,W) → one token per patch
        #   - learned: MLP probe per patch
        #   - attention: Query-based pooling per patch
        #   Output: (batch, num_patches, d_model)
        #
        # Grid reconstruction (FANO-faithful):
        #   - grid_uniform: Reshape to full grid → uniform subsample
        #   - grid_adaptive: Reshape to full grid → learned top-K
        #   Output: (batch, num_samples, d_model)
        x = self.pooling(x)
        
        # ================================================================
        # STEP 5: Project to perceiver dimension
        # ================================================================
        # (batch, num_tokens, d_model) → (batch, num_tokens, perc_dim)
        x = self.perc_proj(x)
        
        # ================================================================
        # STEP 6: Perceiver cross-attention
        # ================================================================
        # Q: num_latent_tokens learned queries (e.g., 256)
        # K,V: num_tokens from pooling (e.g., 64)
        # Cross-attention compresses to fixed latent size
        # Output: (batch, num_latent_tokens, perc_dim)
        
        block_kwargs = {}
        if condition is not None:
            # Pass timestep/velocity conditioning to perceiver
            block_kwargs["cond"] = condition
        
        x = self.perceiver(kv=x, **block_kwargs)
        
        return x
    
    def __repr__(self):
        return (
            f"CfdFanoPerceiver(\n"
            f"  grid={self.H}×{self.W}, "
            f"patches={self.num_patches_h}×{self.num_patches_w}={self.num_patches_h*self.num_patches_w}, "
            f"patch_size={self.H//self.num_patches_h}×{self.W//self.num_patches_w}\n"
            f"  input_dim={self.input_dim}, d_model={self.d_model}, "
            f"fano_layers={self.num_fano_layers}, heads={self.num_heads}\n"
            f"  fourier_modes={self.fourier_modes}, pooling={self.pooling_type}\n"
            f"  → latent_tokens={self.num_latent_tokens}, perc_dim={self.perc_dim}\n"
            f")"
        )

