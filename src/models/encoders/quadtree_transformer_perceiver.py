from functools import partial

import einops
import torch
from kappamodules.layers import LinearProjection, ContinuousSincosEmbed
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class QuadtreeTransformerPerceiver(SingleModelBase):
    """
    Encoder that processes variable-length quadtree nodes directly without supernode pooling.
    
    Flow:
    1. Embed quadtree nodes (features + positions + depth) -> [B, N, embed_dim]
    2. Apply transformer blocks with masking -> [B, N, enc_dim]
    3. PerceiverPoolingBlock compresses to fixed size -> [B, num_latent_tokens, perc_dim]
    
    This skips the supernode concept entirely and processes all quadtree nodes directly.
    Supports quadtree dict format from SHINE_mapping/quadtree with point_hierarchies, pyramids, features.
    """
    
    def __init__(
        self,
        embed_dim,
        enc_dim,
        perc_dim,
        enc_depth,
        enc_num_attn_heads,
        perc_num_attn_heads,
        num_latent_tokens=None,
        use_enc_norm=False,
        drop_path_rate=0.0,
        init_weights="xavier_uniform",
        use_positional_embedding=True,
        max_quadtree_level=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_latent_tokens = num_latent_tokens
        self.use_enc_norm = use_enc_norm
        self.drop_path_rate = drop_path_rate
        self.init_weights = init_weights
        self.use_positional_embedding = use_positional_embedding
        self.max_quadtree_level = max_quadtree_level

        # Input shape is (None, input_dim) for variable-length sequences
        _, input_dim = self.input_shape
        ndim = 2
        self.static_ctx["ndim"] = ndim

        # Node embedding: project features to embed_dim
        self.node_embed = nn.Linear(input_dim, embed_dim)
        
        # Positional embedding (for quadtree node centers/positions)
        if use_positional_embedding:
            # Spatial position embedding (2D or 3D)
            self.pos_embed = ContinuousSincosEmbed(dim=embed_dim, ndim=ndim)
            # Depth/level embedding (1D, normalized by max_level)
            self.depth_embed = nn.Linear(1, embed_dim)
        else:
            self.pos_embed = None
            self.depth_embed = None

        # Encoder normalization (optional)
        self.enc_norm = nn.LayerNorm(embed_dim, eps=1e-6) if use_enc_norm else nn.Identity()
        
        # Project to encoder dimension
        self.enc_proj = LinearProjection(embed_dim, enc_dim)
        
        # Transformer blocks (disabled - not used)
        # if "condition_dim" in self.static_ctx:
        #     block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        # else:
        #     block_ctor = PrenormBlock
        # self.blocks = nn.ModuleList([
        #     block_ctor(
        #         dim=enc_dim,
        #         num_heads=enc_num_attn_heads,
        #         init_weights=init_weights,
        #         drop_path=drop_path_rate
        #     )
        #     for _ in range(enc_depth)
        # ])
        self.blocks = None

        # Perceiver pooling
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.static_ctx["condition_dim"],
                    init_weights=init_weights,
                ),
            )
        else:
            block_ctor = partial(
                PerceiverPoolingBlock,
                perceiver_kwargs=dict(init_weights=init_weights),
            )
        self.perceiver = block_ctor(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_latent_tokens,
        )

        # Output shape
        self.output_shape = (num_latent_tokens, perc_dim)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="perceiver.query")]

    def forward(
        self,
        x=None,
        node_positions=None,
        node_depths=None,
        batch_idx=None,
        condition=None,
        static_tokens=None,
        quadtree_dict=None,
        **kwargs
    ):
        """
        Forward pass for variable-length quadtree nodes.
        
        Args:
            x: Node features [N_total, input_dim] where N_total varies per batch (optional if quadtree_dict provided)
            node_positions: Node positions/centers [N_total, ndim] in [-1, 1] range (optional)
            node_depths: Node depths/levels [N_total] (optional)
            batch_idx: Batch indices [N_total] to group nodes by batch
            condition: Optional conditioning [B, cond_dim]
            static_tokens: Optional static tokens (not used, kept for compatibility)
            quadtree_dict: Dictionary with 'point_hierarchies', 'pyramids', 'features' (optional)
                - point_hierarchies: (M, 2) integer coordinates at different levels
                - pyramids: (1, 2, max_level+1) with [level, start_idx] per level
                - features: (M, C) features at each node
        
        Returns:
            Latent representation [B, num_latent_tokens, perc_dim]
        """
        # Extract from quadtree_dict if provided
        if quadtree_dict is not None:
            point_hier = quadtree_dict['point_hierarchies']  # (M, 2) integer coords
            pyramids = quadtree_dict['pyramids'][0]  # (2, max_level+1)
            features = quadtree_dict['features']  # (M, C)
            max_level = pyramids.shape[1] - 1
            
            # Convert integer coordinates to [-1, 1] range
            # Integer coords are at different levels, need to convert based on level
            node_positions_list = []
            node_depths_list = []
            
            for level in range(max_level + 1):
                start_idx = int(pyramids[1, level].item())
                if level < max_level:
                    end_idx = int(pyramids[1, level + 1].item())
                else:
                    end_idx = point_hier.shape[0]
                
                if end_idx > start_idx:
                    level_nodes = point_hier[start_idx:end_idx]  # (N_level, 2)
                    # Convert integer coords to [-1, 1] range
                    # At level L, coords are in [0, 2^L), convert to [-1, 1]
                    level_coords_float = (level_nodes.float() / (2 ** level)) * 2.0 - 1.0
                    node_positions_list.append(level_coords_float)
                    node_depths_list.append(torch.full((level_nodes.shape[0],), level, dtype=torch.float32, device=point_hier.device))
            
            if node_positions_list:
                node_positions = torch.cat(node_positions_list, dim=0)  # (M, 2)
                node_depths = torch.cat(node_depths_list, dim=0)  # (M,)
                x = features  # (M, C)
            else:
                # Empty quadtree
                device = point_hier.device
                node_positions = torch.zeros((0, 2), device=device)
                node_depths = torch.zeros((0,), dtype=torch.float32, device=device)
                x = torch.zeros((0, features.shape[1]), device=device)
        
        # Step 1: Embed nodes
        x = self.node_embed(x)  # [N_total, embed_dim]
        
        # Add positional embedding (spatial + depth)
        if self.use_positional_embedding:
            if node_positions is not None:
                x = x + self.pos_embed(node_positions)  # Spatial position
            if node_depths is not None and self.depth_embed is not None:
                # Normalize depth by max_level and embed
                depth_normalized = (node_depths / self.max_quadtree_level).unsqueeze(-1)  # (N_total, 1)
                depth_emb = self.depth_embed(depth_normalized)  # (N_total, embed_dim)
                x = x + depth_emb
        
        # Step 2: Convert to dense batch format with masking
        # This handles variable-length sequences by padding to max length in batch
        if batch_idx is not None:
            x, mask = to_dense_batch(x, batch_idx)  # [B, N_max, embed_dim], [B, N_max]
        else:
            # If no batch_idx, assume single batch
            x = x.unsqueeze(0)  # [1, N, embed_dim]
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=x.device)  # [1, N]
        
        # Prepare attention mask for transformer blocks
        # If all sequences are fully valid, we can skip masking for efficiency
        if torch.all(mask):
            attn_mask = None
        else:
            # Reshape mask for attention: [B, 1, 1, N_max]
            # The mask from to_dense_batch is True for valid positions
            # Transformer blocks expect this format (True = valid, can attend)
            attn_mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")
        
        # Step 3: Apply encoder normalization and projection
        x = self.enc_norm(x)
        x = self.enc_proj(x)  # [B, N_max, enc_dim]
        
        # Step 4: Apply transformer blocks with masking (disabled - not used)
        # block_kwargs = {}
        # if condition is not None:
        #     block_kwargs["cond"] = condition
        # if attn_mask is not None:
        #     block_kwargs["attn_mask"] = attn_mask
        # 
        # for blk in self.blocks:
        #     x = blk(x, **block_kwargs)
        
        # Step 5: Project to perceiver dimension
        x = self.perc_proj(x)  # [B, N_max, perc_dim]
        
        # Step 6: PerceiverPoolingBlock compresses variable N_max to fixed num_latent_tokens
        # Pass attn_mask to mask out padded positions in the KV attention
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        if attn_mask is not None:
            block_kwargs["attn_mask"] = attn_mask
        # Only pass 'cond' if it's a DitPerceiverPoolingBlock
        if isinstance(self.perceiver, DitPerceiverPoolingBlock):
            x = self.perceiver(kv=x, cond=block_kwargs.get("cond"), attn_mask=block_kwargs.get("attn_mask"))
        else:
            x = self.perceiver(kv=x, attn_mask=block_kwargs.get("attn_mask"))
        
        return x

