"""
Encoder for AMR quadtree nodes to replace UPT's supernode-based encoder.

Key differences from EncoderSupernodes:
1. No SupernodePooling - quadtree nodes are already aggregated patches
2. Positional encoding includes depth (hierarchical refinement level)
3. Handles variable number of nodes with padding/truncation
"""

from functools import partial
import torch
from torch import nn
import einops
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock


class QuadtreePositionalEncoding(nn.Module):
    """
    Positional encoding for quadtree nodes.
    Encodes (depth, x, y) positions where depth indicates refinement level.
    """
    def __init__(self, dim, ndim=2, max_depth=10):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.max_depth = max_depth
        
        # Allocate dimensions: depth gets 1/3, x gets 1/3, y gets 1/3
        self.dim_depth = dim // 3
        self.dim_x = dim // 3
        self.dim_y = dim - self.dim_depth - self.dim_x
        
        # Learnable depth embedding (discrete levels)
        self.depth_embed = nn.Embedding(max_depth, self.dim_depth)
        
        # Continuous position encoding for x, y
        self.pos_embed_x = self._create_sincos_encoding(self.dim_x)
        self.pos_embed_y = self._create_sincos_encoding(self.dim_y)
        
    def _create_sincos_encoding(self, dim):
        """Create sin/cos positional encoding weights"""
        # Frequency bands
        freq = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))
        return freq
    
    def forward(self, depth, x, y):
        """
        Args:
            depth: (batch_size * num_nodes,) tensor of depth levels (int)
            x: (batch_size * num_nodes,) tensor of x positions (float, normalized)
            y: (batch_size * num_nodes,) tensor of y positions (float, normalized)
        
        Returns:
            pos_encoding: (batch_size * num_nodes, dim) tensor
        """
        # Depth embedding (discrete)
        depth_enc = self.depth_embed(depth.long())  # (N, dim_depth)
        
        # X position encoding (continuous sincos)
        freq_x = self.pos_embed_x.to(x.device)
        x_enc = torch.zeros(len(x), self.dim_x, device=x.device)
        x_enc[:, 0::2] = torch.sin(x.unsqueeze(1) * freq_x)
        x_enc[:, 1::2] = torch.cos(x.unsqueeze(1) * freq_x)
        
        # Y position encoding (continuous sincos)
        freq_y = self.pos_embed_y.to(y.device)
        y_enc = torch.zeros(len(y), self.dim_y, device=y.device)
        y_enc[:, 0::2] = torch.sin(y.unsqueeze(1) * freq_y)
        y_enc[:, 1::2] = torch.cos(y.unsqueeze(1) * freq_y)
        
        # Concatenate all encodings
        pos_encoding = torch.cat([depth_enc, x_enc, y_enc], dim=1)
        
        return pos_encoding


class EncoderAMRQuadtree(nn.Module):
    """
    Encoder for AMR quadtree nodes.
    Replaces UPT's EncoderSupernodes but works with quadtree partitioned data.
    """
    def __init__(
            self,
            input_dim,
            ndim,
            enc_dim,
            enc_depth,
            enc_num_heads,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            max_depth=10,
            init_weights="truncnormal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.ndim = ndim
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights
        
        # Feature projection
        self.input_proj = LinearProjection(input_dim, enc_dim, init_weights=init_weights)
        
        # Quadtree positional encoding (depth, x, y)
        self.pos_encoding = QuadtreePositionalEncoding(
            dim=enc_dim,
            ndim=ndim,
            max_depth=max_depth,
        )
        
        # Transformer blocks
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )
        
        # Optional perceiver pooling to reduce token count
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )
    
    def forward(self, node_feat, node_pos, depth, batch_idx, condition=None):
        """
        Args:
            node_feat: (batch_size * max_nodes, input_dim) - features of quadtree nodes
            node_pos: (batch_size * max_nodes, ndim) - (x, y) positions of quadtree nodes
            depth: (batch_size * max_nodes,) - depth level of each quadtree node
            batch_idx: (batch_size * max_nodes,) - which batch each node belongs to
            condition: (batch_size, cond_dim) - optional timestep conditioning
        
        Returns:
            latent: (batch_size, num_nodes or num_latent_tokens, dim)
        """
        # Check inputs
        assert node_feat.ndim == 2, "expected sparse tensor (batch_size * max_nodes, input_dim)"
        assert node_pos.ndim == 2, "expected sparse tensor (batch_size * max_nodes, ndim)"
        assert depth.ndim == 1, "expected 1D tensor of depth levels"
        assert batch_idx.ndim == 1, "expected 1D tensor for batch assignment"
        assert len(node_feat) == len(node_pos) == len(depth) == len(batch_idx)
        
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"
        
        # Prepare conditioning kwargs for DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition
        
        # Project features
        x = self.input_proj(node_feat)  # (N_total, enc_dim)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(depth, node_pos[:, 0], node_pos[:, 1])  # (N_total, enc_dim)
        x = x + pos_enc
        
        # Reshape to batch format for transformer
        batch_size = batch_idx.max() + 1
        num_nodes_per_sample = len(node_feat) // batch_size
        
        x = einops.rearrange(
            x,
            "(batch_size num_nodes) dim -> batch_size num_nodes dim",
            batch_size=batch_size,
            num_nodes=num_nodes_per_sample,
        )
        
        # Apply transformer blocks
        x = self.blocks(x, **cond_kwargs)
        
        # Optional perceiver pooling
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)
        
        return x

