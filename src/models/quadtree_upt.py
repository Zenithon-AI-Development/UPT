"""
Quadtree-UPT: UPT with AMR quadtree-based encoding.

Replaces random supernode sampling with physics-driven quadtree partitioning.
"""

import torch
from torch import nn


class QuadtreeUPT(nn.Module):
    """
    UPT model using quadtree nodes instead of random supernodes.
    
    Architecture:
        1. EncoderAMRQuadtree: Encodes quadtree nodes to latent space
        2. Approximator: Temporal propagation (unchanged from UPT)
        3. Decoder: Decodes to output grid (unchanged from UPT)
    """
    
    def __init__(self, conditioner, encoder, approximator, decoder):
        super().__init__()
        self.conditioner = conditioner
        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder
    
    def forward(
        self,
        node_feat,
        node_pos,
        depth,
        batch_idx,
        output_pos,
        timestep,
    ):
        """
        Forward pass for quadtree-UPT.
        
        Args:
            node_feat: (batch_size * max_nodes, input_dim)
            node_pos: (batch_size * max_nodes, 2)
            depth: (batch_size * max_nodes,)
            batch_idx: (batch_size * max_nodes,)
            output_pos: (batch_size, H*W, 2)
            timestep: (batch_size,)
        
        Returns:
            pred: (batch_size, H*W, output_dim)
        """
        # Get timestep conditioning
        condition = self.conditioner(timestep)
        
        # Encode quadtree nodes to latent space
        latent = self.encoder(
            node_feat=node_feat,
            node_pos=node_pos,
            depth=depth,
            batch_idx=batch_idx,
            condition=condition,
        )
        # latent shape: (batch_size, max_nodes or num_latent_tokens, enc_dim)
        
        # Temporal propagation
        latent = self.approximator(latent, condition=condition)
        # latent shape: (batch_size, num_tokens, dim)
        
        # Decode to output grid
        pred = self.decoder(
            x=latent,
            output_pos=output_pos,
            condition=condition,
        )
        # pred shape: (batch_size, H*W, output_dim)
        
        return pred
    
    @torch.no_grad()
    def rollout(
        self,
        initial_grid,
        num_steps,
        collator,
    ):
        """
        Autoregressive rollout for multiple timesteps.
        
        Args:
            initial_grid: (C, H, W) initial state
            num_steps: number of rollout steps
            collator: QuadtreeCollator instance
        
        Returns:
            predictions: List of (H, W, C) predictions
        """
        self.eval()
        
        current_grid = initial_grid
        predictions = []
        
        for step in range(num_steps):
            # Create batch with single sample
            batch_sample = [{
                "input_grid": current_grid,
                "target_grid": current_grid,  # Dummy target
                "timestep": step,
                "index": 0,
            }]
            
            # Collate
            batch = collator(batch_sample)
            
            # Forward pass
            pred = self.forward(
                node_feat=batch["node_feat"],
                node_pos=batch["node_pos"],
                depth=batch["depth"],
                batch_idx=batch["batch_idx"],
                output_pos=batch["output_pos"],
                timestep=batch["timestep"],
            )
            # pred shape: (1, H*W, C)
            
            # Reshape to grid
            C = initial_grid.shape[0]
            H = initial_grid.shape[1]
            W = initial_grid.shape[2]
            pred_grid = pred[0].reshape(H, W, C)  # (H, W, C)
            predictions.append(pred_grid)
            
            # Use prediction as next input
            current_grid = pred_grid.permute(2, 0, 1)  # (C, H, W)
        
        return predictions

