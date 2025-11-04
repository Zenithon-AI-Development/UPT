"""
Collator for quadtree-based UPT training.

This collator:
1. Applies quadtree partitioning to each sample
2. Extracts node features and positions
3. Pads/truncates to max_nodes
4. Creates batch tensors for UPT
"""

import torch
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from amr_transformer.tree import quadtree_partition_parallel, pad_to_power_of_k


class QuadtreeCollator:
    """
    Collator that applies quadtree partitioning and prepares data for UPT.
    """
    def __init__(
        self,
        max_nodes=5376,
        k=2,
        min_size=4,
        max_size=16,
        common_refine_threshold=0.4,
        integral_refine_threshold=0.1,
        vorticity_threshold=0.5,
        momentum_threshold=0.5,
        shear_threshold=0.5,
        condition_type='grad',
        deterministic=False,
    ):
        self.max_nodes = max_nodes
        self.k = k
        self.min_size = min_size
        self.max_size = max_size
        self.common_refine_threshold = common_refine_threshold
        self.integral_refine_threshold = integral_refine_threshold
        self.vorticity_threshold = vorticity_threshold
        self.momentum_threshold = momentum_threshold
        self.shear_threshold = shear_threshold
        self.condition_type = condition_type
        self.deterministic = deterministic
        
    def __call__(self, batch):
        """
        Args:
            batch: List of dicts with keys:
                - input_grid: (C, H, W) input state
                - target_grid: (C, H, W) target state
                - timestep: scalar timestep index
                - index: sample index
        
        Returns:
            collated_batch: Dict with:
                - node_feat: (batch_size * max_nodes, input_dim)
                - node_pos: (batch_size * max_nodes, 2)  # (x, y)
                - depth: (batch_size * max_nodes,)
                - batch_idx: (batch_size * max_nodes,)
                - output_pos: (batch_size, H*W, 2)  # for decoder
                - output_feat: (batch_size * H*W, C)  # target
                - timestep: (batch_size,)
        """
        batch_size = len(batch)
        
        all_node_feats = []
        all_node_positions = []
        all_depths = []
        all_output_grids = []
        all_timesteps = []
        
        for sample_idx, sample in enumerate(batch):
            input_grid = sample["input_grid"]  # (C, H, W)
            target_grid = sample["target_grid"]  # (C, H, W)
            timestep = sample["timestep"]
            
            # Convert to (H, W, C) for quadtree
            C, H, W = input_grid.shape
            input_hwc = input_grid.permute(1, 2, 0)  # (H, W, C)
            target_hwc = target_grid.permute(1, 2, 0)  # (H, W, C)
            
            # Pad to power of k
            fill_value = 0.0
            input_padded = pad_to_power_of_k(input_hwc.unsqueeze(0), fill_value, self.k)[0]  # (H_pad, W_pad, C)
            target_padded = pad_to_power_of_k(target_hwc.unsqueeze(0), fill_value, self.k)[0]  # (H_pad, W_pad, C)
            
            # Create mask (all ones for uniform grid)
            mask = torch.ones(input_padded.shape[0], input_padded.shape[1], 1, device=input_padded.device)
            
            # Use velocity fields as feature for refinement (channels 2, 3 for vx, vy)
            # For TRL2D: [density, pressure, vx, vy]
            if C >= 4:
                feature_field = input_padded[:, :, 2:4]  # (H, W, 2) velocity
            else:
                feature_field = input_padded[:, :, :2]  # Use first 2 channels
            
            # Run quadtree partitioning
            regions, patches_tensor, labels_tensor = quadtree_partition_parallel(
                inputs=input_padded,
                labels=target_padded,
                mask=mask,
                feature_field=feature_field,
                k=self.k,
                min_size=self.min_size,
                max_size=self.max_size,
                common_refine_threshold=self.common_refine_threshold,
                integral_refine_threshold=self.integral_refine_threshold,
                vorticity_threshold=self.vorticity_threshold,
                momentum_threshold=self.momentum_threshold,
                shear_threshold=self.shear_threshold,
                condition_type=self.condition_type,
            )
            
            # patches_tensor shape: (N, k*k, C+3) where last 3 are (depth, x, y)
            # Reshape to (N, k, k, C+3)
            N = patches_tensor.shape[0]
            patches_reshaped = patches_tensor.reshape(N, self.k, self.k, patches_tensor.shape[-1])
            
            # Extract position encoding and features
            # Average over k×k patch to get node-level features
            node_features = patches_reshaped[:, :, :, :-3].mean(dim=(1, 2))  # (N, C)
            depth_vals = patches_reshaped[:, 0, 0, -3]  # (N,) - depth is same for all k×k
            x_vals = patches_reshaped[:, 0, 0, -2]  # (N,) - x position
            y_vals = patches_reshaped[:, 0, 0, -1]  # (N,) - y position
            
            # Normalize positions to [0, 1]
            H_pad, W_pad = input_padded.shape[0], input_padded.shape[1]
            x_norm = x_vals / H_pad
            y_norm = y_vals / W_pad
            
            # Sort by (depth, x, y) for consistency
            sort_key = depth_vals * 1e6 + x_norm * 1e3 + y_norm
            sort_indices = torch.argsort(sort_key)
            
            node_features = node_features[sort_indices]
            depth_vals = depth_vals[sort_indices]
            x_norm = x_norm[sort_indices]
            y_norm = y_norm[sort_indices]
            
            # Truncate or pad to max_nodes
            actual_N = N  # Store actual count before truncation
            if N > self.max_nodes:
                print(f"WARNING: Sample {sample_idx} has {N} nodes, truncating to {self.max_nodes}")
                node_features = node_features[:self.max_nodes]
                depth_vals = depth_vals[:self.max_nodes]
                x_norm = x_norm[:self.max_nodes]
                y_norm = y_norm[:self.max_nodes]
                N = self.max_nodes
            
            if N < self.max_nodes:
                # Pad with zeros
                pad_size = self.max_nodes - N
                node_features = torch.cat([
                    node_features,
                    torch.zeros(pad_size, C, device=node_features.device)
                ], dim=0)
                depth_vals = torch.cat([
                    depth_vals,
                    torch.zeros(pad_size, device=depth_vals.device)
                ], dim=0)
                x_norm = torch.cat([
                    x_norm,
                    torch.zeros(pad_size, device=x_norm.device)
                ], dim=0)
                y_norm = torch.cat([
                    y_norm,
                    torch.zeros(pad_size, device=y_norm.device)
                ], dim=0)
            
            # Stack positions
            node_positions = torch.stack([x_norm, y_norm], dim=1)  # (max_nodes, 2)
            
            all_node_feats.append(node_features)
            all_node_positions.append(node_positions)
            all_depths.append(depth_vals)
            all_output_grids.append(target_hwc)
            all_timesteps.append(timestep)
        
        # Concatenate all samples
        node_feat = torch.cat(all_node_feats, dim=0)  # (batch_size * max_nodes, C)
        node_pos = torch.cat(all_node_positions, dim=0)  # (batch_size * max_nodes, 2)
        depth = torch.cat(all_depths, dim=0)  # (batch_size * max_nodes,)
        
        # Create batch_idx
        batch_idx = torch.cat([
            torch.full((self.max_nodes,), i, dtype=torch.long)
            for i in range(batch_size)
        ])
        
        # Output positions (uniform grid for decoder)
        # Use original grid positions
        output_pos_list = []
        output_feat_list = []
        for target_hwc in all_output_grids:
            H, W, C = target_hwc.shape
            # Create grid positions
            y_coords = torch.linspace(0, 1, H, device=target_hwc.device)
            x_coords = torch.linspace(0, 1, W, device=target_hwc.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            output_pos_sample = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (H*W, 2)
            output_pos_list.append(output_pos_sample)
            output_feat_list.append(target_hwc.reshape(-1, C))
        
        output_pos = torch.stack(output_pos_list, dim=0)  # (batch_size, H*W, 2)
        output_feat = torch.cat(output_feat_list, dim=0)  # (batch_size * H*W, C)
        
        # Timesteps
        timestep_tensor = torch.tensor(all_timesteps, dtype=torch.long)
        
        return {
            "node_feat": node_feat,
            "node_pos": node_pos,
            "depth": depth,
            "batch_idx": batch_idx,
            "output_pos": output_pos,
            "output_feat": output_feat,
            "timestep": timestep_tensor,
        }

