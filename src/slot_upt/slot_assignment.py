"""
Slot assignment utilities for Stage 1 slot-based UPT.

Implements voxel grid assignment: assigns cells to fixed [M, N] slot structure.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def assign_cells_to_slots_voxel_grid(
    mesh_pos: torch.Tensor,
    features: torch.Tensor,
    M: int,
    N: int,
    batch_idx: torch.Tensor,
    num_timesteps: int = 1,
    ndim: int = 2,
    levels_per_cell: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Assign cells to slots using voxel grid partitioning.
    
    Args:
        mesh_pos: [B*K, d_x] cell positions (flattened batch)
        features: [B*K, C] or [B*K, T*C] cell features
        M: Number of supernodes (voxel grid size)
        N: Number of slots per supernode
        batch_idx: [B*K] batch indices
        num_timesteps: Number of timesteps (T)
        ndim: Spatial dimension (2 or 3)
    
    Returns:
        subnode_feats: [B, T, M, N, C] slot features (zeros for empty slots)
        subnode_mask: [B, T, M, N] mask (1 for real, 0 for empty)
        slot2cell: [B, T, M, N] cell indices (-1 for empty slots)
        slot_positions: [M, N, d_x] canonical slot positions
        subnode_level: [B, T, M, N] levels (or None if levels_per_cell is None)
    """
    device = mesh_pos.device
    batch_size = batch_idx.max().item() + 1
    num_cells = mesh_pos.shape[0]
    
    # Determine if features are time-varying
    # If features are [B*K, T*C], reshape to [B*K, T, C]
    if features.ndim == 2:
        if features.shape[1] % num_timesteps == 0:
            C = features.shape[1] // num_timesteps
            features = features.view(num_cells, num_timesteps, C)
        else:
            # Single timestep, add time dimension
            C = features.shape[1]
            features = features.unsqueeze(1)  # [B*K, 1, C]
            num_timesteps = 1
    else:
        C = features.shape[-1]
    
    # Compute voxel grid bounds per batch
    # For simplicity, use a regular grid M_x × M_y (or M_x × M_y × M_z for 3D)
    # Choose grid dimensions that multiply to M
    if ndim == 2:
        # 2D: M_x × M_y = M
        M_x = int(np.sqrt(M))
        M_y = M // M_x
        while M_x * M_y < M:
            M_x += 1
            M_y = M // M_x
        grid_shape = (M_x, M_y)
    else:
        # 3D: M_x × M_y × M_z = M
        M_x = int(np.cbrt(M))
        M_y = M_z = M_x
        while M_x * M_y * M_z < M:
            M_x += 1
            M_y = M_z = M_x
        grid_shape = (M_x, M_y, M_z)
    
    # Initialize output tensors
    subnode_feats = torch.zeros(batch_size, num_timesteps, M, N, C, device=device, dtype=features.dtype)
    subnode_mask = torch.zeros(batch_size, num_timesteps, M, N, device=device, dtype=torch.bool)
    slot2cell = torch.full((batch_size, num_timesteps, M, N), -1, device=device, dtype=torch.long)
    subnode_level = None
    if levels_per_cell is not None:
        subnode_level = torch.full((batch_size, num_timesteps, M, N), 0, device=device, dtype=torch.long)
    
    # Process each batch element
    for b in range(batch_size):
        batch_mask = batch_idx == b
        batch_pos = mesh_pos[batch_mask]  # [K_b, d_x]
        batch_feats = features[batch_mask]  # [K_b, T, C]
        K_b = batch_pos.shape[0]
        
        if K_b == 0:
            continue
        
        # Compute voxel grid bounds for this batch
        pos_min = batch_pos.min(dim=0)[0]  # [d_x]
        pos_max = batch_pos.max(dim=0)[0]  # [d_x]
        pos_range = pos_max - pos_min
        pos_range = torch.clamp(pos_range, min=1e-6)  # Avoid division by zero
        
        # Assign each cell to a voxel (supernode m)
        # Normalize positions to [0, 1] then scale to grid indices
        normalized_pos = (batch_pos - pos_min) / pos_range  # [K_b, d_x]
        
        if ndim == 2:
            grid_scale = torch.tensor([M_x, M_y], device=device, dtype=normalized_pos.dtype)
            grid_indices = (normalized_pos * grid_scale).long()
            max_vals = torch.tensor([M_x - 1, M_y - 1], device=device, dtype=grid_indices.dtype)
            min_vals = torch.zeros_like(max_vals)
            grid_indices = torch.max(grid_indices, min_vals)
            grid_indices = torch.min(grid_indices, max_vals)
            # Convert 2D grid index to linear supernode index
            m_indices = grid_indices[:, 0] * M_y + grid_indices[:, 1]
        else:
            grid_scale = torch.tensor([M_x, M_y, M_z], device=device, dtype=normalized_pos.dtype)
            grid_indices = (normalized_pos * grid_scale).long()
            max_vals = torch.tensor([M_x - 1, M_y - 1, M_z - 1], device=device, dtype=grid_indices.dtype)
            min_vals = torch.zeros_like(max_vals)
            grid_indices = torch.max(grid_indices, min_vals)
            grid_indices = torch.min(grid_indices, max_vals)
            m_indices = grid_indices[:, 0] * (M_y * M_z) + grid_indices[:, 1] * M_z + grid_indices[:, 2]
        
        # Clamp to valid range [0, M-1]
        m_indices = torch.clamp(m_indices, 0, M - 1)
        
        # For each supernode m, collect assigned cells and order them
        for m in range(M):
            cell_mask = m_indices == m
            assigned_cells = torch.where(cell_mask)[0]
            
            if len(assigned_cells) == 0:
                continue
            
            # Get positions and features for assigned cells
            assigned_pos = batch_pos[assigned_cells]  # [|S_m|, d_x]
            assigned_feats = batch_feats[assigned_cells]  # [|S_m|, T, C]
            
            # Compute voxel center for ordering
            if ndim == 2:
                m_x = (m // M_y) % M_x
                m_y = m % M_y
                voxel_center = pos_min + pos_range * torch.tensor(
                    [(m_x + 0.5) / M_x, (m_y + 0.5) / M_y],
                    device=device, dtype=pos_min.dtype
                )
            else:
                m_x = (m // (M_y * M_z)) % M_x
                m_y = (m // M_z) % M_y
                m_z = m % M_z
                voxel_center = pos_min + pos_range * torch.tensor(
                    [(m_x + 0.5) / M_x, (m_y + 0.5) / M_y, (m_z + 0.5) / M_z],
                    device=device, dtype=pos_min.dtype
                )
            
            # Order cells by distance to voxel center
            distances = torch.norm(assigned_pos - voxel_center.unsqueeze(0), dim=1)
            sorted_indices = torch.argsort(distances)
            
            # Assign first N cells to slots
            num_to_assign = min(len(assigned_cells), N)
            slot_indices = sorted_indices[:num_to_assign]
            
            # Map back to global cell indices (within batch)
            global_cell_indices = assigned_cells[slot_indices]
            
            # Store in output tensors
            for t in range(num_timesteps):
                for n_idx, n in enumerate(range(num_to_assign)):
                    # Get local cell index within batch
                    local_cell_idx = assigned_cells[slot_indices[n_idx]].item()
                    # Find global index in original mesh_pos
                    global_indices = torch.where(batch_mask)[0]
                    global_idx = global_indices[local_cell_idx].item()
                    
                    subnode_feats[b, t, m, n] = assigned_feats[n_idx, t]
                    subnode_mask[b, t, m, n] = True
                    slot2cell[b, t, m, n] = global_idx
                    if subnode_level is not None:
                        level_val = levels_per_cell[global_idx]
                        if level_val.ndim > 0 and level_val.numel() > 1:
                            level_val = level_val.flatten()[0]
                        subnode_level[b, t, m, n] = int(level_val)
    
    # Compute canonical slot positions
    slot_positions = get_slot_positions(M, N, ndim, device=device)
    
    return subnode_feats, subnode_mask, slot2cell, slot_positions, subnode_level


def get_slot_positions(
    M: int,
    N: int,
    ndim: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute canonical positions for each slot (m, n).
    
    For each supernode m, positions are arranged in a regular pattern
    (e.g., grid or radial pattern) within the voxel.
    
    Args:
        M: Number of supernodes
        N: Number of slots per supernode
        ndim: Spatial dimension (2 or 3)
        device: Device for tensor
    
    Returns:
        slot_positions: [M, N, d_x] canonical slot positions
    """
    if device is None:
        device = torch.device("cpu")
    
    # Determine grid shape for supernodes
    if ndim == 2:
        M_x = int(np.sqrt(M))
        M_y = M // M_x
        while M_x * M_y < M:
            M_x += 1
            M_y = M // M_x
    else:
        M_x = int(np.cbrt(M))
        M_y = M_z = M_x
        while M_x * M_y * M_z < M:
            M_x += 1
            M_y = M_z = M_x
    
    # For each supernode, arrange slots in a regular pattern
    # Simple approach: arrange in a small grid within each voxel
    if ndim == 2:
        N_x = int(np.sqrt(N))
        N_y = N // N_x
        while N_x * N_y < N:
            N_x += 1
            N_y = N // N_x
        
        slot_positions = torch.zeros(M, N, ndim, device=device)
        for m in range(M):
            m_x = (m // M_y) % M_x
            m_y = m % M_y
            # Voxel center
            voxel_center_x = (m_x + 0.5) / M_x
            voxel_center_y = (m_y + 0.5) / M_y
            
            for n in range(N):
                n_x = (n // N_y) % N_x
                n_y = n % N_y
                # Offset within voxel (normalized to [0, 1])
                offset_x = (n_x + 0.5) / N_x - 0.5
                offset_y = (n_y + 0.5) / N_y - 0.5
                # Scale offset to voxel size
                slot_positions[m, n, 0] = voxel_center_x + offset_x / M_x
                slot_positions[m, n, 1] = voxel_center_y + offset_y / M_y
    else:
        N_x = int(np.cbrt(N))
        N_y = N_z = N_x
        while N_x * N_y * N_z < N:
            N_x += 1
            N_y = N_z = N_x
        
        slot_positions = torch.zeros(M, N, ndim, device=device)
        for m in range(M):
            m_x = (m // (M_y * M_z)) % M_x
            m_y = (m // M_z) % M_y
            m_z = m % M_z
            voxel_center_x = (m_x + 0.5) / M_x
            voxel_center_y = (m_y + 0.5) / M_y
            voxel_center_z = (m_z + 0.5) / M_z
            
            for n in range(N):
                n_x = (n // (N_y * N_z)) % N_x
                n_y = (n // N_z) % N_y
                n_z = n % N_z
                offset_x = (n_x + 0.5) / N_x - 0.5
                offset_y = (n_y + 0.5) / N_y - 0.5
                offset_z = (n_z + 0.5) / N_z - 0.5
                slot_positions[m, n, 0] = voxel_center_x + offset_x / M_x
                slot_positions[m, n, 1] = voxel_center_y + offset_y / M_y
                slot_positions[m, n, 2] = voxel_center_z + offset_z / M_z
    
    return slot_positions


def scatter_slots_to_cells(
    subnode_feats: torch.Tensor,
    slot2cell: torch.Tensor,
    num_cells: int,
) -> torch.Tensor:
    """
    Scatter slot features back to original cell positions.
    
    Args:
        subnode_feats: [B, T, M, N, C] slot features
        slot2cell: [B, T, M, N] cell indices (-1 for empty slots)
        num_cells: Total number of cells (for output shape)
    
    Returns:
        cell_feats: [B*K, T, C] or [B*K, T*C] cell features
    """
    B, T, M, N, C = subnode_feats.shape
    device = subnode_feats.device
    
    # Initialize output
    cell_feats = torch.zeros(num_cells, T, C, device=device, dtype=subnode_feats.dtype)
    
    # Scatter for each batch and timestep
    # Use index_add_ for efficient scattering
    for b in range(B):
        for t in range(T):
            for m in range(M):
                for n in range(N):
                    cell_global_idx = slot2cell[b, t, m, n].item()
                    if cell_global_idx >= 0 and cell_global_idx < num_cells:
                        # If multiple slots map to same cell, take the last one
                        # (or could average - for now, last write wins)
                        cell_feats[cell_global_idx, t] = subnode_feats[b, t, m, n]
    
    # Return prediction for final timestep by default
    if T == 1:
        return cell_feats.squeeze(1)  # [num_cells, C]
    return cell_feats[:, -1]  # [num_cells, C]

