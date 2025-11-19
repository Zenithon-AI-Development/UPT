from __future__ import annotations

from typing import Dict, Optional

import torch

from slot_upt.slot_assignment import assign_cells_to_slots_voxel_grid

from .interfaces import SubnodeBatch


class VoxelGridBackend:
    def __init__(self, M: int, N: int, ndim: int = 2):
        if M is None or N is None:
            raise ValueError("VoxelGridBackend requires both M (supernodes) and N (slots per supernode).")
        self.M = int(M)
        self.N = int(N)
        self.ndim = int(ndim)

    def generate(self, batch_ctx: Dict[str, torch.Tensor]) -> SubnodeBatch:
        mesh_pos: torch.Tensor = batch_ctx["mesh_pos"]
        features: torch.Tensor = batch_ctx["features"]
        batch_idx: torch.Tensor = batch_ctx["batch_idx"]
        num_timesteps: int = int(batch_ctx.get("num_timesteps", 1))
        levels_per_cell: Optional[torch.Tensor] = batch_ctx.get("levels_per_cell")

        subnode_feats, subnode_mask, slot2cell, slot_positions, subnode_level = assign_cells_to_slots_voxel_grid(
            mesh_pos=mesh_pos,
            features=features,
            M=self.M,
            N=self.N,
            batch_idx=batch_idx,
            num_timesteps=num_timesteps,
            ndim=self.ndim,
            levels_per_cell=levels_per_cell,
        )

        return SubnodeBatch(
            subnode_feats=subnode_feats,
            subnode_mask=subnode_mask,
            slot2cell=slot2cell,
            slot_positions=slot_positions,
            subnode_level=subnode_level,
            subnode_pos=None,
        )

