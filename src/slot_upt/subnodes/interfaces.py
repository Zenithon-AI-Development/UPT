from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass
class SubnodeBatch:
    subnode_feats: torch.Tensor  # [B, T, M, N, C]
    subnode_mask: torch.Tensor  # [B, T, M, N]
    slot2cell: torch.Tensor  # [B, T, M, N]
    slot_positions: torch.Tensor  # [M, N, dx]
    subnode_level: Optional[torch.Tensor] = None  # [B, T, M, N]
    subnode_pos: Optional[torch.Tensor] = None  # [B, T, M, N, dx]


class GridBackendBase(Protocol):
    def generate(self, batch_ctx: Dict[str, torch.Tensor]) -> SubnodeBatch:
        ...


def make_backend(name: str, **kwargs) -> GridBackendBase:
    name_lc = name.lower()
    if name_lc == "voxel":
        from .voxel_backend import VoxelGridBackend

        return VoxelGridBackend(**kwargs)
    if name_lc == "amr":
        from .amr_backend import AmrGridBackend

        return AmrGridBackend(**kwargs)
    raise ValueError(f"Unknown subnode backend '{name}'. Expected one of ['voxel', 'amr'].")

