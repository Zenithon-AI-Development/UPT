from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from slot_upt.slot_assignment import get_slot_positions
from slot_upt.amr.quadtree import (
    pad_to_power_of_k,
    quadtree_partition_parallel,
)
from .interfaces import SubnodeBatch


@dataclass
class AmrBackendConfig:
    call_mode: str = "inline"  # 'inline' or 'precomputed'
    k: int = 2
    min_size: int = 1
    max_size: int = 4
    nonzero_ratio_threshold: float = 0.99
    common_refine_threshold: float = 0.4
    integral_refine_threshold: float = 0.1
    vorticity_threshold: float = 0.5
    momentum_threshold: float = 0.5
    shear_threshold: float = 0.5
    condition_type: str = "grad"
    pad_fill_value: float = 0.0


class AmrGridBackend:
    """
    Converts AMR quadtree cells/patches into the slot-based SubnodeBatch interface.
    """

    def __init__(
        self,
        M: int,
        N: int,
        ndim: int = 2,
        config: Optional[AmrBackendConfig] = None,
    ):
        if M is None or N is None:
            raise ValueError("AmrGridBackend requires both M (supernodes) and N (slots per supernode).")
        self.M = int(M)
        self.N = int(N)
        self.ndim = int(ndim)
        self.config = config or AmrBackendConfig()
        self._grid_shape = self._compute_grid_shape(self.M, self.ndim)
        self._slot_positions_cache: Optional[torch.Tensor] = None

    @staticmethod
    def _compute_grid_shape(M: int, ndim: int) -> Tuple[int, ...]:
        if ndim == 2:
            m_x = int(M**0.5)
            m_y = max(M // max(m_x, 1), 1)
            while m_x * m_y < M:
                m_x += 1
                m_y = max(M // m_x, 1)
            return (m_x, m_y)
        if ndim == 3:
            m_x = int(round(M ** (1 / 3)))
            m_y = m_z = max(m_x, 1)
            while m_x * m_y * m_z < M:
                m_x += 1
                m_y = m_z = max(m_x, 1)
            return (m_x, m_y, m_z)
        raise ValueError(f"Unsupported ndim={ndim}.")

    def _ensure_slot_positions(self, device: torch.device) -> torch.Tensor:
        if self._slot_positions_cache is None or self._slot_positions_cache.device != device:
            self._slot_positions_cache = get_slot_positions(self.M, self.N, self.ndim, device=device)
        return self._slot_positions_cache

    def generate(self, batch_ctx: Dict[str, torch.Tensor]) -> SubnodeBatch:
        mode = self.config.call_mode.lower()
        if mode == "inline":
            return self._generate_inline(batch_ctx)
        if mode == "precomputed":
            return self._generate_from_precomputed(batch_ctx)
        raise ValueError(f"Unsupported AMR backend call_mode='{self.config.call_mode}'.")

    # -------------------------------------------------------------------------
    # Inline AMR execution
    # -------------------------------------------------------------------------
    def _generate_inline(self, batch_ctx: Dict[str, torch.Tensor]) -> SubnodeBatch:
        required = ["amr_inputs"]
        for key in required:
            if key not in batch_ctx:
                raise KeyError(f"Inline AMR backend expects '{key}' in batch context.")

        inputs = batch_ctx["amr_inputs"]  # [B, T, H, W, C]
        labels = batch_ctx.get("amr_labels", inputs)
        mask = batch_ctx.get("amr_mask", torch.ones_like(inputs[..., :1]))
        feature_field = batch_ctx.get("amr_feature_field", inputs)

        if inputs.ndim != 5:
            raise ValueError(f"Expected amr_inputs with shape [B,T,H,W,C]; got {inputs.shape}")

        B, T, _, _, feature_dim = inputs.shape
        device = inputs.device

        spatial_shape_ctx = batch_ctx.get("spatial_shape")
        if spatial_shape_ctx is None:
            raise ValueError("AMR backend requires 'spatial_shape' in context.")
        spatial_shape = tuple(int(v) for v in (spatial_shape_ctx.tolist() if isinstance(spatial_shape_ctx, torch.Tensor) else spatial_shape_ctx))

        positions: List[torch.Tensor] = []
        features: List[torch.Tensor] = []
        levels: List[torch.Tensor] = []

        cfg = self.config

        for b in range(B):
            for t in range(T):
                inp = inputs[b, t]
                lbl = labels[b, t]
                msk = mask[b, t]
                feat = feature_field[b, t]

                orig_h, orig_w = inp.shape[0], inp.shape[1]

                inp_pad = pad_to_power_of_k(inp.unsqueeze(0), cfg.pad_fill_value, cfg.k)[0]
                lbl_pad = pad_to_power_of_k(lbl.unsqueeze(0), cfg.pad_fill_value, cfg.k)[0]
                feat_pad = pad_to_power_of_k(feat.unsqueeze(0), cfg.pad_fill_value, cfg.k)[0]
                msk_pad = pad_to_power_of_k(msk.unsqueeze(0), 1.0, cfg.k)[0]

                regions, patches, _ = quadtree_partition_parallel(
                    inp_pad,
                    lbl_pad,
                    msk_pad,
                    feat_pad,
                    nonzero_ratio_threshold=cfg.nonzero_ratio_threshold,
                    k=cfg.k,
                    min_size=cfg.min_size,
                    max_size=cfg.max_size,
                    common_refine_threshold=cfg.common_refine_threshold,
                    integral_refine_threshold=cfg.integral_refine_threshold,
                    vorticity_threshold=cfg.vorticity_threshold,
                    momentum_threshold=cfg.momentum_threshold,
                    shear_threshold=cfg.shear_threshold,
                    condition_type=cfg.condition_type,
                )

                if patches.numel() == 0:
                    positions.append(torch.empty(0, self.ndim, device=device))
                    features.append(torch.empty(0, feature_dim, device=device))
                    levels.append(torch.empty(0, device=device, dtype=torch.long))
                    continue

                patch_features = patches[..., :-3]
                depth = patches[..., -3].long()
                x1 = patches[..., -2]
                y1 = patches[..., -1]

                patch_features = patch_features.reshape(-1, patch_features.shape[-1])
                depth = depth.reshape(-1)
                x1 = x1.reshape(-1)
                y1 = y1.reshape(-1)

                grid_size = inp_pad.shape[0]
                cell_sizes = grid_size // (cfg.k ** (depth + 1))
                cell_sizes = cell_sizes.to(patch_features.dtype)
                center_x = x1 + 0.5 * cell_sizes
                center_y = y1 + 0.5 * cell_sizes
                valid_mask = (center_x < orig_h) & (center_y < orig_w)
                center_x = center_x[valid_mask]
                center_y = center_y[valid_mask]
                patch_features = patch_features[valid_mask]
                depth = depth[valid_mask]
                if center_x.numel() == 0:
                    positions.append(torch.empty(0, self.ndim, device=device))
                    features.append(torch.empty(0, feature_dim, device=device))
                    levels.append(torch.empty(0, device=device, dtype=torch.long))
                    continue

                pos = torch.stack([center_x / orig_h, center_y / orig_w], dim=-1)
                pos = torch.clamp(pos, 0.0, 1.0 - 1e-6)

                features.append(patch_features[:, :feature_dim])
                positions.append(pos[:, : self.ndim])
                levels.append(depth)

        return self._pack_to_slots(positions, features, levels, B, T, feature_dim, device, spatial_shape)

    # -------------------------------------------------------------------------
    # Precomputed mode
    # -------------------------------------------------------------------------
    def _generate_from_precomputed(self, batch_ctx: Dict[str, torch.Tensor]) -> SubnodeBatch:
        required = ["amr_cell_pos", "amr_cell_feats", "amr_cell_level"]
        for key in required:
            if key not in batch_ctx:
                raise KeyError(f"Precomputed AMR backend expects '{key}' in batch context.")

        pos_tensor = batch_ctx["amr_cell_pos"]  # [B, T, P, dx]
        feat_tensor = batch_ctx["amr_cell_feats"]  # [B, T, P, C]
        level_tensor = batch_ctx["amr_cell_level"]  # [B, T, P]

        if pos_tensor.ndim != 4 or feat_tensor.ndim != 4 or level_tensor.ndim != 3:
            raise ValueError("Precomputed AMR tensors must have shapes [B,T,P,*].")

        B, T, _, feature_dim = feat_tensor.shape
        device = feat_tensor.device

        spatial_shape_ctx = batch_ctx.get("spatial_shape")
        if spatial_shape_ctx is None:
            raise ValueError("AMR backend (precomputed) requires 'spatial_shape' in context.")
        spatial_shape = tuple(int(v) for v in (spatial_shape_ctx.tolist() if isinstance(spatial_shape_ctx, torch.Tensor) else spatial_shape_ctx))

        positions = [pos_tensor[b, t] for b in range(B) for t in range(T)]
        features = [feat_tensor[b, t] for b in range(B) for t in range(T)]
        levels = [level_tensor[b, t].long() for b in range(B) for t in range(T)]

        return self._pack_to_slots(positions, features, levels, B, T, feature_dim, device, spatial_shape)

    # -------------------------------------------------------------------------
    # Slot packing
    # -------------------------------------------------------------------------
    def _pack_to_slots(
        self,
        positions: Sequence[torch.Tensor],
        features: Sequence[torch.Tensor],
        levels: Sequence[torch.Tensor],
        B: int,
        T: int,
        feature_dim: int,
        device: torch.device,
        spatial_shape: Tuple[int, int],
    ) -> SubnodeBatch:
        slot_positions = self._ensure_slot_positions(device)

        subnode_feats = torch.zeros(B, T, self.M, self.N, feature_dim, device=device, dtype=features[0].dtype if features else torch.float32)
        subnode_mask = torch.zeros(B, T, self.M, self.N, device=device, dtype=torch.bool)
        slot2cell = torch.full((B, T, self.M, self.N), -1, device=device, dtype=torch.long)
        subnode_level_tensor = torch.zeros(B, T, self.M, self.N, device=device, dtype=torch.long)
        subnode_pos_tensor = torch.zeros(B, T, self.M, self.N, self.ndim, device=device, dtype=torch.float32)

        grid_shape = self._grid_shape

        H, W = spatial_shape

        for b in range(B):
            for t in range(T):
                idx = b * T + t
                pos_bt = positions[idx]
                feat_bt = features[idx]
                lvl_bt = levels[idx]

                if pos_bt.numel() == 0:
                    continue

                pos_bt = pos_bt.to(device=device, dtype=torch.float32)
                feat_bt = feat_bt.to(device=device)
                lvl_bt = lvl_bt.to(device=device, dtype=torch.long)

                grid_idx = torch.floor(pos_bt * torch.tensor(grid_shape, device=device, dtype=torch.float32)).long()
                grid_idx = torch.clamp(grid_idx, min=0)
                for dim in range(self.ndim):
                    grid_idx[:, dim] = torch.clamp(grid_idx[:, dim], max=grid_shape[dim] - 1)

                if self.ndim == 2:
                    m_indices = grid_idx[:, 0] * grid_shape[1] + grid_idx[:, 1]
                else:
                    m_indices = (
                        grid_idx[:, 0] * (grid_shape[1] * grid_shape[2])
                        + grid_idx[:, 1] * grid_shape[2]
                        + grid_idx[:, 2]
                    )

                pix_scale = torch.tensor([H, W], device=device, dtype=torch.float32)
                pix_coords = pos_bt * pix_scale
                pix_coords = torch.clamp(pix_coords, min=0.0)
                pix_coords[:, 0] = torch.clamp(pix_coords[:, 0], max=H - 1e-6)
                pix_coords[:, 1] = torch.clamp(pix_coords[:, 1], max=W - 1e-6)
                h_idx = torch.clamp(pix_coords[:, 0].long(), 0, H - 1)
                w_idx = torch.clamp(pix_coords[:, 1].long(), 0, W - 1)
                flat_idx = h_idx * W + w_idx

                for m in range(self.M):
                    cell_mask = m_indices == m
                    assigned = torch.nonzero(cell_mask, as_tuple=False).squeeze(-1)
                    if assigned.numel() == 0:
                        continue

                    if self.ndim == 2:
                        m_x = (m // grid_shape[1]) % grid_shape[0]
                        m_y = m % grid_shape[1]
                        voxel_center = torch.tensor(
                            [(m_x + 0.5) / grid_shape[0], (m_y + 0.5) / grid_shape[1]],
                            device=device,
                            dtype=torch.float32,
                        )
                    else:
                        m_x = (m // (grid_shape[1] * grid_shape[2])) % grid_shape[0]
                        m_y = (m // grid_shape[2]) % grid_shape[1]
                        m_z = m % grid_shape[2]
                        voxel_center = torch.tensor(
                            [
                                (m_x + 0.5) / grid_shape[0],
                                (m_y + 0.5) / grid_shape[1],
                                (m_z + 0.5) / grid_shape[2],
                            ],
                            device=device,
                            dtype=torch.float32,
                        )

                    dists = torch.norm(pos_bt[assigned, : self.ndim] - voxel_center.unsqueeze(0), dim=1)
                    sorted_idx = assigned[torch.argsort(dists)]

                    num_to_assign = min(sorted_idx.numel(), self.N)
                    for slot_idx in range(num_to_assign):
                        n = slot_idx
                        cell_idx = sorted_idx[slot_idx]
                        subnode_feats[b, t, m, n] = feat_bt[cell_idx]
                        subnode_mask[b, t, m, n] = True
                        slot2cell[b, t, m, n] = flat_idx[cell_idx].item()
                        subnode_level_tensor[b, t, m, n] = lvl_bt[cell_idx]
                        subnode_pos_tensor[b, t, m, n] = pos_bt[cell_idx, : self.ndim]

        return SubnodeBatch(
            subnode_feats=subnode_feats,
            subnode_mask=subnode_mask,
            slot2cell=slot2cell,
            slot_positions=slot_positions,
            subnode_level=subnode_level_tensor,
            subnode_pos=subnode_pos_tensor,
        )

