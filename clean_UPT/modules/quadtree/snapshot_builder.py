from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class QuadtreeNode:
    center: Tensor
    half_size: Tensor
    depth: int
    features: Tensor
    bbox: Tuple[int, int, int, int]


class QuadtreeSnapshotBuilder:
    def __init__(
            self,
            max_depth: int,
            min_depth: int = 0,
            variance_threshold: float = 1e-3,
            gradient_threshold: float = 1e-3,
            min_cell_size: Optional[int] = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if min_depth < 0 or min_depth > max_depth:
            raise ValueError("min_depth must satisfy 0 <= min_depth <= max_depth")

        self.max_depth = max_depth
        self.min_depth = min_depth
        self.variance_threshold = variance_threshold
        self.gradient_threshold = gradient_threshold
        self.min_cell_size = min_cell_size

    def build(self, snapshot: Tensor, mask: Optional[Tensor] = None) -> List[QuadtreeNode]:
        if snapshot.ndim == 4:
            snapshot = snapshot[-1]
        if snapshot.ndim == 2:
            n_cells = snapshot.shape[0]
            side = int(math.sqrt(n_cells))
            if side * side != n_cells:
                raise ValueError(
                    "Flattened snapshot must correspond to a square grid. "
                    f"Got {n_cells} cells."
                )
            snapshot = snapshot.view(side, side, snapshot.shape[-1])
        if snapshot.ndim != 3:
            raise ValueError("snapshot must be of shape [H, W, C]")

        H, W, _ = snapshot.shape

        mask_grid: Optional[Tensor]
        if mask is not None:
            if mask.ndim == 2:
                mask_grid = mask.unsqueeze(-1)
            elif mask.ndim == 1:
                mask_grid = mask.view(H, W, 1)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask_grid = mask
            else:
                raise ValueError("mask must be [H, W], [H*W], or [H, W, 1]")
            mask_grid = mask_grid.to(snapshot.dtype)
        else:
            mask_grid = None

        nodes: List[QuadtreeNode] = []

        def _should_subdivide(x0: int, y0: int, width: int, height: int, depth: int) -> bool:
            if width <= 1 or height <= 1:
                return False
            if depth < self.min_depth:
                return True
            if depth >= self.max_depth:
                return False
            if self.min_cell_size is not None and max(width, height) <= self.min_cell_size:
                return False

            patch = snapshot[y0: y0 + height, x0: x0 + width]
            if mask_grid is not None:
                mask_patch = mask_grid[y0: y0 + height, x0: x0 + width]
                if mask_patch.sum() <= 1e-6:
                    return False
                patch = patch * mask_patch

            centered = patch - patch.mean(dim=(0, 1), keepdim=True)
            variance = centered.pow(2).mean().item()
            if variance > self.variance_threshold:
                return True

            if height <= 1 or width <= 1:
                return False
            grad_x = patch[1:, :-1] - patch[:-1, :-1]
            grad_y = patch[:-1, 1:] - patch[:-1, :-1]
            if grad_x.numel() == 0 and grad_y.numel() == 0:
                grad_mag = 0.0
            else:
                grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).mean().item()
            return grad_mag > self.gradient_threshold

        def _subdivide(x0: int, y0: int, width: int, height: int, depth: int) -> None:
            can_split = width > 1 and height > 1
            if can_split and _should_subdivide(x0, y0, width, height, depth):
                w_left = width // 2
                w_right = width - w_left
                h_top = height // 2
                h_bottom = height - h_top

                if w_left > 0 and w_right > 0 and h_top > 0 and h_bottom > 0:
                    x_offsets = (0, w_left)
                    y_offsets = (0, h_top)
                    widths = (w_left, w_right)
                    heights = (h_top, h_bottom)
                    for j in range(2):
                        for i in range(2):
                            child_w = widths[i]
                            child_h = heights[j]
                            if child_w <= 0 or child_h <= 0:
                                continue
                            _subdivide(
                                x0 + x_offsets[i],
                                y0 + y_offsets[j],
                                child_w,
                                child_h,
                                depth + 1,
                            )
                    return

            patch = snapshot[y0: y0 + height, x0: x0 + width]
            if mask_grid is not None:
                mask_patch = mask_grid[y0: y0 + height, x0: x0 + width]
                weight = mask_patch.sum()
                if weight > 1e-6:
                    mean_feat = (patch * mask_patch).sum(dim=(0, 1)) / weight
                else:
                    mean_feat = patch.mean(dim=(0, 1))
            else:
                mean_feat = patch.mean(dim=(0, 1))

            center = torch.tensor(
                [
                    (x0 + width / 2) / W,
                    (y0 + height / 2) / H,
                ],
                device=snapshot.device,
                dtype=snapshot.dtype,
            )
            half_size = torch.tensor(
                [
                    (width / W) / 2.0,
                    (height / H) / 2.0,
                ],
                device=snapshot.device,
                dtype=snapshot.dtype,
            )
            node = QuadtreeNode(
                center=center,
                half_size=half_size,
                depth=depth,
                features=mean_feat,
                bbox=(x0, y0, x0 + width, y0 + height),
            )
            nodes.append(node)

        _subdivide(0, 0, W, H, depth=0)
        return nodes

    @staticmethod
    def sample_supernodes(
        nodes: List[QuadtreeNode],
        num_supernodes: int,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> List[QuadtreeNode]:
        if len(nodes) <= num_supernodes:
            return nodes
        if device is None:
            device = nodes[0].features.device
        idxs = torch.randperm(len(nodes), generator=generator, device=device)[:num_supernodes]
        return [nodes[int(i)] for i in idxs]

    @staticmethod
    def nodes_to_tensors(
        nodes: List[QuadtreeNode],
        device: torch.device,
        target_len: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        if target_len is None:
            target_len = len(nodes)
        if target_len <= 0:
            raise ValueError("target_len must be positive")

        feature_dim = nodes[0].features.shape[-1] if nodes else 0
        dtype = nodes[0].features.dtype if nodes else torch.float32

        centers = torch.zeros(target_len, 2, dtype=dtype, device=device)
        features = torch.zeros(target_len, feature_dim, dtype=dtype, device=device)
        depths = torch.zeros(target_len, dtype=torch.long, device=device)
        half_sizes = torch.zeros(target_len, 2, dtype=dtype, device=device)
        bbox = torch.zeros(target_len, 4, dtype=torch.long, device=device)
        mask = torch.zeros(target_len, dtype=torch.bool, device=device)

        for idx, node in enumerate(nodes[:target_len]):
            centers[idx] = node.center.to(device=device, dtype=dtype)
            features[idx] = node.features.to(device=device, dtype=dtype)
            depths[idx] = int(node.depth)
            half_sizes[idx] = node.half_size.to(device=device, dtype=dtype)
            bbox[idx] = torch.tensor(node.bbox, dtype=torch.long, device=device)
            mask[idx] = True

        return {
            "centers": centers,
            "features": features,
            "depths": depths,
            "half_sizes": half_sizes,
            "bbox": bbox,
            "mask": mask,
        }

    @staticmethod
    def assign_nodes_to_supernodes(
        *,
        all_nodes: Dict[str, Tensor],
        supernodes: Dict[str, Tensor],
        max_children: int,
    ) -> Dict[str, Tensor]:
        device = all_nodes["centers"].device
        dtype = all_nodes["features"].dtype
        feature_dim = all_nodes["features"].shape[-1]
        total_supernodes = supernodes["centers"].shape[0]

        sub_features = torch.zeros(total_supernodes, max_children, feature_dim, dtype=dtype, device=device)
        sub_centers = torch.zeros(total_supernodes, max_children, 2, dtype=dtype, device=device)
        sub_relative_centers = torch.zeros(total_supernodes, max_children, 2, dtype=dtype, device=device)
        sub_depths = torch.zeros(total_supernodes, max_children, dtype=torch.long, device=device)
        sub_distances = torch.zeros(total_supernodes, max_children, dtype=dtype, device=device)
        sub_mask = torch.zeros(total_supernodes, max_children, dtype=torch.bool, device=device)
        sub_indices = torch.full((total_supernodes, max_children), -1, dtype=torch.long, device=device)

        super_mask = supernodes["mask"]
        active_super_idx = torch.nonzero(super_mask, as_tuple=False).flatten()
        if active_super_idx.numel() == 0:
            return {
                "features": sub_features,
                "centers": sub_centers,
                "relative_centers": sub_relative_centers,
                "depths": sub_depths,
                "distances": sub_distances,
                "mask": sub_mask,
                "indices": sub_indices,
                "supernode_mask": super_mask,
            }

        active_super_centers = supernodes["centers"][active_super_idx]
        all_centers = all_nodes["centers"]
        distances = torch.cdist(active_super_centers, all_centers)
        nearest = distances.argmin(dim=0)

        for local_idx, global_idx in enumerate(active_super_idx):
            node_idx = torch.nonzero(nearest == local_idx, as_tuple=False).flatten()
            if node_idx.numel() == 0:
                continue
            sorted_local = distances[local_idx, node_idx].argsort()
            node_idx = node_idx[sorted_local][:max_children]
            count = node_idx.numel()
            if count == 0:
                continue

            sub_mask[global_idx, :count] = True
            sub_indices[global_idx, :count] = node_idx
            sub_features[global_idx, :count] = all_nodes["features"][node_idx]
            sub_centers[global_idx, :count] = all_nodes["centers"][node_idx]
            sub_relative_centers[global_idx, :count] = (
                all_nodes["centers"][node_idx] - supernodes["centers"][global_idx].unsqueeze(0)
            )
            sub_depths[global_idx, :count] = all_nodes["depths"][node_idx]
            sub_distances[global_idx, :count] = distances[local_idx, node_idx]

        return {
            "features": sub_features,
            "centers": sub_centers,
            "relative_centers": sub_relative_centers,
            "depths": sub_depths,
            "distances": sub_distances,
            "mask": sub_mask,
            "indices": sub_indices,
            "supernode_mask": super_mask,
        }
