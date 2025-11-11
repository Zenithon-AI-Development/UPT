import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from kappadata.wrappers import ModeWrapper

from datasets.collators.cfd_simformer_collator import CfdSimformerCollator

from ..snapshot_builder import QuadtreeNode, QuadtreeSnapshotBuilder


class ZpinchQuadtreeCollator(CfdSimformerCollator):
    def __init__(
        self,
        num_supernodes: int,
        grid_size: Optional[int] = None,
        quadtree_max_depth: Optional[int] = None,
        quadtree_min_depth: int = 0,
        quadtree_variance_threshold: float = 1e-3,
        quadtree_gradient_threshold: float = 1e-3,
        quadtree_min_cell_size: Optional[int] = None,
        max_nodes_per_supernode: Optional[int] = None,
        region_overlap: float = 0.05,
        capacity_margin: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_supernodes=None, **kwargs)
        self.num_supernodes = num_supernodes
        self.grid_size = grid_size
        self.quadtree_max_depth = quadtree_max_depth
        self.quadtree_min_depth = quadtree_min_depth
        self.quadtree_variance_threshold = quadtree_variance_threshold
        self.quadtree_gradient_threshold = quadtree_gradient_threshold
        self.quadtree_min_cell_size = quadtree_min_cell_size
        self.user_supplied_max_children = max_nodes_per_supernode
        self.region_overlap = region_overlap
        self.capacity_margin = capacity_margin
        self._builder_cache: Dict[Tuple[int, int], QuadtreeSnapshotBuilder] = {}
        self.max_children: Optional[int] = None

        self.grid_rows, self.grid_cols = self._factor_grid(num_supernodes)
        self.supernode_centers = self._make_grid_centers(self.grid_rows, self.grid_cols)
        self.region_bounds = self._make_region_bounds(self.grid_rows, self.grid_cols, self.region_overlap)
        self.supernode_half_sizes = self._compute_region_half_sizes(self.grid_rows, self.grid_cols, self.region_overlap)

    @staticmethod
    def _factor_grid(n: int) -> Tuple[int, int]:
        rows = int(math.floor(math.sqrt(n)))
        while rows > 1 and n % rows != 0:
            rows -= 1
        cols = n // rows
        return rows, cols

    @staticmethod
    def _make_grid_centers(rows: int, cols: int) -> torch.Tensor:
        centers: List[Tuple[float, float]] = []
        for r in range(rows):
            cy = (r + 0.5) / rows
            for c in range(cols):
                cx = (c + 0.5) / cols
                centers.append((cx, cy))
        return torch.tensor(centers, dtype=torch.float32)

    @staticmethod
    def _make_region_bounds(rows: int, cols: int, overlap: float) -> List[Tuple[float, float, float, float]]:
        bounds: List[Tuple[float, float, float, float]] = []
        overlap_x = overlap / cols
        overlap_y = overlap / rows
        for r in range(rows):
            y0 = r / rows - overlap_y
            y1 = (r + 1) / rows + overlap_y
            y0 = max(y0, 0.0)
            y1 = min(y1, 1.0)
            for c in range(cols):
                x0 = c / cols - overlap_x
                x1 = (c + 1) / cols + overlap_x
                x0 = max(x0, 0.0)
                x1 = min(x1, 1.0)
                bounds.append((x0, x1, y0, y1))
        return bounds

    @staticmethod
    def _compute_region_half_sizes(rows: int, cols: int, overlap: float) -> torch.Tensor:
        overlap_x = overlap / cols
        overlap_y = overlap / rows
        half_w = 0.5 * (1.0 / cols + 2 * overlap_x)
        half_h = 0.5 * (1.0 / rows + 2 * overlap_y)
        return torch.tensor([half_w, half_h], dtype=torch.float32).unsqueeze(0).repeat(rows * cols, 1)

    def _get_builder(self, height: int, width: int) -> QuadtreeSnapshotBuilder:
        key = (height, width)
        builder = self._builder_cache.get(key)
        if builder is None:
            max_dim = max(height, width)
            default_max_depth = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else 0
            max_depth = self.quadtree_max_depth or default_max_depth
            builder = QuadtreeSnapshotBuilder(
                max_depth=max_depth,
                min_depth=self.quadtree_min_depth,
                variance_threshold=self.quadtree_variance_threshold,
                gradient_threshold=self.quadtree_gradient_threshold,
                min_cell_size=self.quadtree_min_cell_size,
            )
            self._builder_cache[key] = builder
            self._ensure_capacity(builder)
        return builder

    def _ensure_capacity(self, builder: QuadtreeSnapshotBuilder) -> None:
        if self.max_children is not None:
            return
        area = 1.0 / (self.grid_rows * self.grid_cols)
        base_capacity = area * (4 ** builder.max_depth)
        capacity = int(math.ceil(base_capacity * (1.0 + self.capacity_margin)))
        if self.user_supplied_max_children is not None:
            if self.user_supplied_max_children < capacity:
                import warnings

                warnings.warn(
                    "max_nodes_per_supernode was too small; overriding with computed capacity",
                    RuntimeWarning,
                )
            self.max_children = max(self.user_supplied_max_children, capacity)
        else:
            self.max_children = capacity

    def _region_index(self, center: torch.Tensor) -> int:
        cx, cy = center.tolist()
        for idx, (x0, x1, y0, y1) in enumerate(self.region_bounds):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return idx
        # Fallback: choose nearest center
        centers = self.supernode_centers
        distances = ((centers - center.unsqueeze(0)) ** 2).sum(dim=1)
        return int(distances.argmin().item())

    def _initialize_tensors(
        self,
        snapshot: torch.Tensor,
        feature_dim: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.max_children is None:
            raise RuntimeError("max_children is not initialized; ensure builder has been created before collation.")
        device = snapshot.device
        dtype = snapshot.dtype
        M = self.num_supernodes
        K = self.max_children
        super_features = torch.zeros(M, feature_dim, dtype=dtype, device=device)
        super_centers = self.supernode_centers.to(device=device, dtype=dtype).clone()
        super_half_sizes = self.supernode_half_sizes.to(device=device, dtype=dtype).clone()
        super_depths = torch.zeros(M, dtype=torch.long, device=device)
        super_bbox = torch.zeros(M, 4, dtype=torch.long, device=device)
        super_mask = torch.zeros(M, dtype=torch.bool, device=device)

        sub_features = torch.zeros(M, K, feature_dim, dtype=dtype, device=device)
        sub_rel_centers = torch.zeros(M, K, 2, dtype=dtype, device=device)
        sub_centers = torch.zeros(M, K, 2, dtype=dtype, device=device)
        sub_depths = torch.zeros(M, K, dtype=torch.long, device=device)
        sub_distances = torch.zeros(M, K, dtype=dtype, device=device)
        sub_mask = torch.zeros(M, K, dtype=torch.bool, device=device)
        sub_indices = torch.full((M, K), -1, dtype=torch.long, device=device)
        return (
            super_features,
            super_centers,
            super_half_sizes,
            super_depths,
            super_bbox,
            super_mask,
            sub_features,
            sub_rel_centers,
            sub_centers,
            sub_depths,
            sub_distances,
            sub_mask,
            sub_indices,
        )

    def _assign_nodes_to_regions(
        self,
        nodes: List[QuadtreeNode],
    ) -> List[List[int]]:
        assignments: List[List[int]] = [[] for _ in range(self.num_supernodes)]
        for idx, node in enumerate(nodes):
            region_idx = self._region_index(node.center)
            assignments[region_idx].append(idx)
        return assignments

    def _fill_supernode_tensors(
        self,
        nodes: List[QuadtreeNode],
        assignments: List[List[int]],
        tensors: Tuple[torch.Tensor, ...],
        snapshot: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        (
            super_features,
            super_centers,
            super_half_sizes,
            super_depths,
            super_bbox,
            super_mask,
            sub_features,
            sub_rel_centers,
            sub_centers,
            sub_depths,
            sub_distances,
            sub_mask,
            sub_indices,
        ) = tensors

        H, W, _ = snapshot.shape
        max_children = sub_features.shape[1]

        for region_idx, node_indices in enumerate(assignments):
            if not node_indices:
                continue
            super_mask[region_idx] = True
            node_indices.sort(key=lambda idx: (-nodes[idx].depth, nodes[idx].center[1].item(), nodes[idx].center[0].item()))
            count = min(len(node_indices), max_children)
            selected = node_indices[:count]

            child_feats = torch.stack([nodes[i].features for i in selected], dim=0).to(super_features)
            child_centers = torch.stack([nodes[i].center for i in selected], dim=0).to(super_features)
            child_depths = torch.tensor([nodes[i].depth for i in selected], dtype=torch.long, device=super_features.device)

            sub_features[region_idx, :count] = child_feats
            sub_centers[region_idx, :count] = child_centers
            sub_rel_centers[region_idx, :count] = child_centers - super_centers[region_idx].unsqueeze(0)
            sub_depths[region_idx, :count] = child_depths
            distances = torch.norm(sub_rel_centers[region_idx, :count], dim=-1)
            sub_distances[region_idx, :count] = distances
            sub_mask[region_idx, :count] = True
            sub_indices[region_idx, :count] = torch.tensor(selected, dtype=torch.long, device=sub_indices.device)

            super_features[region_idx] = child_feats.mean(dim=0)
            super_depths[region_idx] = int(min(nodes[i].depth for i in selected))

            x0, x1, y0, y1 = self.region_bounds[region_idx]
            super_bbox[region_idx] = torch.tensor(
                [
                    int(round(x0 * W)),
                    int(round(y0 * H)),
                    int(round(x1 * W)),
                    int(round(y1 * H)),
                ],
                dtype=torch.long,
                device=super_bbox.device,
            )

        return (
            super_features,
            super_centers,
            super_half_sizes,
            super_depths,
            super_bbox,
            super_mask,
            sub_features,
            sub_rel_centers,
            sub_centers,
            sub_depths,
            sub_distances,
            sub_mask,
            sub_indices,
        )

    def collate(self, batch, dataset_mode, ctx=None):
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)

        batch, ctx_list = zip(*batch)

        snapshots: List[torch.Tensor] = []
        nodes_per_sample: List[List[QuadtreeNode]] = []

        if self.grid_size is not None:
            if isinstance(self.grid_size, (tuple, list)) and len(self.grid_size) == 2:
                expected_shape = (int(self.grid_size[0]), int(self.grid_size[1]))
            else:
                side = int(self.grid_size)
                expected_shape = (side, side)
        else:
            expected_shape = None

        for sample in batch:
            target = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="target")
            num_points, num_channels = target.shape
            geometry = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="geometry2d")
            if expected_shape is not None:
                height, width = expected_shape
            elif geometry is not None and geometry.ndim == 3:
                height, width = geometry.shape[-2], geometry.shape[-1]
                if height * width != num_points or height <= 0 or width <= 0:
                    width = int(round(math.sqrt(num_points)))
                    height = int(num_points / width) if width > 0 else 0
            else:
                width = int(round(math.sqrt(num_points)))
                height = int(num_points / width) if width > 0 else 0
            if height * width != num_points or height <= 0 or width <= 0:
                raise ValueError(f"Cannot reshape snapshot of length {num_points} into a rectangular grid")

            snapshot = target.view(height, width, num_channels)
            builder = self._get_builder(snapshot.shape[0], snapshot.shape[1])
            nodes = builder.build(snapshot)

            snapshots.append(snapshot)
            nodes_per_sample.append(nodes)

        pairs = list(zip(batch, ctx_list))
        result, ctx_out = super().collate(pairs, dataset_mode, ctx)

        super_features_list = []
        super_centers_list = []
        super_half_sizes_list = []
        super_depths_list = []
        super_bbox_list = []
        super_mask_list = []
        sub_features_list = []
        sub_rel_centers_list = []
        sub_centers_list = []
        sub_depths_list = []
        sub_distances_list = []
        sub_mask_list = []
        sub_indices_list = []
        all_nodes_tensors = []

        for nodes, snapshot in zip(nodes_per_sample, snapshots):
            feature_dim = nodes[0].features.shape[-1] if nodes else 0
            tensors = self._initialize_tensors(snapshot, feature_dim)
            assignments = self._assign_nodes_to_regions(nodes)
            filled = self._fill_supernode_tensors(nodes, assignments, tensors, snapshot)

            (
                super_features,
                super_centers,
                super_half_sizes,
                super_depths,
                super_bbox,
                super_mask,
                sub_features,
                sub_rel_centers,
                sub_centers,
                sub_depths,
                sub_distances,
                sub_mask,
                sub_indices,
            ) = filled

            super_features_list.append(super_features)
            super_centers_list.append(super_centers)
            super_half_sizes_list.append(super_half_sizes)
            super_depths_list.append(super_depths)
            super_bbox_list.append(super_bbox)
            super_mask_list.append(super_mask)
            sub_features_list.append(sub_features)
            sub_rel_centers_list.append(sub_rel_centers)
            sub_centers_list.append(sub_centers)
            sub_depths_list.append(sub_depths)
            sub_distances_list.append(sub_distances)
            sub_mask_list.append(sub_mask)
            sub_indices_list.append(sub_indices)

            nodes_tensor = QuadtreeSnapshotBuilder.nodes_to_tensors(nodes, device=snapshot.device)
            all_nodes_tensors.append(nodes_tensor)

        ctx_out["quadtree_supernodes"] = {
            "features": torch.stack(super_features_list, dim=0),
            "centers": torch.stack(super_centers_list, dim=0),
            "half_sizes": torch.stack(super_half_sizes_list, dim=0),
            "depths": torch.stack(super_depths_list, dim=0),
            "bbox": torch.stack(super_bbox_list, dim=0),
            "mask": torch.stack(super_mask_list, dim=0),
        }
        ctx_out["quadtree_all_nodes"] = all_nodes_tensors
        ctx_out["quadtree_subnodes"] = {
            "features": torch.stack(sub_features_list, dim=0),
            "centers": torch.stack(sub_centers_list, dim=0),
            "relative_centers": torch.stack(sub_rel_centers_list, dim=0),
            "depths": torch.stack(sub_depths_list, dim=0),
            "distances": torch.stack(sub_distances_list, dim=0),
            "mask": torch.stack(sub_mask_list, dim=0),
            "indices": torch.stack(sub_indices_list, dim=0),
            "supernode_mask": torch.stack(super_mask_list, dim=0),
        }

        return result, ctx_out


__all__ = ["ZpinchQuadtreeCollator"]
