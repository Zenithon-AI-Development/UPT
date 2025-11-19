import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from kappadata.wrappers import ModeWrapper

from .cfd_simformer_collator import CfdSimformerCollator
from modules.quadtree import QuadtreeNode, QuadtreeSnapshotBuilder


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
            max_nodes_per_supernode: int = 64,
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
        self.max_nodes_per_supernode = max_nodes_per_supernode
        self._builder_cache: Dict[Tuple[int, int], QuadtreeSnapshotBuilder] = {}

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
        return builder

    def _snapshot_from_sample(
            self,
            sample: Tuple,
            dataset_mode: str,
            expected_shape: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        target = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="target")
        num_points, num_channels = target.shape
        geometry = ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="geometry2d")
        if expected_shape is not None:
            height, width = expected_shape
        elif geometry is not None and geometry.ndim == 3:
            height, width = geometry.shape[-2], geometry.shape[-1]
        else:
            width = int(round(math.sqrt(num_points)))
            height = int(num_points / width) if width > 0 else 0
            if height * width != num_points:
                raise ValueError(f"Cannot reshape snapshot of length {num_points} into rectangular grid")
        snapshot = target.view(height, width, num_channels)
        return snapshot

    def collate(self, batch, dataset_mode, ctx=None):
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)

        batch, ctx_list = zip(*batch)

        snapshots: List[torch.Tensor] = []
        all_nodes: List[List[QuadtreeNode]] = []
        supernodes: List[List[QuadtreeNode]] = []

        if self.grid_size is not None:
            if isinstance(self.grid_size, (tuple, list)) and len(self.grid_size) == 2:
                expected_shape = (int(self.grid_size[0]), int(self.grid_size[1]))
            else:
                side = int(self.grid_size)
                expected_shape = (side, side)
        else:
            expected_shape = None

        for sample in batch:
            snapshot = self._snapshot_from_sample(
                sample=sample,
                dataset_mode=dataset_mode,
                expected_shape=expected_shape,
            )
            builder = self._get_builder(snapshot.shape[0], snapshot.shape[1])
            nodes = builder.build(snapshot)
            sampled = builder.sample_supernodes(
                nodes,
                num_supernodes=self.num_supernodes,
                device=snapshot.device,
            )
            snapshots.append(snapshot)
            all_nodes.append(nodes)
            supernodes.append(sampled)

        pairs = list(zip(batch, ctx_list))
        result, ctx_out = super().collate(pairs, dataset_mode, ctx)

        all_nodes_tensors = [
            QuadtreeSnapshotBuilder.nodes_to_tensors(
                nodes,
                device=snapshot.device,
            )
            for nodes, snapshot in zip(all_nodes, snapshots)
        ]
        supernode_tensors = [
            QuadtreeSnapshotBuilder.nodes_to_tensors(
                nodes,
                device=snapshot.device,
                target_len=self.num_supernodes,
            )
            for nodes, snapshot in zip(supernodes, snapshots)
        ]

        centers = torch.stack([t["centers"] for t in supernode_tensors], dim=0)
        features = torch.stack([t["features"] for t in supernode_tensors], dim=0)
        depths = torch.stack([t["depths"] for t in supernode_tensors], dim=0)
        half_sizes = torch.stack([t["half_sizes"] for t in supernode_tensors], dim=0)
        bbox = torch.stack([t["bbox"] for t in supernode_tensors], dim=0)
        mask = torch.stack([t["mask"] for t in supernode_tensors], dim=0)

        subnode_assignments = [
            QuadtreeSnapshotBuilder.assign_nodes_to_supernodes(
                all_nodes=all_nodes_tensor,
                supernodes=supernode_tensor,
                max_children=self.max_nodes_per_supernode,
            )
            for all_nodes_tensor, supernode_tensor in zip(all_nodes_tensors, supernode_tensors)
        ]

        sub_features = torch.stack([entry["features"] for entry in subnode_assignments], dim=0)
        sub_centers = torch.stack([entry["centers"] for entry in subnode_assignments], dim=0)
        sub_rel_centers = torch.stack(
            [entry["relative_centers"] for entry in subnode_assignments],
            dim=0,
        )
        sub_depths = torch.stack([entry["depths"] for entry in subnode_assignments], dim=0)
        sub_distances = torch.stack([entry["distances"] for entry in subnode_assignments], dim=0)
        sub_mask = torch.stack([entry["mask"] for entry in subnode_assignments], dim=0)
        sub_indices = torch.stack([entry["indices"] for entry in subnode_assignments], dim=0)
        supernode_mask = torch.stack([entry["supernode_mask"] for entry in subnode_assignments], dim=0)

        ctx_out["quadtree_supernodes"] = {
            "centers": centers,
            "features": features,
            "depths": depths,
            "half_sizes": half_sizes,
            "bbox": bbox,
            "mask": mask,
        }
        ctx_out["quadtree_all_nodes"] = all_nodes_tensors
        ctx_out["quadtree_subnodes"] = {
            "features": sub_features,
            "centers": sub_centers,
            "relative_centers": sub_rel_centers,
            "depths": sub_depths,
            "distances": sub_distances,
            "mask": sub_mask,
            "indices": sub_indices,
            "supernode_mask": supernode_mask,
        }

        return result, ctx_out
