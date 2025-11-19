import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple

import h5py
import torch

os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from quadtree_upt import QuadtreeSnapshotBuilder
from quadtree_upt.aggregators import (
    QuadtreeLocalAttentionAggregator,
    QuadtreeMaskedMLPAggregator,
)


def _discover_array_key(h5: h5py.File) -> str:
    queue = [h5]
    while queue:
        group = queue.pop(0)
        for key, value in group.items():
            if isinstance(value, h5py.Dataset) and value.ndim in (4, 5):
                return value.name
            if isinstance(value, h5py.Group):
                queue.append(value)
    raise KeyError("Could not find dataset with ndim 4 or 5 in HDF5 file")


def _canonize_axes(arr: torch.Tensor) -> torch.Tensor:
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(arr.shape)}")
    if arr.shape[-1] <= 128 and arr.shape[1] == arr.shape[2]:
        return arr
    if arr.shape[1] <= 128 and arr.shape[2] == arr.shape[3]:
        return arr.permute(0, 2, 3, 1)
    raise ValueError(f"Cannot canonize array with shape {tuple(arr.shape)}")


def load_zpinch_snapshot(args: argparse.Namespace) -> Tuple[torch.Tensor, Path]:
    split_dir = Path(args.root) / args.split
    files = sorted(split_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {split_dir}")
    file_path = files[args.file_index % len(files)]
    with h5py.File(file_path, "r") as h5:
        key = _discover_array_key(h5)
        dataset = h5[key]
        time_index = min(args.time_index, dataset.shape[0] - 1)
        frame = torch.from_numpy(dataset[time_index]).float()
    frame = _canonize_axes(frame.unsqueeze(0)).squeeze(0)
    return frame, file_path


def load_trl_snapshot(args: argparse.Namespace) -> Tuple[torch.Tensor, Path]:
    split_dir = Path(args.root)
    files = sorted(split_dir.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {split_dir}")
    file_path = files[args.file_index % len(files)]
    field = args.field
    time_idx = args.time_index
    field_idx = args.field_index
    with h5py.File(file_path, "r") as h5:
        if field == "density":
            data = h5["t0_fields/density"]
        elif field == "pressure":
            data = h5["t0_fields/pressure"]
        else:
            raise ValueError("Supported fields: density, pressure")
        sample = data[field_idx % data.shape[0]]
        time_idx = min(time_idx, sample.shape[0] - 1)
        slice_ = torch.from_numpy(sample[time_idx]).float()
    snapshot = slice_.unsqueeze(-1)
    return snapshot, file_path


def pad_to_square(snapshot: torch.Tensor) -> torch.Tensor:
    H, W, C = snapshot.shape
    side = max(H, W)
    padded = torch.zeros(side, side, C, dtype=snapshot.dtype, device=snapshot.device)
    padded[:H, :W] = snapshot
    return padded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug quadtree supernode sampling")
    parser.add_argument("--root", default="/home/workspace/flash/ZEN_WELL_train/64x64/data/")
    parser.add_argument("--split", default="train")
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--num-supernodes", type=int, default=2048)
    parser.add_argument("--max-children", type=int, default=32)
    parser.add_argument("--min-depth", type=int, default=2)
    parser.add_argument("--variance-threshold", type=float, default=1e-4)
    parser.add_argument("--gradient-threshold", type=float, default=1e-4)
    parser.add_argument("--dataset", choices=["zpinch", "trl"], default="zpinch")
    parser.add_argument("--field", default="density")
    parser.add_argument("--field-index", type=int, default=0)
    parser.add_argument("--square-pad", action="store_true")
    parser.add_argument("--aggregator", choices=["none", "mlp", "attention"], default="none")
    parser.add_argument("--agg-dim", type=int, default=128)
    parser.add_argument("--agg-heads", type=int, default=4)
    return parser


def load_snapshot(args: argparse.Namespace) -> Tuple[torch.Tensor, Path]:
    if args.dataset == "zpinch":
        snapshot, path = load_zpinch_snapshot(args)
    else:
        snapshot, path = load_trl_snapshot(args)
    if args.square_pad:
        snapshot = pad_to_square(snapshot)
    return snapshot, path


def main(args: argparse.Namespace) -> None:
    snapshot, file_path = load_snapshot(args)
    H, W, C = snapshot.shape
    print(f"Loaded frame from {file_path.name}: grid {H}x{W}, channels={C}")

    builder = QuadtreeSnapshotBuilder(
        max_depth=int(math.ceil(math.log2(max(H, W)))),
        min_depth=args.min_depth,
        variance_threshold=args.variance_threshold,
        gradient_threshold=args.gradient_threshold,
    )
    nodes = builder.build(snapshot)
    total_nodes = len(nodes)
    print(f"Quadtree leaf count: {total_nodes}")

    supernodes = builder.sample_supernodes(
        nodes,
        num_supernodes=args.num_supernodes,
        device=snapshot.device,
    )
    print(f"Sampled supernodes: {len(supernodes)}")

    all_nodes_tensor = QuadtreeSnapshotBuilder.nodes_to_tensors(nodes, device=snapshot.device)
    supernodes_tensor = QuadtreeSnapshotBuilder.nodes_to_tensors(
        supernodes,
        device=snapshot.device,
        target_len=args.num_supernodes,
    )

    assignments = QuadtreeSnapshotBuilder.assign_nodes_to_supernodes(
        all_nodes=all_nodes_tensor,
        supernodes=supernodes_tensor,
        max_children=args.max_children,
    )

    if args.aggregator != "none":
        sub_feats = assignments["features"].unsqueeze(0)
        sub_mask = assignments["mask"].unsqueeze(0)
        super_feats = supernodes_tensor["features"].unsqueeze(0)
        if args.aggregator == "mlp":
            aggregator = QuadtreeMaskedMLPAggregator(
                input_dim=sub_feats.shape[-1],
                output_dim=args.agg_dim,
            )
        else:
            aggregator = QuadtreeLocalAttentionAggregator(
                input_dim=sub_feats.shape[-1],
                output_dim=args.agg_dim,
                supernode_dim=super_feats.shape[-1],
                num_heads=args.agg_heads,
            )
        agg_output = aggregator(sub_feats, sub_mask, super_feats)
        print(f"Aggregator output shape: {agg_output.shape}")

    print(f"Supernode features shape: {supernodes_tensor['features'].shape}")
    print(f"Subnode features shape: {assignments['features'].shape}")
    print(f"Subnode mask shape: {assignments['mask'].shape}")
    mean_children = assignments["mask"].float().sum(dim=1).mean().item()
    max_children = assignments["mask"].float().sum(dim=1).max().item()
    print(f"Average active subnodes per supernode: {mean_children:.2f}")
    print(f"Max active subnodes per supernode (clipped to {args.max_children}): {max_children:.0f}")


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())

