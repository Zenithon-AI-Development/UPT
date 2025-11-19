#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from datasets.well_trl2d_dataset import WellTrl2dDataset


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu()
    print(
        f"{name:>16s}: shape={tuple(tensor.shape)} "
        f"min={tensor.min().item():+.4f} "
        f"max={tensor.max().item():+.4f} "
        f"mean={tensor.mean().item():+.4f} "
        f"std={tensor.std(unbiased=False).item():+.4f}"
    )


def main(args: argparse.Namespace) -> None:
    dataset = WellTrl2dDataset(
        split="train",
        num_input_timesteps=args.num_input,
        num_output_timesteps=args.num_output,
        norm="mean0std1_auto",
        clamp=0,
        clamp_mode="log",
        max_num_sequences=args.max_sequences,
    )
    idx = args.index

    base_sample = dataset._item(idx)
    in_shape = base_sample["input_fields"].shape
    _, H, W, C = in_shape

    print(f"Loaded WellTrl2dDataset with {len(dataset)} samples.")
    print(f"Feature channels: {C}, grid: ({H}, {W})")
    if hasattr(dataset, "mean") and hasattr(dataset, "std"):
        print("Channel stats:")
        print(f"  mean: {dataset.mean}")
        print(f"  std : {dataset.std}")

    diagnostics = {}

    mesh_pos = dataset.getitem_mesh_pos(idx)
    query_pos = dataset.getitem_query_pos(idx)
    x = dataset.getitem_x(idx)
    target = dataset.getitem_target(idx)
    target_t0 = dataset.getitem_target_t0(idx)
    geometry2d = dataset.getitem_geometry2d(idx)

    summarize_tensor("mesh_pos", mesh_pos)
    summarize_tensor("query_pos", query_pos)
    summarize_tensor("x", x)
    summarize_tensor("target", target)
    summarize_tensor("target_t0", target_t0)
    summarize_tensor("geometry2d", geometry2d)

    metadata = base_sample.get("metadata", {})
    field_names = metadata.get("field_names")
    print(f"Field names: {field_names}")

    diagnostics["mesh_pos"] = mesh_pos
    diagnostics["query_pos"] = query_pos
    diagnostics["x"] = x
    diagnostics["target"] = target
    diagnostics["target_t0"] = target_t0
    diagnostics["geometry2d"] = geometry2d
    diagnostics["mean"] = getattr(dataset, "mean", None)
    diagnostics["std"] = getattr(dataset, "std", None)
    diagnostics["field_names"] = field_names

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"trl2d_sample{idx}_diagnostics.pt"
    torch.save(diagnostics, out_path)
    print(f"Saved diagnostics to {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump TRL2D dataset diagnostics.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect.")
    parser.add_argument("--num-input", type=float, default=1, help="Number of input timesteps.")
    parser.add_argument("--num-output", type=int, default=1, help="Number of output timesteps.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=1,
        help="Limit on number of sequences to load for determinism.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarking/diagnostics/trl2d",
        help="Where to store the dumped tensors.",
    )
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

