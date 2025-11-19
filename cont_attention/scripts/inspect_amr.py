import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    if yaml is None:
        print(f"[warn] pyyaml not installed; skipping parse of {path}")
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[warn] failed reading yaml {path}: {e}")
        return {}


def discover_h5_files(data_root: Path) -> List[Path]:
    files: List[Path] = []
    if not data_root.exists():
        return files
    for suffix in ("*.h5", "*.hdf5", "*.hdf"):
        files.extend(sorted(data_root.rglob(suffix)))
    return files


def _sample_stats(ds: h5py.Dataset) -> Tuple[float, float, float]:
    try:
        nd = ds.ndim
        if nd == 0:
            arr = ds[()]
        elif nd == 1:
            n = min(ds.shape[0], 4096)
            arr = ds[:n]
        else:
            arr = ds[0]
        import numpy as np  # lazy import

        a = np.asarray(arr)
        a = a.astype("float32", copy=False)
        return float(a.min()), float(a.max()), float(a.mean())
    except Exception:
        return float("nan"), float("nan"), float("nan")


def summarize_h5(path: Path, compute_stats: bool = False, max_items: int = 64) -> List[str]:
    lines: List[str] = []
    try:
        with h5py.File(path, "r") as f:
            lines.append(f"File: {path}")

            pending = ["/"]
            visited = set()
            count = 0
            while pending and count < max_items:
                name = pending.pop(0)
                if name in visited:
                    continue
                visited.add(name)
                grp = f[name] if name != "/" else f
                if isinstance(grp, h5py.Group):
                    for k in grp.keys():
                        full = k if name == "/" else f"{name.rstrip('/')}/{k}"
                        obj = grp[k]
                        if isinstance(obj, h5py.Group):
                            lines.append(f"  [G] {full}")
                            pending.append(full)
                        else:
                            shape = getattr(obj, "shape", None)
                            dtype = getattr(obj, "dtype", None)
                            ds_line = f"  [D] {full} shape={shape} dtype={dtype}"
                            if compute_stats:
                                mn, mx, me = _sample_stats(obj)
                                ds_line += f" min={mn:.4g} max={mx:.4g} mean={me:.4g}"
                            lines.append(ds_line)
                        count += 1
                        if count >= max_items:
                            break
            if count >= max_items:
                lines.append(f"  ... truncated after {max_items} items ...")
    except Exception as e:
        lines.append(f"File: {path} (error opening file: {e})")
    return lines


def main():
    parser = argparse.ArgumentParser("Inspect AMR HDF5 dataset structure and quick stats")
    parser.add_argument("--root", type=str, default="/home/workspace/flash/64x64_amr", help="AMR dataset root (contains dataset_amr.yaml and data/)")
    parser.add_argument("--limit", type=int, default=3, help="Max number of HDF5 files to summarize")
    parser.add_argument("--stats", action="store_true", help="Compute quick min/max/mean on small slices")
    parser.add_argument("--max_items", type=int, default=64, help="Max items (groups+datasets) to list per file")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    yaml_path = root / "dataset_amr.yaml"
    data_dir = root / "data"

    print(f"Root: {root}")
    print(f"YAML: {yaml_path}   (exists={yaml_path.exists()})")
    print(f"Data dir: {data_dir} (exists={data_dir.exists()})")

    meta = read_yaml(yaml_path)
    if meta:
        print("\nMetadata (dataset_amr.yaml):")
        for k in ["dataset_name", "n_spatial_dims", "grid_type", "field_names", "n_files"]:
            if k in meta:
                print(f"- {k}: {meta[k]}")
        for k in ["n_trajectories_per_file", "n_steps_per_trajectory"]:
            if k in meta:
                try:
                    length = len(meta[k])
                except Exception:
                    length = "?"
                example = meta[k][:3] if isinstance(meta[k], list) else meta[k]
                print(f"- {k}: len={length} example={example}")

    files = discover_h5_files(data_dir)
    if not files:
        print("\nNo HDF5 files found under data/.")
        return

    print(f"\nFound {len(files)} HDF5 file(s). Showing first {min(args.limit, len(files))}:")
    for fp in files[: args.limit]:
        lines = summarize_h5(fp, compute_stats=args.stats, max_items=args.max_items)
        print("\n".join(lines))

    print("\nNotes:")
    print("- TRL2D pipeline expects fixed (H,W) grids; AMR varies per timestep.")
    print("- If datasets include per-step point clouds (e.g., position[T, N_t, 2] and fields[T, N_t, C]), use Lagrangian pipeline.")
    print("- Verify which arrays encode positions vs. fields to choose model inputs/targets.")


if __name__ == "__main__":
    main()


