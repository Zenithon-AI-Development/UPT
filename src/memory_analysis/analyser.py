import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import matplotlib

# use non-interactive backend (safe on servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MemoryDict = Dict[str, float]
Record = Dict[str, Dict]

# Main-block keys (bytes)
MAIN_BLOCK_KEYS = [
    "mem/data_loading_bytes",
    "mem/forward_bytes",
    "mem/loss_bytes",
    "mem/backward_bytes",
    "mem/optim_step_bytes",
]

# Forward subdivisions (bytes)
FORWARD_SUBDIVISION_KEYS = [
    "mem/model/prepare_bytes",
    "mem/model/conditioner_bytes",
    "mem/model/encoder_bytes",
    "mem/model/latent_bytes",
    "mem/model/decoder_bytes",
]

UNITS = {
    "bytes": 1.0,
    "MiB": 1024.0 * 1024.0,
    "GiB": 1024.0 * 1024.0 * 1024.0,
}


def load_records(pkl_path: Path) -> List[Record]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list) and len(data) > 0, "Expected non-empty list of per-update records"
    return data


def extract_memory(record: Record) -> MemoryDict:
    mem = record.get("memory", {})
    # normalize to float
    return {str(k): float(v) for k, v in mem.items()}


def compute_aggregate(
    records: List[Record],
    update_index: Optional[int] = None,
) -> MemoryDict:
    if update_index is not None:
        assert 0 <= update_index < len(records), f"update_index out of range [0, {len(records) - 1}]"
        return extract_memory(records[update_index])
    # mean across all updates in file (bytes)
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        m = extract_memory(rec)
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    means = {k: (sums[k] / max(1, counts[k])) for k in sums.keys()}
    return means


def _read_output_base_from_static_config() -> Optional[Path]:
    """Try to read output_path from src/static_config.yaml to discover base save folder."""
    static_cfg = Path(__file__).resolve().parents[1] / "static_config.yaml"
    if not static_cfg.exists():
        return None
    try:
        with open(static_cfg, "r") as f:
            data = yaml.safe_load(f) or {}
        out = data.get("output_path", None)
        return Path(out).expanduser() if out else None
    except Exception:
        return None


def resolve_pkl_path(pkl: Optional[str], run: Optional[str], epoch: Optional[int]) -> Path:
    """
    Resolve the path to updates_epoch{E}_mem.pkl.
    - If pkl is provided, use it directly.
    - Else, 'run' should be a relative path like 'helmholtz_overfit/4bicdqj4'.
      The base will be <output_path>/ from static_config.yaml + run + '/profiling/memory'.
    """
    if pkl:
        p = Path(pkl).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        return p
    if not run:
        raise ValueError("Either --pkl or --run must be provided")
    base = _read_output_base_from_static_config()
    if base is None:
        base = Path(__file__).resolve().parents[2] / "benchmarking" / "save"
    run_dir = base / run
    prof_dir = run_dir / "profiling" / "memory"
    if not prof_dir.exists():
        raise FileNotFoundError(f"{prof_dir} not found")
    if epoch is None:
        cands = sorted(prof_dir.glob("updates_epoch*_mem.pkl"))
        if not cands:
            raise FileNotFoundError(f"No updates_epoch*_mem.pkl in {prof_dir}")
        return cands[-1]
    return prof_dir / f"updates_epoch{epoch}_mem.pkl"


def _bytes_to_units(values: List[float], units: str) -> List[float]:
    denom = float(UNITS.get(units, UNITS["MiB"]))
    if denom == 0:
        denom = 1.0
    return [float(v) / denom for v in values]


def _ensure_vals(source: MemoryDict, keys: List[str]) -> List[float]:
    return [float(source.get(k, 0.0)) for k in keys]


def print_summary(aggregate: MemoryDict, update_index: Optional[int], units: str = "MiB") -> None:
    header = f"Update index: {update_index}" if update_index is not None else "Mean across updates"
    print(header)
    printed = set()
    for k in MAIN_BLOCK_KEYS:
        v = aggregate.get(k, 0.0) / UNITS.get(units, UNITS["MiB"])
        print(f"{k:>28s} : {v:8.2f} {units}")
        printed.add(k)
    for k in FORWARD_SUBDIVISION_KEYS:
        v = aggregate.get(k, 0.0) / UNITS.get(units, UNITS["MiB"])
        print(f"{k:>28s} : {v:8.2f} {units}")
        printed.add(k)
    # prepare sub-keys
    for k in ["mem/model/prepare/to_device_bytes", "mem/model/prepare/graph_bytes"]:
        v = aggregate.get(k, 0.0) / UNITS.get(units, UNITS["MiB"])
        print(f"{k:>28s} : {v:8.2f} {units}")
        printed.add(k)
    # encoder internals
    for k in sorted(k for k in aggregate.keys() if k.startswith("mem/model/encoder/") and k not in printed):
        v = aggregate.get(k, 0.0) / UNITS.get(units, UNITS["MiB"])
        print(f"{k:>28s} : {v:8.2f} {units}")
        printed.add(k)


def plot_bars(
    ax,
    keys: List[str],
    values_bytes: List[float],
    title: str,
    units: str,
):
    xs = list(range(len(keys)))
    vals = _bytes_to_units(values_bytes, units)
    bar_w = 0.35
    ax.bar(xs, vals, width=bar_w, color="#4C78B8")
    ax.set_xticks(xs)
    ax.set_xticklabels([k.split("/")[-1].replace("_bytes", "") for k in keys], rotation=20, ha="right")
    ax.set_ylabel(units)
    ax.set_title(title)


def plot_summary(
    aggregate: MemoryDict,
    output_dir: Path,
    update_index: Optional[int],
    units: str = "MiB",
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_token = f"update_{update_index}" if update_index is not None else "mean"

    # Figure 1: main blocks (side-by-side)
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    main_vals = _ensure_vals(aggregate, MAIN_BLOCK_KEYS)
    plot_bars(ax1, MAIN_BLOCK_KEYS, main_vals, "Memory peaks: main blocks", units)
    fig1_path = output_dir / f"memory_main_blocks_{name_token}.png"
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Figure 2: forward subdivisions (include prepare sub-steps if present)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    fwd_keys = list(FORWARD_SUBDIVISION_KEYS)
    # ensure prepare subkeys are visible if present
    for extra in ["mem/model/prepare/to_device_bytes", "mem/model/prepare/graph_bytes"]:
        if aggregate.get(extra, 0.0) > 0.0 and extra not in fwd_keys:
            fwd_keys.append(extra)
    fwd_vals = _ensure_vals(aggregate, fwd_keys)
    plot_bars(ax2, fwd_keys, fwd_vals, "Memory peaks: forward subdivisions", units)
    fig2_path = output_dir / f"memory_forward_subdivisions_{name_token}.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    # Figure 3: encoder internals (any key with prefix)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    enc_items = [(k, v) for k, v in aggregate.items() if isinstance(k, str) and k.startswith("mem/model/encoder/")]
    enc_items.sort(key=lambda kv: kv[0])
    if enc_items:
        enc_keys = [k for k, _ in enc_items]
        enc_vals = [float(v) for _, v in enc_items]
    else:
        enc_keys = []
        enc_vals = []
    plot_bars(ax3, enc_keys, enc_vals, "Memory peaks: encoder internals", units)
    fig3_path = output_dir / f"memory_encoder_internals_{name_token}.png"
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)

    return fig1_path, fig2_path, fig3_path


def main():
    parser = argparse.ArgumentParser(description="Analyse per-update memory .pkl files")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", type=str, help="Full path to memory_epoch{E}.pkl")
    src.add_argument("--run", type=str, help="Relative run path like 'helmholtz_overfit/4bicdqj4'")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load when using --run (default: latest)")
    parser.add_argument("--units", type=str, choices=["bytes", "MiB", "GiB"], default="MiB", help="Units for display/plots")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--print", dest="print_mode", action="store_true", help="Print summary to stdout")
    mode_group.add_argument("--plot", dest="plot_mode", action="store_true", help="Save plots to disk")
    parser.add_argument("--update", type=int, default=None, help="Specific update index within file (0-based)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for plots (default: src/memory_analysis/plots/<run>_<epoch>)")
    args = parser.parse_args()

    # resolve pkl
    pkl_path = resolve_pkl_path(args.pkl, args.run, args.epoch)
    records = load_records(pkl_path)
    aggregate = compute_aggregate(records, update_index=args.update)

    if args.print_mode:
        print_summary(aggregate, update_index=args.update, units=args.units)
    else:
        # default: save under src/memory_analysis/plots/<run_id>_<epochTag>/
        base_out = Path(__file__).resolve().parent / "plots"
        try:
            run_id = pkl_path.parents[2].name  # .../<run_id>/profiling/memory/updates_epochX_mem.pkl
        except Exception:
            run_id = "run"
        epoch_tag = pkl_path.stem  # e.g., updates_epoch1_mem
        if epoch_tag.endswith("_mem"):
            epoch_tag = epoch_tag[:-4]
        default_out = base_out / f"{run_id}_{epoch_tag}"
        out_dir = Path(args.outdir).expanduser() if args.outdir is not None else default_out
        f1, f2, f3 = plot_summary(aggregate, out_dir, update_index=args.update, units=args.units)
        print(f"Saved: {f1}")
        print(f"Saved: {f2}")
        print(f"Saved: {f3}")


if __name__ == "__main__":
    main()


