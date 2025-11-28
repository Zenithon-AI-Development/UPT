import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import matplotlib

# use non-interactive backend (safe on servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TimingDict = Dict[str, float]
Record = Dict[str, Dict]

# Top-level timing keys we expect (wall-clock update + coarse phases)
TOP_LEVEL_KEYS = [
    "time/update_total",
    "time/data_loading",
    "time/forward",
    "time/loss",
    "time/backward",
    "time/optim_step",
]

# Forward sub-phase keys (times are in seconds, summed across micro-steps)
FORWARD_SUBDIVISION_KEYS = [
    "time/model/prepare",
    "time/model/conditioner",
    "time/model/encoder",
    "time/model/latent",
    "time/model/decoder",
]


def load_records(pkl_path: Path) -> List[Record]:
    """Load and return list of per-update timing records from a .pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list) and len(data) > 0, "Expected non-empty list of per-update records"
    return data


def extract_timings(record: Record) -> TimingDict:
    timings = record.get("timings", {})
    # normalize to float
    return {str(k): float(v) for k, v in timings.items()}


def compute_aggregate(
    records: List[Record],
    update_index: Optional[int] = None,
) -> TimingDict:
    if update_index is not None:
        assert 0 <= update_index < len(records), f"update_index out of range [0, {len(records) - 1}]"
        return extract_timings(records[update_index])

    # mean across all updates in file
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for rec in records:
        t = extract_timings(rec)
        for k, v in t.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
    means = {k: (sums[k] / max(1, counts[k])) for k in sums.keys()}
    return means


def _ensure_keys(source: TimingDict, keys: List[str]) -> List[float]:
    return [float(source.get(k, 0.0)) for k in keys]


def print_summary(aggregate: TimingDict, update_index: Optional[int]) -> None:
    header = f"Update index: {update_index}" if update_index is not None else "Mean across updates"
    print(header)
    # stable order: top-level first, then forward subdivisions (remaining keys alphabetical)
    printed_keys = set()
    for k in TOP_LEVEL_KEYS:
        print(f"{k:>24s} : {aggregate.get(k, 0.0):.6f} s")
        printed_keys.add(k)
    for k in FORWARD_SUBDIVISION_KEYS:
        print(f"{k:>24s} : {aggregate.get(k, 0.0):.6f} s")
        printed_keys.add(k)
    # print any other timing keys that might exist
    for k in sorted(k for k in aggregate.keys() if k not in printed_keys):
        print(f"{k:>24s} : {aggregate.get(k, 0.0):.6f} s")


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
    Resolve the path to updates_epoch{E}_time.pkl.
    - If pkl is provided, use it directly.
    - Else, 'run' should be a relative path like 'helmholtz_overfit/4bicdqj4'.
      The base will be <output_path>/ from static_config.yaml + run + '/profiling/timings'.
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
        # fallback to repo-relative default used in this project
        base = Path(__file__).resolve().parents[2] / "benchmarking" / "save"
    run_dir = base / run
    prof_dir = run_dir / "profiling" / "timings"
    if not prof_dir.exists():
        raise FileNotFoundError(f"{prof_dir} not found")
    if epoch is None:
        cands = sorted(prof_dir.glob("updates_epoch*_time.pkl"))
        if not cands:
            raise FileNotFoundError(f"No updates_epoch*_time.pkl in {prof_dir}")
        return cands[-1]
    return prof_dir / f"updates_epoch{epoch}_time.pkl"


def plot_summary(
    aggregate: TimingDict,
    output_dir: Path,
    update_index: Optional[int],
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name_token = f"update_{update_index}" if update_index is not None else "mean"

    bar_w = 0.2 # thinner bars (keep figure size fixed)

    # Figure 1: total vs stacked major phases
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # left: total update time
    total_update_time = float(aggregate.get("time/update_total", 0.0))
    axes[0].bar([0], [total_update_time], width=bar_w, color="#4C78B8")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["update_total"])
    axes[0].set_ylabel("seconds")
    axes[0].set_title("Total update time")
    # keep bar thin and centered with whitespace on sides
    axes[0].set_xlim(-0.5, 0.5)

    # right: stacked phases
    stacked_keys = ["time/data_loading", "time/forward", "time/loss", "time/backward", "time/optim_step"]
    stacked_vals = _ensure_keys(aggregate, stacked_keys)
    bottom = 0.0
    colors = ["#72B7B2", "#F5A623", "#D4505A", "#59A14F", "#8C6D31"]
    for key, val, color in zip(stacked_keys, stacked_vals, colors):
        axes[1].bar([0], [val], width=bar_w, bottom=bottom, label=key.replace("time/", ""), color=color)
        bottom += val
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["update_breakdown"])
    axes[1].set_ylabel("seconds")
    axes[1].legend(loc="upper right", fontsize=8)
    # keep bar thin and centered with whitespace on sides
    axes[1].set_xlim(-0.5, 0.5)

    fig1_path = output_dir / f"timings_update_breakdown_{name_token}.png"
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Figure 2: forward total vs stacked model subdivisions
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # left: total forward time
    total_forward_time = float(aggregate.get("time/forward", 0.0))
    axes2[0].bar([0], [total_forward_time], width=bar_w, color="#4C78B8")
    axes2[0].set_xticks([0])
    axes2[0].set_xticklabels(["forward_total"])
    axes2[0].set_ylabel("seconds")
    axes2[0].set_title("Total forward time")
    axes2[0].set_xlim(-0.5, 0.5)
    # right: stacked model subdivisions (expand prepare into sub-steps)
    prep_total = float(aggregate.get("time/model/prepare", 0.0))
    prep_graph = float(aggregate.get("time/model/prepare/graph", 0.0))
    prep_to_device = float(aggregate.get("time/model/prepare/to_device", 0.0))
    prep_other = max(0.0, prep_total - (prep_graph + prep_to_device))
    fwd_keys = []
    fwd_vals = []
    if prep_graph > 0:
        fwd_keys.append("time/model/prepare/graph")
        fwd_vals.append(prep_graph)
    if prep_to_device > 0:
        fwd_keys.append("time/model/prepare/to_device")
        fwd_vals.append(prep_to_device)
    if prep_other > 0:
        fwd_keys.append("time/model/prepare/other")
        fwd_vals.append(prep_other)
    for k in ["time/model/conditioner", "time/model/encoder", "time/model/latent", "time/model/decoder"]:
        fwd_keys.append(k)
        fwd_vals.append(float(aggregate.get(k, 0.0)))
    bottom = 0.0
    colors2 = ["#59A14F", "#EDC948", "#B07AA1", "#FF9DA6", "#8CD17D", "#F28E2B", "#4E79A7", "#E15759"]
    for idx, (key, val) in enumerate(zip(fwd_keys, fwd_vals)):
        color = colors2[idx % len(colors2)]
        axes2[1].bar([0], [val], width=bar_w, bottom=bottom, label=key.replace("time/model/", ""), color=color)
        bottom += val
    axes2[1].set_xticks([0])
    axes2[1].set_xticklabels(["model_breakdown"])
    axes2[1].set_ylabel("seconds")
    axes2[1].legend(loc="upper right", fontsize=8)
    axes2[1].set_xlim(-0.5, 0.5)

    fig2_path = output_dir / f"timings_forward_breakdown_{name_token}.png"
    fig2.savefig(fig2_path, dpi=150)
    plt.close(fig2)

    # Figure 3: encoder total vs stacked encoder internals
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    # left: total encoder time (coarse)
    total_encoder_time = float(aggregate.get("time/model/encoder", 0.0))
    axes3[0].bar([0], [total_encoder_time], width=bar_w, color="#4C78B8")
    axes3[0].set_xticks([0])
    axes3[0].set_xticklabels(["encoder_total"])
    axes3[0].set_ylabel("seconds")
    axes3[0].set_title("Total encoder time")
    axes3[0].set_xlim(-0.5, 0.5)
    # right: stacked encoder internals (all keys with prefix)
    enc_internal_items = [(k, v) for k, v in aggregate.items() if isinstance(k, str) and k.startswith("time/model/encoder/")]
    # stable order by key name
    enc_internal_items.sort(key=lambda kv: kv[0])
    bottom = 0.0
    # simple color cycle
    color_cycle = ["#72B7B2", "#F5A623", "#D4505A", "#9C755F", "#59A14F", "#8CD17D", "#B6992D", "#BAB0AC"]
    for idx, (k, v) in enumerate(enc_internal_items):
        label = k.replace("time/model/encoder/", "")
        axes3[1].bar([0], [float(v)], width=bar_w, bottom=bottom, label=label, color=color_cycle[idx % len(color_cycle)])
        bottom += float(v)
    axes3[1].set_xticks([0])
    axes3[1].set_xticklabels(["encoder_breakdown"])
    axes3[1].set_ylabel("seconds")
    if enc_internal_items:
        axes3[1].legend(loc="upper right", fontsize=8)
    axes3[1].set_xlim(-0.5, 0.5)

    fig3_path = output_dir / f"timings_encoder_breakdown_{name_token}.png"
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)

    return fig1_path, fig2_path, fig3_path


def main():
    parser = argparse.ArgumentParser(description="Analyse per-update timing .pkl files")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", type=str, help="Full path to updates_epoch{E}.pkl")
    src.add_argument("--run", type=str, help="Relative run path like 'helmholtz_overfit/4bicdqj4'")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load when using --run (default: latest)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--print", dest="print_mode", action="store_true", help="Print summary to stdout")
    mode_group.add_argument("--plot", dest="plot_mode", action="store_true", help="Save plots to disk")
    parser.add_argument("--update", type=int, default=None, help="Specific update index within file (0-based)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory for plots (default: src/time_analysis/plots/<run>_<epoch>)")
    args = parser.parse_args()

    # resolve pkl
    pkl_path = resolve_pkl_path(args.pkl, args.run, args.epoch)
    records = load_records(pkl_path)
    aggregate = compute_aggregate(records, update_index=args.update)

    if args.print_mode:
        print_summary(aggregate, update_index=args.update)
    else:
        # default: save under src/time_analysis/plots/<run_id>_<epochTag>/
        base_out = Path(__file__).resolve().parent / "plots"
        try:
            run_id = pkl_path.parents[2].name  # .../<run_id>/profiling/timings/updates_epochX_time.pkl
        except Exception:
            run_id = "run"
        epoch_tag = pkl_path.stem  # e.g., updates_epoch1_time
        if epoch_tag.endswith("_time"):
            epoch_tag = epoch_tag[:-5]
        default_out = base_out / f"{run_id}_{epoch_tag}"
        out_dir = Path(args.outdir).expanduser() if args.outdir is not None else default_out
        f1, f2, f3 = plot_summary(aggregate, out_dir, update_index=args.update)
        print(f"Saved: {f1}")
        print(f"Saved: {f2}")
        print(f"Saved: {f3}")


if __name__ == "__main__":
    main()


