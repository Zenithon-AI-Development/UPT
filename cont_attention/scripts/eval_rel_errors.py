# scripts/eval_rel_errors.py
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from torch_geometric.nn import radius_graph

# UPT bits
from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs


# ---------------- helpers ----------------
class _MiniDC:
    """Minimal stand-in for DataContainer so conditioners can query dataset props."""
    def __init__(self, ds): self._ds = ds
    def get_dataset(self, *_, **__): return self._ds

def rel_l1(pred, target, eps=1e-12):
    num = torch.sum(torch.abs(pred - target))
    den = torch.sum(torch.abs(target)) + eps
    return (num / den).item()

def rel_l2(pred, target, eps=1e-12):
    num = torch.linalg.vector_norm((pred - target).reshape(pred.shape[0], -1), ord=2).sum()
    den = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), ord=2).sum() + eps
    return (num / den).item()

def _repo_root() -> Path:
    # this file is .../UPT/src/scripts/eval_rel_errors.py  -> repo root = parents[2]
    return Path(__file__).resolve().parents[2]

def _find_static_config_uri() -> Path:
    candidates = [
        _repo_root() / "static_config.yaml",
        _repo_root() / "src" / "static_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find static_config.yaml")

def _load_stage_hp(stage_id: str) -> dict:
    hp_path = _repo_root() / "benchmarking" / "save" / "stage1" / stage_id / "hp_resolved.yaml"
    if not hp_path.exists():
        raise FileNotFoundError(
            f"hp_resolved.yaml not found for stage_id={stage_id} at:\n  {hp_path}"
        )
    with open(hp_path, "r") as f:
        return yaml.safe_load(f)

def _find_component_checkpoints(stage_id: str, checkpoint: str) -> dict:
    """
    Return {component_name: Path} for files like:
      'cfd_simformer_model.encoder cp=best_model.loss.online.x_hat.E1 model.th'
    under benchmarking/save/stage1/<stage_id>/checkpoints/
    """
    ckpt_dir = _repo_root() / "benchmarking" / "save" / "stage1" / stage_id / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint folder not found:\n  {ckpt_dir}")

    pats = list(ckpt_dir.glob(f"*cp={checkpoint}*model.th"))
    if not pats:
        pats = [p for p in ckpt_dir.glob(f"*cp={checkpoint}*") if p.name.endswith("model.th")]
    if not pats:
        raise FileNotFoundError(f"No model checkpoints with cp={checkpoint} in:\n  {ckpt_dir}")

    comp2path = {}
    for p in pats:
        stem_before_cp = p.name.split(" cp=")[0]
        component = stem_before_cp.split(".")[-1]  # encoder/decoder/latent/conditioner/...
        comp2path[component] = p
    return comp2path

def _strip_module_prefix(sd: dict) -> dict:
    # remove leading 'module.' if present (from DDP)
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def _as_decoder_output_shape(shape_from_ds) -> tuple:
    """
    Perceiver-style decoders expect a 2-tuple: (flattened_non_channel, channels).
    We build shapes from the TRAIN dataset (non-rollout), so target is typically (None, F).
    """
    shp = list(shape_from_ds) if isinstance(shape_from_ds, (list, tuple)) else list(tuple(shape_from_ds))
    if len(shp) == 2:
        return (shp[0], shp[1])
    if len(shp) >= 2:
        num_channels = shp[-1]
        non_channel = int(np.prod([d for d in shp[:-1] if d is not None]))
        return (non_channel, int(num_channels))
    raise ValueError(f"Unsupported output shape from dataset: {shape_from_ds}")


# ---------------- making a single-sample CFD batch ----------------
def _make_single_sample_batch(ds, idx: int, device: str, radius_r=5.0, radius_max_nn=32, num_supernodes=None):
    """
    Build the exact positional-arg bundle that cfd_simformer_model.forward(...) expects,
    for **one** sequence.
    Returns (inputs, target).
    """
    # core fields from dataset
    x          = ds.getitem_x(idx)             # (N, inF)
    target     = ds.getitem_target(idx)        # (N, outF) or (N,outF,T)
    geometry2d = ds.getitem_geometry2d(idx)    # (2,H,W)
    timestep   = ds.getitem_timestep(idx)      # scalar long -> must be 1-D
    velocity   = ds.getitem_velocity(idx)      # scalar float -> must be 1-D
    mesh_pos   = ds.getitem_mesh_pos(idx)      # (N,2)
    query_pos  = ds.getitem_query_pos(idx)     # (N,2)
    mesh_edges = ds.getitem_mesh_edges(idx)    # None

    # --- make timestep/velocity 1-D batch vectors of length 1 ---
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep, dtype=torch.long)
    timestep = timestep.reshape(-1)            # -> 1-D
    if timestep.numel() == 0:
        timestep = torch.zeros(1, dtype=torch.long)
    elif timestep.numel() == 1:
        # keep shape (1,), not scalar
        timestep = timestep.view(1)

    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity, dtype=torch.float32)
    velocity = velocity.reshape(-1)
    if velocity.numel() == 0:
        velocity = torch.zeros(1, dtype=torch.float32)
    elif velocity.numel() == 1:
        velocity = velocity.view(1)

    # graph indexing for a single-sample "batch"
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx      = torch.zeros(N, dtype=torch.long)   # all nodes belong to sample 0
    unbatch_idx    = torch.zeros(Nq, dtype=torch.long)  # all query points to batch 0 
    unbatch_select = torch.zeros(1, dtype=torch.long)   # select batch 0

    # move to device
    to_dev = lambda t: (None if t is None else t.to(device, non_blocking=True))
    x              = to_dev(x)
    target         = to_dev(target)
    geometry2d     = to_dev(geometry2d)
    timestep       = to_dev(timestep)
    velocity       = to_dev(velocity)
    mesh_pos       = to_dev(mesh_pos)
    query_pos      = to_dev(query_pos)
    batch_idx      = to_dev(batch_idx)
    unbatch_idx    = to_dev(unbatch_idx)
    unbatch_select = to_dev(unbatch_select)
    
    # Add batch dimension to query_pos for perceiver
    query_pos = query_pos.unsqueeze(0)  # (Nq, 2) -> (1, Nq, 2)
    
    # Build mesh edges if None
    if mesh_edges is None:
        flow = "target_to_source" if num_supernodes is not None else "source_to_target"
        edge_index = radius_graph(
            x=mesh_pos, r=radius_r, batch=batch_idx, loop=True,
            max_num_neighbors=radius_max_nn, flow=flow
        )
        mesh_edges = edge_index.T  # transpose to expected format
    else:
        mesh_edges = to_dev(mesh_edges)

    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    return inputs, target

# def _make_single_sample_batch(ds, idx: int, device: str):
#     """
#     Build the exact positional-arg bundle that cfd_simformer_model.forward(...) expects,
#     for **one** sequence (no collator). Returns (inputs, target) where inputs is a tuple:
#       (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
#        mesh_edges, batch_idx, unbatch_idx, unbatch_select)
#     """
#     # core fields
#     x          = ds.getitem_x(idx)             # (N, inF)
#     target     = ds.getitem_target(idx)        # (N, outF) or (N,outF,T) in rollout settings
#     geometry2d = ds.getitem_geometry2d(idx)    # (2,H,W)
#     timestep   = ds.getitem_timestep(idx)      # scalar long
#     velocity   = ds.getitem_velocity(idx)      # scalar float
#     mesh_pos   = ds.getitem_mesh_pos(idx)      # (N,2)
#     query_pos  = ds.getitem_query_pos(idx)     # (N,2)
#     mesh_edges = ds.getitem_mesh_edges(idx)    # None (created on GPU in trainer), forward still expects an arg

#     # graph indexing for a single-sample "batch"
#     N = x.shape[0]
#     batch_idx      = torch.zeros(N, dtype=torch.long)          # all nodes belong to sample 0
#     unbatch_idx    = torch.arange(N, dtype=torch.long)         # identity "unbatch" indexing
#     unbatch_select = torch.ones(N, dtype=torch.bool)           # select all nodes

#     # move to device
#     to_dev = lambda t: (None if t is None else t.to(device, non_blocking=True))
#     x          = to_dev(x)
#     target     = to_dev(target)
#     geometry2d = to_dev(geometry2d)
#     timestep   = to_dev(timestep)
#     velocity   = to_dev(velocity)
#     mesh_pos   = to_dev(mesh_pos)
#     query_pos  = to_dev(query_pos)
#     mesh_edges = None  # keep as None
#     batch_idx      = to_dev(batch_idx)
#     unbatch_idx    = to_dev(unbatch_idx)
#     unbatch_select = to_dev(unbatch_select)

#     inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
#               mesh_edges, batch_idx, unbatch_idx, unbatch_select)
#     return inputs, target


# ---------------- main evaluate ----------------
@torch.no_grad()
def evaluate(stage_id: str,
             checkpoint: str = "best_model.loss.online.x_hat.E1",
             split: str = "test",
             device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    # 1) Load HP + static config (paths like training)
    hp = _load_stage_hp(stage_id)
    static = StaticConfig(uri=str(_find_static_config_uri()))

    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name=f"eval_{stage_id}",
        stage_id=stage_id,
        temp_path=static.temp_path,
    )

    if "datasets" not in hp or "train" not in hp["datasets"]:
        raise RuntimeError("Resolved HP does not contain a 'datasets.train' section.")
    train_kwargs = hp["datasets"]["train"]

    # 2) Choose an eval split.
    # Prefer the requested; else try common names; if only 'test_rollout' exists,
    # derive a plain 'test' by copying TRAIN kwargs but setting split='test'.
    ds_candidates = [split, "test", "valid", "val", "evaluation", "eval", "test_rollout"]
    available = [k for k in ds_candidates if k in hp["datasets"]]
    derived_from_train = False
    if available:
        ds_key = available[0]
        eval_kwargs = hp["datasets"][ds_key]
        if ds_key == "test_rollout":
            eval_kwargs = dict(train_kwargs)
            eval_kwargs["split"] = "test"
            derived_from_train = True
    else:
        eval_kwargs = dict(train_kwargs)
        eval_kwargs["split"] = "test"
        derived_from_train = True

    # Build datasets (use TRAIN for shapes; EVAL for actual evaluation)
    ds_train = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **train_kwargs,
    )
    ds_eval = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **eval_kwargs,
    )

    # 3) Build model (give it a tiny DataContainer so conditioners can read timesteps)
    try:
        input_shape  = ds_train.getshape_x()
        output_shape = _as_decoder_output_shape(ds_train.getshape_target())
    except Exception:
        # emergency fallback
        x0  = ds_train.getitem_x(0)
        y0  = ds_train.getitem_target(0)
        input_shape = tuple(x0.shape)
        y0 = torch.as_tensor(y0)
        output_shape = (int(np.prod(y0.shape[:-1])), int(y0.shape[-1]))

    mini_dc = _MiniDC(ds_train)
    model = model_from_kwargs(
        **hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=mini_dc,
    ).to(device).eval()

    # 4) Load per-component checkpoints for cp=...
    comp_ckpts = _find_component_checkpoints(stage_id, checkpoint)
    loaded_any = False
    for comp, path in comp_ckpts.items():
        if hasattr(model, comp):
            state = torch.load(path.as_posix(), map_location="cpu")
            sd = _strip_module_prefix(state.get("state_dict", state))
            getattr(model, comp).load_state_dict(sd, strict=False)
            loaded_any = True
    if not loaded_any and comp_ckpts:
        # Fallback: load the first file into the whole model
        path = next(iter(comp_ckpts.values()))
        state = torch.load(path.as_posix(), map_location="cpu")
        sd = _strip_module_prefix(state.get("state_dict", state))
        model.load_state_dict(sd, strict=False)

    # Get radius graph parameters from trainer config
    radius_r = hp.get("trainer", {}).get("radius_graph_r", 5.0)
    radius_max_nn = hp.get("trainer", {}).get("radius_graph_max_num_neighbors", 32)
    num_supernodes = None
    for key in ["num_supernodes", "vars"]:
        if key in hp and "num_supernodes" in hp[key]:
            num_supernodes = hp[key]["num_supernodes"]
            break
    if num_supernodes is None and "datasets" in hp and "train" in hp["datasets"]:
        collators = hp["datasets"]["train"].get("collators", [])
        if collators and len(collators) > 0:
            num_supernodes = collators[0].get("num_supernodes")

    # 5) Eval loop (batch size = 1; we hand-build CFD inputs)
    mse_sum = rel1_sum = rel2_sum = 0.0
    n = 0

    num_samples = len(ds_eval)
    for idx in range(num_samples):
        inputs, target = _make_single_sample_batch(ds_eval, idx, device, radius_r, radius_max_nn, num_supernodes)

        out = model(*inputs)
        if isinstance(out, dict):
            pred = None
            for key in ("x_hat", "pred", "y_hat", "out"):
                if key in out and torch.is_tensor(out[key]):
                    pred = out[key]; break
            if pred is None:
                pred = next((v for v in out.values() if torch.is_tensor(v)), None)
                if pred is None:
                    raise RuntimeError("Model output dict does not contain a prediction tensor.")
        else:
            pred = out

        # If eval_kwargs came from a rollout split, target could be (N,F,T).
        # For a one-step metric, compare first step.
        if pred.ndim == target.ndim - 1 and target.shape[-1] > 1:
            target_cmp = target[..., 0]
        else:
            target_cmp = target

        if pred.shape != target_cmp.shape:
            # last-ditch alignment if there is a small time/channel length mismatch
            if pred.ndim == target_cmp.ndim and pred.shape[-1] != target_cmp.shape[-1]:
                minL = min(pred.shape[-1], target_cmp.shape[-1])
                pred = pred[..., :minL]
                target_cmp = target_cmp[..., :minL]
            else:
                raise RuntimeError(f"Pred/target shape mismatch: {pred.shape} vs {target_cmp.shape}")

        mse_sum  += torch.mean((pred - target_cmp) ** 2).item()
        rel1_sum += rel_l1(pred, target_cmp)
        rel2_sum += rel_l2(pred, target_cmp)
        n += 1

    return {
        "mse": mse_sum / max(n, 1),
        "rel_l1": rel1_sum / max(n, 1),
        "rel_l2": rel2_sum / max(n, 1),
        "samples": n,
        "derived_eval_from_train": bool(derived_from_train),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_id", required=True, help="Folder under benchmarking/save/stage1/")
    ap.add_argument("--checkpoint", default="best_model.loss.online.x_hat.E1",
                    help="the cp=... token used in filenames")
    ap.add_argument("--split", default="test",
                    help="preferred dataset split key (script falls back smartly)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = evaluate(stage_id=args.stage_id,
                   checkpoint=args.checkpoint,
                   split=args.split,
                   device=device)
    print(f"[{args.stage_id}] cp={args.checkpoint} -> {res}")


if __name__ == "__main__":
    main()
