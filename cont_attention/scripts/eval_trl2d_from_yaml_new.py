#!/usr/bin/env python3
import os
import argparse
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import radius_graph

# --- project imports (repo root, PYTHONPATH=src) ---
from utils.kappaconfig.util import get_stage_hp
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from configs.static_config import StaticConfig
from utils.logging_util import add_global_handlers
from utils.seed import set_seed
from datasets import dataset_from_kwargs
from utils.data_container import DataContainer
from models import model_from_kwargs

EVAL_MODE = "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

# ----------------- small utils -----------------
def to_dev_any(obj, device):
    if obj is None: return None
    if torch.is_tensor(obj): return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)): return type(obj)(to_dev_any(x, device) for x in obj)
    if isinstance(obj, dict): return {k: to_dev_any(v, device) for k, v in obj.items()}
    return obj

def unpack_mode_tuple(mode_str, tup):
    keys = mode_str.split()
    assert len(keys) == len(tup), f"Mode tuple length {len(tup)} != items {len(keys)} ({keys})"
    return {k: tup[i] for i, k in enumerate(keys)}

def safe_get(d, *keys, default=None):
    cur = d
    try:
        for k in keys:
            cur = cur[k] if not isinstance(k, int) else cur[k]
        return cur
    except Exception:
        return default

def derive_num_supernodes(stage_hp):
    v = safe_get(stage_hp, "vars", "num_supernodes")
    if v is not None: return v
    v = safe_get(stage_hp, "datasets", "train", "collators", 0, "num_supernodes")
    if v is not None: return v
    for split in ("train", "valid_rollout", "test_rollout", "test"):
        v = safe_get(stage_hp, "datasets", split, "collators", 0, "num_supernodes")
        if v is not None: return v
    return None

def derive_radius_params(stage_hp, args):
    r = safe_get(stage_hp, "trainer", "radius_graph_r", default=args.radius_graph_r)
    k = safe_get(stage_hp, "trainer", "radius_graph_max_num_neighbors",
                 default=args.radius_graph_max_num_neighbors)
    return r, k

def rel_l1(pred, tgt, eps=1e-9):
    return (pred - tgt).abs().sum() / (tgt.abs().sum() + eps)

def rel_l2(pred, tgt, eps=1e-9):
    return torch.linalg.vector_norm(pred - tgt) / (torch.linalg.vector_norm(tgt) + eps)

def parse_ckpt_T(ckpt_dir):
    try:
        cand = [f for f in os.listdir(ckpt_dir) if f.startswith("cfd_simformer_model.conditioner") and "model.th" in f]
        if not cand: return None
        path = os.path.join(ckpt_dir, sorted(cand)[-1])
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "timestep_embed" in sd:
            return sd["timestep_embed"].shape[0]
        for k in ("state_dict", "model", "weights"):
            if k in sd and "timestep_embed" in sd[k]:
                return sd[k]["timestep_embed"].shape[0]
    except Exception:
        pass
    return None

def clamp_timesteps_inplace(timestep, T_rows):
    if T_rows is None: return
    if torch.is_tensor(timestep):
        timestep.clamp_(min=0, max=T_rows - 1)

@torch.no_grad()
def prepare_kwargs(batch, device, radius_r, radius_max_nn, num_supernodes=None, make_graph_on_gpu=True):
    """Converts (tuple, ctx) from collate into kwargs expected by CfdSimformerModel.forward."""
    batch, ctx = batch
    items = unpack_mode_tuple(EVAL_MODE, batch)
    items = {k: to_dev_any(v, device) for k, v in items.items()}
    ctx = to_dev_any(ctx, device)

    kw = dict(
        x=items["x"],
        geometry2d=items["geometry2d"],
        timestep=items["timestep"],
        velocity=items["velocity"],
        mesh_pos=items["mesh_pos"],
        query_pos=items["query_pos"],
        batch_idx=ctx["batch_idx"],
        unbatch_idx=ctx["unbatch_idx"],
        unbatch_select=ctx["unbatch_select"],
        target=items["target"],
    )
    mesh_edges = items["mesh_edges"]
    if mesh_edges is None and make_graph_on_gpu:
        assert radius_r is not None and radius_max_nn is not None, \
            "Need radius_graph_r and radius_graph_max_num_neighbors to build mesh edges."
        flow = "target_to_source" if num_supernodes is not None else "source_to_target"
        edge_index = radius_graph(
            x=kw["mesh_pos"], r=radius_r, batch=kw["batch_idx"], loop=True,
            max_num_neighbors=radius_max_nn, flow=flow
        )
        kw["mesh_edges"] = edge_index.T
    else:
        kw["mesh_edges"] = mesh_edges
    return kw

def unwrap_to_kd_base(ds, max_hops=16):
    """Stop unwrapping when you reach the first object that no longer has .dataset/.root_dataset
    or when an object repeats (cycle/self-loop protection)."""
    cur = ds
    seen = {id(cur)}
    for _ in range(max_hops):
        nxt = getattr(cur, "dataset", None)
        if nxt is None:
            nxt = getattr(cur, "root_dataset", None)
        if nxt is None: break
        if id(nxt) in seen or nxt is cur: break
        cur = nxt
        seen.add(id(cur))
    return cur

def cuda_timer(enabled=True):
    """Use CUDA events + synchronize for accurate GPU timings. (Best practice for latency measurement.)"""
    if not enabled or not torch.cuda.is_available():
        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            @property
            def ms(self): return None
        return _Noop()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    class _Timer:
        def __enter__(self):
            torch.cuda.synchronize()
            start.record()
            return self
        def __exit__(self, *a):
            end.record()
            torch.cuda.synchronize()
            self._ms = start.elapsed_time(end)
            return False
        @property
        def ms(self): return self._ms
    return _Timer()

# --- build GT sequence for AR from the base dataset (no repo changes) ---
@torch.no_grad()
def build_gt_seq_for_first_sample(
    first_batch,                 # the batch you already collated (use the 1st sample)
    base_dataset,                # the unwrapped KDDataset you already computed
    max_T=101,                   # cap on timesteps to collect
    device="cuda",
):
    """
    Returns:
      gt_seq  : [Nq, C, T] tensor with ground-truth targets for the same case as first sample
      qp      : [Nq, 2] query positions (to sanity check length matches your rollout)
      meta    : dict with keys ('t_first','t_last','T','case_key')
    If not enough frames are found, raises a RuntimeError with an explanation.
    """
    # Get the first sample's per-sample fields from your collated batch
    (tup, ctx) = first_batch
    # unpack (x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target)
    x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target = tup
    # We’ll look only at sample 0 in the batch
    # geometry2d, timestep, velocity are batched; query_pos is [B, Nq, 2]
    qpos_0   = query_pos[0].to(device)
    geom_0   = geometry2d[0].to(device)
    t0       = int(timestep[0].item())

    # Build a consistent "case key" using cheap checksums (sum + numel)
    def _tensor_key(t):
        # Use fp32 for checksum stability
        t = t.detach().to(torch.float32, copy=False)
        return (int(t.numel()), float(t.sum().cpu().item()))
    key_geom = _tensor_key(geom_0)
    key_qpos = (int(qpos_0.shape[0]), float(qpos_0.sum().cpu().item()))
    case_key = (key_geom, key_qpos)

    # A ModeWrapper that returns exactly the fields we need, one sample at a time
    from kappadata.wrappers import ModeWrapper
    mw_seq = ModeWrapper(base_dataset, mode="mesh_pos query_pos geometry2d timestep target", return_ctx=False)

    frames = []   # list of (t, target_tensor) for this case
    Nq_ref = qpos_0.shape[0]
    C_ref  = None

    # walk the whole split once (cheap enough) and collect frames that match this case
    for idx in range(len(mw_seq)):
        mesh_pos_i, qpos_i, geom_i, t_i, target_i = mw_seq[idx]
        # quick shape/key checks (on CPU is fine)
        if int(qpos_i.shape[0]) != Nq_ref:
            continue
        if _tensor_key(geom_i) != key_geom:
            continue
        # query_pos checksum must match (same selection of query points)
        if (int(qpos_i.shape[0]), float(qpos_i.to(torch.float32).sum().item())) != key_qpos:
            continue
        # passed the key checks -> accept this frame
        t_i = int(t_i.item())
        if C_ref is None:
            C_ref = int(target_i.shape[1])
        frames.append((t_i, target_i))

    if len(frames) == 0:
        raise RuntimeError("Could not find any matching frames for the first sample's case.")

    # sort by timestep and clip to [0 .. max_T-1] range if needed
    frames.sort(key=lambda z: z[0])
    # detect contiguous sub-sequence that includes t0 (best effort)
    # find all frames with the same keys that cover [t0 .. t0+K]
    # We’ll just take frames from the smallest t up to at most max_T frames.
    frames = frames[:max_T]
    T = len(frames)
    if T < 2:
        raise RuntimeError(f"Found only {T} frame(s) for this case – need at least 2 for a rollout curve.")

    # stack to [T, Nq, C] then permute to [Nq, C, T]
    targets = torch.stack([f[1] for f in frames], dim=0)          # [T, Nq, C]  (because target_i is [Nq, C])
    targets = targets.to(device).permute(1, 2, 0).contiguous()    # [Nq, C, T]
    return targets, qpos_0, dict(t_first=frames[0][0], t_last=frames[-1][0], T=T, case_key=str(case_key))


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--rollout_max_T", type=int, default=100)
    ap.add_argument("--grid_sizes", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--radius_graph_r", type=float, default=5.0)
    ap.add_argument("--radius_graph_max_num_neighbors", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="enable autocast for eval/train timings")
    args = ap.parse_args()

    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = "src"

    # speed hygiene
    torch.backends.cudnn.benchmark = True

    add_global_handlers(log_file_uri=None)
    set_seed(0)
    print("set seed to 0")

    # -------- Static/paths + HPs --------
    static = StaticConfig(uri="static_config.yaml", datasets_were_preloaded=False)
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name="eval",
        stage_id="manual",
        temp_path=static.temp_path,
    )
    stage_hp = get_stage_hp(args.cfg, template_path="zztemplates",
                            testrun=False, minmodelrun=False, mindatarun=False, mindurationrun=False)

    num_supernodes = derive_num_supernodes(stage_hp)
    r_graph, k_graph = derive_radius_params(stage_hp, args)

    # -------- Build datasets --------
    dcp = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    datasets = {}
    for key, kwargs in stage_hp["datasets"].items():
        datasets[key] = dataset_from_kwargs(
            dataset_config_provider=dcp,
            path_provider=path_provider,
            **kwargs,
        )
    # train-like eval set so Tin*C agrees with training
    test_like_train = dict(stage_hp["datasets"]["test"])
    test_like_train["split"] = "test"
    test_like_train["num_input_timesteps"] = stage_hp["datasets"]["test"].get("num_input_timesteps", 4)
    datasets["eval_like_train"] = dataset_from_kwargs(
        dataset_config_provider=dcp,
        path_provider=path_provider,
        **test_like_train,
    )
    data_container = DataContainer(
        **datasets,
        num_workers=0,
        pin_memory=True,
        config_provider=None,
        seed=0,
    )

    ds_train, _ = data_container.get_dataset("train", mode="x")
    input_shape  = ds_train.getshape_x()
    output_shape = ds_train.getshape_target()

    # ---- Conditioner's T for construction (we’ll override with checkpoint value later if present)
    T_yaml = 100
    class _StubDs:
        def __init__(self, T): self._T = T
        def getdim_timestep(self): return self._T
    class _StubDC:
        def __init__(self, T): self._ds = _StubDs(T)
        def get_dataset(self, *args, **kwargs): return self._ds
    stub_dc = _StubDC(T_yaml)

    # -------- Model (constructed with stub DC, then swap real DC) --------
    model = model_from_kwargs(
        **stage_hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=stub_dc,
    ).to(args.device).eval()

    def _swap_dc(m, real_dc):
        if hasattr(m, "data_container"): m.data_container = real_dc
        for name in ("conditioner", "geometry_encoder", "encoder", "latent", "decoder"):
            sub = getattr(m, name, None)
            if sub is not None and hasattr(sub, "data_container"):
                sub.data_container = real_dc
    _swap_dc(model, data_container)

    # -------- Pick eval dataset & ModeWrapper + collate --------
    ds_test, _ = data_container.get_dataset("eval_like_train", mode="x")
    base = unwrap_to_kd_base(ds_test)

    from kappadata.wrappers import ModeWrapper
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data.dataloader import default_collate as dcollate

    mw_test = ModeWrapper(base, mode=EVAL_MODE, return_ctx=True)
    MODE_KEYS = EVAL_MODE.split()
    IDX = {k: i for i, k in enumerate(MODE_KEYS)}

    def cfd_eval_collate(samples):
        items_list, ctx_list = zip(*samples)

        mesh_pos_list = [items[IDX["mesh_pos"]] for items in items_list]
        mesh_lens = [len(t) for t in mesh_pos_list]
        mesh_pos = torch.cat(mesh_pos_list, dim=0)

        batch_idx = torch.empty(sum(mesh_lens), dtype=torch.long)
        s = 0
        for b, L in enumerate(mesh_lens):
            batch_idx[s:s+L] = b
            s += L

        x_list = [items[IDX["x"]] for items in items_list]
        for i, xi in enumerate(x_list): assert len(xi) == mesh_lens[i]
        x = torch.cat(x_list, dim=0)

        query_pos_list = [items[IDX["query_pos"]] for items in items_list]
        target_list    = [items[IDX["target"]]    for items in items_list]
        query_pos = pad_sequence(query_pos_list, batch_first=True)
        target    = torch.cat(target_list, dim=0)

        B = len(query_pos_list)
        max_q = max(len(q) for q in query_pos_list)
        unbatch_idx = torch.empty(max_q * B, dtype=torch.long)
        unbatch_select = []
        u = 0
        uid = 0
        for q in query_pos_list:
            L = len(q)
            unbatch_idx[u:u+L] = uid
            unbatch_select.append(uid); uid += 1; u += L
            pad = max_q - L
            if pad > 0:
                unbatch_idx[u:u+pad] = uid
                uid += 1; u += pad
        unbatch_select = torch.tensor(unbatch_select, dtype=torch.long)

        edges_any = True
        mesh_edges_list = []
        off = 0
        for i, items in enumerate(items_list):
            ei = items[IDX["mesh_edges"]]
            if ei is None or ei.numel() == 0:
                edges_any = False
                break
            mesh_edges_list.append(ei + off)
            off += mesh_lens[i]
        mesh_edges = torch.cat(mesh_edges_list, dim=0) if edges_any else None

        geometry2d = dcollate([items[IDX["geometry2d"]] for items in items_list])
        timestep   = dcollate([items[IDX["timestep"]]   for items in items_list])
        velocity   = dcollate([items[IDX["velocity"]]   for items in items_list])

        batch_tuple = (x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target)
        ctx = {"batch_idx": batch_idx, "unbatch_idx": unbatch_idx, "unbatch_select": unbatch_select}
        return (batch_tuple, ctx)

    dl_test = DataLoader(
        mw_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=cfd_eval_collate,
    )

    # -------- Load submodule checkpoints (best -> last) --------
    def _load_sd(path):
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "weights")):
            for k in ("state_dict", "model", "weights"):
                if k in sd:
                    sd = sd[k]; break
        return sd

    loaded = []
    for name, stem in {
        "conditioner": "cfd_simformer_model.conditioner",
        "encoder":     "cfd_simformer_model.encoder",
        "latent":      "cfd_simformer_model.latent",
        "decoder":     "cfd_simformer_model.decoder",
    }.items():
        candidates = [f for f in os.listdir(args.ckpt_dir) if f.startswith(stem) and "model.th" in f]
        best = [f for f in candidates if "best_" in f or "best_model" in f]
        pick = best or [f for f in candidates if "last" in f] or candidates
        if not pick: continue
        path = os.path.join(args.ckpt_dir, sorted(pick)[-1])
        getattr(model, name).load_state_dict(_load_sd(path), strict=False)
        loaded.append(path)
    if loaded:
        print("[ckpt] loaded:\n  " + "\n  ".join(loaded))
    else:
        print("[ckpt] WARNING: no submodule checkpoints found in", args.ckpt_dir)

    T_ckpt = parse_ckpt_T(args.ckpt_dir)

    # --------------- (A) One-step metrics on test ---------------
    print(f"[sanity] len(test) = {len(mw_test)}")
    mse_sum = r1_sum = r2_sum = 0.0
    n = 0
    t0 = time.perf_counter()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (args.amp and args.device.startswith("cuda") and torch.cuda.is_available()) else nullcontext()
    with torch.no_grad(), amp_ctx:
        for batch in dl_test:
            kw = prepare_kwargs(
                batch=batch, device=args.device, radius_r=r_graph, radius_max_nn=k_graph,
                num_supernodes=num_supernodes, make_graph_on_gpu=True,
            )
            clamp_timesteps_inplace(kw["timestep"], T_ckpt)
            out = model(**kw)
            pred, tgt = out["x_hat"], kw["target"]
            mse_sum += torch.mean((pred - tgt)**2).item()
            r1_sum  += rel_l1(pred, tgt).item()
            r2_sum  += rel_l2(pred, tgt).item()
            n += 1
    wall = time.perf_counter() - t0
    print("\n=== One-step metrics (test) ===")
    print(f"MSE     : {mse_sum / max(n,1):.6f}")
    print(f"rel-L1  : {r1_sum  / max(n,1):.6f}")
    print(f"rel-L2  : {r2_sum  / max(n,1):.6f}")
    print(f"Eval wall-clock: {wall:.2f}s for {n} batches")

    # grab a real batch ONCE for sizes (avoid the dummy TypeError)
    first_real = next(iter(dl_test))
    first_real_kw = prepare_kwargs(
        first_real, device=args.device, radius_r=r_graph, radius_max_nn=k_graph,
        num_supernodes=num_supernodes, make_graph_on_gpu=True
    )
    Tin = test_like_train["num_input_timesteps"]
    C_out = first_real_kw["target"].shape[1]
    C_in  = first_real_kw["x"].shape[1] // Tin  # per-timestep channels

    # --------------- (B) Rollout error growth (autoregressive) ---------------
    print("\n=== Autoregressive rollout: error growth ===")
    if not hasattr(model, "rollout"):
        print("Skip: model has no .rollout(...)")
    else:
        # Take one *existing* collated batch
        first_batch = next(iter(dl_test))
        kw = prepare_kwargs(
            batch=first_batch, device=args.device,
            radius_r=r_graph, radius_max_nn=k_graph,
            num_supernodes=num_supernodes, make_graph_on_gpu=True,
        )
        clamp_timesteps_inplace(kw["timestep"], T_ckpt)

        # Run model rollout (returns [B*Nq, C, T]; your code uses B>1, but we'll plot for sample 0)
        with torch.no_grad():
            xhats = model.rollout(
                x=kw["x"], geometry2d=kw["geometry2d"], velocity=kw["velocity"],
                mesh_pos=kw["mesh_pos"], query_pos=kw["query_pos"], mesh_edges=kw["mesh_edges"],
                batch_idx=kw["batch_idx"], unbatch_idx=kw["unbatch_idx"], unbatch_select=kw["unbatch_select"],
                num_rollout_timesteps=min(args.rollout_max_T, T_ckpt or args.rollout_max_T),
                mode="image", intermediate_results=True,
            )  # -> [B*Nq, C, T_pred]

        # Build GT sequence for the SAME case as the first sample (no repo changes required)
        ds_test, _ = data_container.get_dataset("eval_like_train", mode="x")
        base = unwrap_to_kd_base(ds_test)
        try:
            gt_seq, qpos0, meta = build_gt_seq_for_first_sample(first_batch, base, max_T=xhats.shape[-1], device=args.device)
            # take only the first sample's Nq from xhats
            # Our collate stacked B samples; select the first block of Nq rows
            Nq = qpos0.shape[0]
            xhats0 = xhats[:Nq]                     # [Nq, C, T_pred]
            T_use  = min(xhats0.shape[-1], gt_seq.shape[-1])

            xhats0 = xhats0[..., :T_use]
            gt_use = gt_seq[..., :T_use]

            # timestep curves
            mse_t = ((xhats0 - gt_use) ** 2).mean(dim=(0,1))
            l1_t  = (xhats0 - gt_use).abs().sum(dim=(0,1)) / (gt_use.abs().sum(dim=(0,1)) + 1e-9)
            l2_t  = torch.linalg.vector_norm(xhats0 - gt_use, dim=(0,1)) / \
                    (torch.linalg.vector_norm(gt_use, dim=(0,1)) + 1e-9)

            print(f"Using case with timesteps [{meta['t_first']}..{meta['t_last']}] (T={meta['T']}).")
            print("t, MSE, relL1, relL2")
            for t in range(T_use):
                print(f"{t+1:3d}, {mse_t[t].item():.6f}, {l1_t[t].item():.6f}, {l2_t[t].item():.6f}")
        except Exception as e:
            print(f"Skip: could not assemble GT sequence for AR ({e}).")

    # --------------- (C) Inference latency (per-batch, mesh graph prebuilt) ---------------
    print("\n=== Inference latency (per batch) ===")
    model.eval()
    # Precompute a list of kw (with mesh_edges built once) and reuse them for timing
    timing_batches = []
    for i, batch in enumerate(dl_test):
        if i >= 10: break
        kw = prepare_kwargs(batch, device=args.device, radius_r=r_graph, radius_max_nn=k_graph,
                            num_supernodes=num_supernodes, make_graph_on_gpu=True)
        timing_batches.append(kw)
    # Warmup
    with torch.inference_mode(), amp_ctx:
        for kw in timing_batches[:3]:
            _ = model(**kw)
    # Timed
    lat_ms = []
    for kw in timing_batches:
        with torch.inference_mode(), amp_ctx, cuda_timer(enabled=(args.device.startswith("cuda") and torch.cuda.is_available())) as t:
            _ = model(**kw)
        lat_ms.append(t.ms if t and t.ms is not None else 0.0)
    if lat_ms:
        print(f"Avg latency: {sum(lat_ms)/len(lat_ms):.2f} ms  |  min: {min(lat_ms):.2f}  max: {max(lat_ms):.2f}  over {len(lat_ms)} batches")
    else:
        print("CPU-only timing: very small models may read as ~0ms.")

    # --------------- (D) Training throughput (bs=4, reuse same kw) ---------------
    print("\n=== Training throughput (bs=4) ===")
    model.train()
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # use the first timing batch for repeatable timing
    kw_train = timing_batches[0]
    # Warmups
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device.startswith("cuda") and torch.cuda.is_available())
    for _ in range(3):
        with amp_ctx:
            out = model(**kw_train)
            loss = mse(out["x_hat"], kw_train["target"])
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        else:
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    # Timed
    it_lat = []
    iters = 10
    for _ in range(iters):
        with cuda_timer(enabled=(args.device.startswith("cuda") and torch.cuda.is_available())) as t, amp_ctx:
            out = model(**kw_train)
            loss = mse(out["x_hat"], kw_train["target"])
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            else:
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        it_lat.append(t.ms if t and t.ms is not None else 0.0)
    model.eval()
    if it_lat:
        avg_ms = sum(it_lat)/len(it_lat)
        sps = (1000.0/avg_ms) * args.batch_size
        print(f"Avg iter time: {avg_ms:.2f} ms   -> ~{sps:.2f} samples/sec (bs={args.batch_size})")
    else:
        print("CPU-only timing: use wall-clock if needed.")

    # --------------- (E) Speed vs grid size (dummy data, velocity shape fixed) ---------------
    print("\n=== Speed vs grid size (dummy, bs=4) ===")
    # Reuse C_in/C_out inferred from a real batch above
    def make_dummy_batch(grid, bs=4, chans_in=C_in, chans_out=C_out, Tin=test_like_train["num_input_timesteps"]):
        # mesh_pos grid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, grid),
            torch.linspace(-1, 1, grid),
            indexing="ij"
        )
        pts = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [N,2]
        N = pts.shape[0]
        mesh_pos = pts.repeat(bs, 1)
        batch_idx = torch.arange(bs, dtype=torch.long).repeat_interleave(N)
        # x: [bs*N, Tin*C]
        x = torch.randn(bs*N, Tin*chans_in, dtype=torch.float32)
        # query_pos and unbatch helpers
        query_pos = pts.unsqueeze(0).repeat(bs,1,1)          # [bs, N, 2]
        max_q = N; B = bs
        unbatch_idx = torch.empty(max_q * B, dtype=torch.long)
        unbatch_select = []
        u = 0; uid = 0
        for _ in range(bs):
            L = N
            unbatch_idx[u:u+L] = uid
            unbatch_select.append(uid); uid += 1; u += L
        unbatch_select = torch.tensor(unbatch_select, dtype=torch.long)
        # other fields (NOTE: conditioner expects 1-D velocity per sample)
        geometry2d = torch.zeros(bs, 1, dtype=torch.float32)
        timestep   = torch.zeros(bs, dtype=torch.long)
        velocity   = torch.zeros(bs, dtype=torch.float32)  # <-- 1-D, not (bs,2)
        target     = torch.randn(bs*N, chans_out, dtype=torch.float32)
        return (
            (x, mesh_pos, query_pos, None, geometry2d, timestep, velocity, target),
            {"batch_idx": batch_idx, "unbatch_idx": unbatch_idx, "unbatch_select": unbatch_select},
        )

    def time_forward_backward_on_dummy(grid, do_backward=False, iters=5):
        batch = make_dummy_batch(grid, bs=args.batch_size)
        # Build graph ONCE, then reuse the kw for timing
        kw = prepare_kwargs(batch, device=args.device, radius_r=r_graph, radius_max_nn=k_graph,
                            num_supernodes=num_supernodes, make_graph_on_gpu=True)
        local_opt = torch.optim.SGD(model.parameters(), lr=1e-4) if do_backward else None
        loss_fn = nn.MSELoss()
        # Warmup
        for _ in range(2):
            out = model(**kw)
            if do_backward:
                (loss_fn(out["x_hat"], kw["target"])).backward()
                local_opt.step(); local_opt.zero_grad(set_to_none=True)
        # Timed
        ms_list = []
        for _ in range(iters):
            with cuda_timer(enabled=(args.device.startswith("cuda") and torch.cuda.is_available())) as t:
                out = model(**kw)
                if do_backward:
                    (loss_fn(out["x_hat"], kw["target"])).backward()
                    local_opt.step(); local_opt.zero_grad(set_to_none=True)
            ms_list.append(t.ms if t and t.ms is not None else 0.0)
        return sum(ms_list)/len(ms_list), ms_list

    grids = args.grid_sizes or [64, 128, 256, 512]
    print("grid, avg_infer_ms, avg_train_ms")
    for g in grids:
        model.eval()
        infer_ms, _ = time_forward_backward_on_dummy(g, do_backward=False, iters=5)
        model.train()
        train_ms, _ = time_forward_backward_on_dummy(g, do_backward=True, iters=5)
        model.eval()
        print(f"{g:4d}, {infer_ms:.2f}, {train_ms:.2f}")

if __name__ == "__main__":
    main()
