#!/usr/bin/env python3
import os
import argparse
import time

import torch
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

@torch.no_grad()
def eval_autoregressive(model, dl, device, radius_r, radius_max_nn, num_supernodes, max_T=None, T_clip=None):
    # Accumulate per-step metrics
    mse_sums, r1_sums, r2_sums, counts = [], [], [], []
    for batch in dl:
        # Unpack (+ roll targets)
        (tup, ctx) = batch
        has_roll_targets = len(tup) == 9  # 8 base + target_rollout
        if not has_roll_targets:
            raise RuntimeError("No full rollout targets in batch. Add target sequence to ModeWrapper/collate.")

        x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target_next, target_seq = tup
        # Build kw for rollout
        # (rebuild mesh graph on GPU if none)
        from torch_geometric.nn import radius_graph
        batch_idx = ctx["batch_idx"].to(device, non_blocking=True)
        kw = dict(
            x=x.to(device, non_blocking=True),
            geometry2d=geometry2d.to(device, non_blocking=True),
            velocity=velocity.to(device, non_blocking=True),
            mesh_pos=mesh_pos.to(device, non_blocking=True),
            query_pos=query_pos.to(device, non_blocking=True),
            batch_idx=batch_idx,
            unbatch_idx=ctx["unbatch_idx"].to(device, non_blocking=True),
            unbatch_select=ctx["unbatch_select"].to(device, non_blocking=True),
        )
        if mesh_edges is None:
            flow = "target_to_source" if num_supernodes is not None else "source_to_target"
            edge_index = radius_graph(
                x=kw["mesh_pos"], r=radius_r, batch=batch_idx, loop=True,
                max_num_neighbors=radius_max_nn, flow=flow
            )
            kw["mesh_edges"] = edge_index.t()
        else:
            kw["mesh_edges"] = mesh_edges.to(device, non_blocking=True)

        # Run rollout (returns [B*Q, C, T_out] stacked)
        T_out = max_T
        roll = model.rollout(num_rollout_timesteps=T_out, mode="image", intermediate_results=True, **kw)
        # roll: (B*Q, C, T_out)
        if T_clip is not None:
            T_out = min(T_out, T_clip)
            roll = roll[:, :, :T_out]

        # Make sure target_seq matches [B*Q, C, T_target]
        gt = target_seq.to(device, non_blocking=True)
        T_target = gt.shape[-1]
        T_eval = min(roll.shape[-1], T_target)
        roll = roll[:, :, :T_eval]
        gt   = gt[:,   :, :T_eval]

        # compute per-step metrics
        for t in range(T_eval):
            pred_t = roll[:, :, t]
            gt_t   = gt[:, :, t]
            if len(mse_sums) <= t:
                mse_sums.append(0.0); r1_sums.append(0.0); r2_sums.append(0.0); counts.append(0)
            mse_sums[t] += torch.mean((pred_t - gt_t)**2).item()
            r1_sums[t]  += rel_l1(pred_t, gt_t).item()
            r2_sums[t]  += rel_l2(pred_t, gt_t).item()
            counts[t]   += 1

    # aggregate to lists
    mse = [mse_sums[t] / max(1, counts[t]) for t in range(len(counts))]
    r1  = [r1_sums[t]  / max(1, counts[t]) for t in range(len(counts))]
    r2  = [r2_sums[t]  / max(1, counts[t]) for t in range(len(counts))]
    return dict(mse=mse, rel_l1=r1, rel_l2=r2)

@torch.no_grad()
def measure_inference_latency(model, dl, device, iters=50, warmup=10, r_graph=None, k_graph=None, num_supernodes=None):
    times = []
    import torch
    for i, batch in enumerate(dl):
        kw = prepare_kwargs(
            batch=batch, device=device, radius_r=r_graph, radius_max_nn=k_graph,
            num_supernodes=num_supernodes, make_graph_on_gpu=True
        )
        if i < warmup:
            _ = model(**kw)
            continue
        if len(times) >= iters:
            break
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        _ = model(**kw)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  # milliseconds
        times.append(ms)
    if not times:
        return dict(avg_ms=None, p50=None, p90=None, p99=None)
    import numpy as np
    arr = np.array(times, dtype=float)
    return dict(
        avg_ms=arr.mean(),
        p50=float(np.percentile(arr, 50)),
        p90=float(np.percentile(arr, 90)),
        p99=float(np.percentile(arr, 99)),
    )

def measure_training_throughput(model, dl, device, steps=100, warmup=10, lr=1e-4, r_graph=None, k_graph=None, num_supernodes=None):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    import torch, time
    seen = 0
    # Warmup
    it = iter(dl)
    for _ in range(warmup):
        kw = prepare_kwargs(next(it), device, r_graph, k_graph, num_supernodes, True)
        out = model(**kw)
        loss = torch.mean((out["x_hat"] - kw["target"])**2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        seen += kw["target"].shape[0]
    # Timed
    start = time.perf_counter()
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        kw = prepare_kwargs(batch, device, r_graph, k_graph, num_supernodes, True)
        out = model(**kw)
        loss = torch.mean((out["x_hat"] - kw["target"])**2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        seen += kw["target"].shape[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    model.eval()
    return dict(samples_per_sec=seen / wall, steps=steps, wall_s=wall)


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
    args = ap.parse_args()

    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = "src"

    add_global_handlers(log_file_uri=None)
    set_seed(0)
    print("set seed to 0")

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

    # Create a train-like test split so x has same Tin*C as in training
    train_like = dict(stage_hp["datasets"]["test"])
    train_like["split"] = "valid"
    train_like["num_input_timesteps"] = stage_hp["datasets"]["test"].get("num_input_timesteps", 4)
    datasets["eval_like_train"] = dataset_from_kwargs(
        dataset_config_provider=dcp,
        path_provider=path_provider,
        **train_like,
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

    # Determine T for conditioner __init__
    def pick_num_timesteps(stage_hp):
        # t = stage_hp["datasets"]["train"].get("max_num_timesteps", None)
        # if t is not None: return t
        # t = stage_hp.get("vars", {}).get("max_num_timesteps", None)
        # if t is not None: return t
        return 100
    T_yaml = pick_num_timesteps(stage_hp)

    # Stub DC so conditioner can query getdim_timestep during __init__
    class _StubDs:
        def __init__(self, T): self._T = T
        def getdim_timestep(self): return self._T
    class _StubDC:
        def __init__(self, T): self._ds = _StubDs(T)
        def get_dataset(self, *args, **kwargs): return self._ds

    stub_dc = _StubDC(T_yaml)

    model = model_from_kwargs(
        **stage_hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=stub_dc,
    ).to(args.device).eval()

    # Swap in real data_container everywhere
    def _swap_dc(m, real_dc):
        if hasattr(m, "data_container"): m.data_container = real_dc
        for name in ("conditioner", "geometry_encoder", "encoder", "latent", "decoder"):
            sub = getattr(m, name, None)
            if sub is not None and hasattr(sub, "data_container"):
                sub.data_container = real_dc
    _swap_dc(model, data_container)

    if hasattr(model, "conditioner") and model.conditioner is not None:
        try: model.conditioner.num_total_timesteps = T_yaml
        except Exception: pass

    # -------- Pick eval dataset AND unwrap to base before wrapping --------
    ds_test, _ = data_container.get_dataset("eval_like_train", mode="x")

    def unwrap_to_kd_base(ds, max_hops=16):
        cur = ds
        seen = {id(cur)}
        for _ in range(max_hops):
            nxt = getattr(cur, "dataset", None)
            if nxt is None:
                nxt = getattr(cur, "root_dataset", None)
            if nxt is None:
                break
            # stop if no progress or we’ve already seen this object (cycle/self-loop)
            if id(nxt) in seen or nxt is cur:
                break
            cur = nxt
            seen.add(id(cur))
        return cur

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
        for i, xi in enumerate(x_list):
            assert len(xi) == mesh_lens[i], f"x length != mesh length at sample {i}"
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
            unbatch_select.append(uid)
            uid += 1
            u += L
            pad = max_q - L
            if pad > 0:
                unbatch_idx[u:u+pad] = uid
                uid += 1
                u += pad
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

        batch_tuple = (
            x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target
        )
        ctx = {
            "batch_idx": batch_idx,
            "unbatch_idx": unbatch_idx,
            "unbatch_select": unbatch_select,
        }
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

    # -------- Load submodule checkpoints --------
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
        if not pick:
            continue
        path = os.path.join(args.ckpt_dir, sorted(pick)[-1])
        getattr(model, name).load_state_dict(_load_sd(path), strict=False)
        loaded.append(path)
    if loaded:
        print("[ckpt] loaded:\n  " + "\n  ".join(loaded))
    else:
        print("[ckpt] WARNING: no submodule checkpoints found in", args.ckpt_dir)

    T_ckpt = parse_ckpt_T(args.ckpt_dir)

    # -------- One-step metrics on test --------
    print(f"[sanity] len(test) = {len(mw_test)}")

    mse_sum = 0.0
    r1_sum  = 0.0
    r2_sum  = 0.0
    n       = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in dl_test:
            kw = prepare_kwargs(
                batch=batch, device=args.device,
                radius_r=r_graph, radius_max_nn=k_graph,
                num_supernodes=derive_num_supernodes(stage_hp),
                make_graph_on_gpu=True,
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

if __name__ == "__main__":
    main()
