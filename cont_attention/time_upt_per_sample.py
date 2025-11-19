#!/usr/bin/env python3
"""
Simple example script to measure timing per sample for UPT simformer model.

Usage:
    python time_upt_per_sample.py --stage_id <stage_id> [--num_samples 10]
    
Example:
    python time_upt_per_sample.py --stage_id m0z2lv8q --num_samples 5
"""
import argparse
import time
import torch
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from torch_geometric.nn import radius_graph


class MiniDataContainer:
    """Minimal DataContainer for model initialization."""
    def __init__(self, ds):
        self._ds = ds
    def get_dataset(self, *_, **__):
        return self._ds


def load_model_and_dataset(stage_id, checkpoint="best_model.loss.online.x_hat.E1"):
    """Load model and dataset from stage_id."""
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    hp_path = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "hp_resolved.yaml"
    ckpt_dir = repo_root / "benchmarking" / "save" / "stage1" / stage_id / "checkpoints"
    
    if not hp_path.exists():
        raise FileNotFoundError(f"HP file not found: {hp_path}")
    
    # Load config
    with open(hp_path) as f:
        hp = yaml.safe_load(f)
    
    static_config_path = repo_root / "src" / "static_config.yaml"
    static = StaticConfig(uri=str(static_config_path))
    
    # Build dataset
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name=f"timing_{stage_id}",
        stage_id=stage_id,
        temp_path=static.temp_path,
    )
    
    # Get test dataset
    test_kwargs = hp["datasets"].get("test", hp["datasets"]["train"])
    if "split" not in test_kwargs:
        test_kwargs["split"] = "test"
    
    dataset = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **test_kwargs,
    )
    
    # Build model
    train_kwargs = hp["datasets"]["train"]
    train_ds = dataset_from_kwargs(
        dataset_config_provider=dataset_config_provider,
        path_provider=path_provider,
        **train_kwargs,
    )
    
    input_shape = train_ds.getshape_x()
    output_shape = train_ds.getshape_target()
    if len(output_shape) == 2:
        output_shape = (output_shape[0], output_shape[1])
    
    model = model_from_kwargs(
        **hp["model"],
        input_shape=input_shape,
        output_shape=output_shape,
        update_counter=None,
        path_provider=path_provider,
        data_container=MiniDataContainer(train_ds),
    )
    
    # Load checkpoints
    for component in ["conditioner", "encoder", "latent", "decoder"]:
        ckpt_files = list(ckpt_dir.glob(f"*{component}*cp={checkpoint}*model.th"))
        if ckpt_files and hasattr(model, component):
            state = torch.load(str(ckpt_files[0]), map_location="cpu")
            sd = state.get("state_dict", state)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            getattr(model, component).load_state_dict(sd, strict=False)
    
    # Get radius graph params
    radius_r = hp.get("trainer", {}).get("radius_graph_r", 5.0)
    radius_max_nn = hp.get("trainer", {}).get("radius_graph_max_num_neighbors", 32)
    num_supernodes = None
    if "vars" in hp and "num_supernodes" in hp["vars"]:
        num_supernodes = hp["vars"]["num_supernodes"]
    
    return model, dataset, radius_r, radius_max_nn, num_supernodes


def prepare_sample(dataset, idx, device, radius_r, radius_max_nn, num_supernodes):
    """Prepare a single sample for forward pass."""
    # Get data
    x = dataset.getitem_x(idx)
    target = dataset.getitem_target(idx)
    geometry2d = dataset.getitem_geometry2d(idx)
    timestep = dataset.getitem_timestep(idx)
    velocity = dataset.getitem_velocity(idx)
    mesh_pos = dataset.getitem_mesh_pos(idx)
    query_pos = dataset.getitem_query_pos(idx)
    
    # Convert to tensors
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep, dtype=torch.long).view(1)
    else:
        timestep = timestep.view(1)
    
    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity, dtype=torch.float32).view(1)
    else:
        velocity = velocity.view(1)
    
    # Create batch indices
    N = x.shape[0]
    Nq = query_pos.shape[0]
    batch_idx = torch.zeros(N, dtype=torch.long)
    unbatch_idx = torch.zeros(Nq, dtype=torch.long)
    unbatch_select = torch.zeros(1, dtype=torch.long)
    
    # Move to device
    to_dev = lambda t: t.to(device, non_blocking=True) if t is not None else None
    x = to_dev(x)
    target = to_dev(target)
    geometry2d = to_dev(geometry2d)
    timestep = to_dev(timestep)
    velocity = to_dev(velocity)
    mesh_pos = to_dev(mesh_pos)
    query_pos = to_dev(query_pos)
    batch_idx = to_dev(batch_idx)
    unbatch_idx = to_dev(unbatch_idx)
    unbatch_select = to_dev(unbatch_select)
    
    # Add batch dimension to query_pos
    query_pos = query_pos.unsqueeze(0)
    
    # Build mesh edges
    flow = "target_to_source" if num_supernodes is not None else "source_to_target"
    edge_index = radius_graph(
        x=mesh_pos, r=radius_r, batch=batch_idx, loop=True,
        max_num_neighbors=radius_max_nn, flow=flow
    )
    mesh_edges = edge_index.T
    
    inputs = (x, geometry2d, timestep, velocity, mesh_pos, query_pos,
              mesh_edges, batch_idx, unbatch_idx, unbatch_select)
    return inputs, target


@torch.no_grad()
def time_forward_pass(model, inputs, device, warmup=True):
    """Time a single forward pass."""
    if warmup:
        # Warmup pass (not timed)
        _ = model(*inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timed pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    
    outputs = model(*inputs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed, outputs


def main():
    parser = argparse.ArgumentParser(description="Time UPT model per sample")
    parser.add_argument("--stage_id", required=True, help="Stage ID (e.g., m0z2lv8q)")
    parser.add_argument("--checkpoint", default="best_model.loss.online.x_hat.E1",
                        help="Checkpoint name")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to time")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from stage: {args.stage_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Load model and dataset
    model, dataset, radius_r, radius_max_nn, num_supernodes = load_model_and_dataset(
        args.stage_id, args.checkpoint
    )
    model = model.to(device).eval()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Timing {args.num_samples} samples...")
    print()
    
    # Time samples
    times = []
    num_to_time = min(args.num_samples, len(dataset))
    
    for idx in range(num_to_time):
        inputs, target = prepare_sample(
            dataset, idx, device, radius_r, radius_max_nn, num_supernodes
        )
        elapsed, outputs = time_forward_pass(model, inputs, device, warmup=(idx == 0))
        times.append(elapsed)
        
        if idx < 3 or idx == num_to_time - 1:
            print(f"Sample {idx}: {elapsed*1000:.2f} ms")
    
    # Statistics
    print()
    print("="*60)
    print(f"Timing Statistics (n={num_to_time})")
    print("="*60)
    print(f"Mean:   {sum(times)/len(times)*1000:.2f} ms")
    print(f"Median: {sorted(times)[len(times)//2]*1000:.2f} ms")
    print(f"Min:    {min(times)*1000:.2f} ms")
    print(f"Max:    {max(times)*1000:.2f} ms")
    print("="*60)


if __name__ == "__main__":
    main()



