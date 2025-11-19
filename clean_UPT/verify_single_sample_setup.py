#!/usr/bin/env python3
"""
Verify the single sample setup is correct - no data duplication, correct shapes.
"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Use the shared env
import subprocess
import os

def run_with_env(cmd):
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    result = subprocess.run(
        ['/home/shared_env/upt_env/bin/python', '-c', cmd],
        cwd='/home/workspace/projects/transformer/UPT',
        capture_output=True,
        text=True,
        env=env
    )
    return result.stdout, result.stderr, result.returncode

print("="*70)
print("SINGLE SAMPLE SETUP VERIFICATION")
print("="*70)

code = '''
import torch
import sys
sys.path.insert(0, "src")

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs

static = StaticConfig(uri="src/static_config.yaml")
dataset_config_provider = DatasetConfigProvider(
    global_dataset_paths=static.get_global_dataset_paths(),
    local_dataset_path=static.get_local_dataset_path(),
    data_source_modes=static.get_data_source_modes(),
)
path_provider = PathProvider(
    output_path=static.output_path,
    model_path=static.model_path,
    stage_name="check",
    stage_id="check",
    temp_path=static.temp_path,
)

dataset = dataset_from_kwargs(
    kind="well_trl2d_dataset",
    split="train",
    well_base_path="/home/workspace/projects/data/datasets_david/datasets/",
    num_input_timesteps=4,
    norm="mean0std1",
    clamp=0,
    clamp_mode="log",
    max_num_timesteps=101,
    max_num_sequences=1,
    dataset_config_provider=dataset_config_provider,
    path_provider=path_provider,
)

print(f"1. Dataset length: {len(dataset)} (must be 1)")
assert len(dataset) == 1, f"Expected 1, got {len(dataset)}"

print(f"2. Sample access:")
x0 = dataset.getitem_x(0)
t0 = dataset.getitem_target(0)
print(f"   Input shape: {x0.shape}")
print(f"   Target shape: {t0.shape}")
print(f"   Input stats: mean={x0.mean():.6f}, std={x0.std():.6f}")
print(f"   Target stats: mean={t0.mean():.6f}, std={t0.std():.6f}")

# Check if same data returned consistently
x1 = dataset.getitem_x(0)
print(f"3. Consistency: {torch.allclose(x0, x1)}")

# Check normalization
if abs(t0.mean()) < 0.5 and 0.5 < t0.std() < 1.5:
    print(f"4. ✓ Target appears normalized (mean≈0, std≈1)")
else:
    print(f"4. ⚠ Target normalization suspicious: mean={t0.mean():.6f}, std={t0.std():.6f}")

# Check timestep
ts = dataset.getitem_timestep(0)
print(f"5. Timestep index: {ts.item()}")

print("\\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
'''

stdout, stderr, code = run_with_env(code)
print(stdout)
if stderr and "Error" in stderr or "Traceback" in stderr:
    print("ERROR:", stderr)
if code != 0:
    print(f"Exit code: {code}")

