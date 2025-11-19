#!/usr/bin/env python3
"""Check overfitting results and see if we achieved target."""
import yaml
import sys
from pathlib import Path

if len(sys.argv) > 1:
    stage_id = sys.argv[1]
else:
    # Find latest
    stage_dir = Path("benchmarking/save/stage1")
    if not stage_dir.exists():
        print("No stage directory found!")
        sys.exit(1)
    stages = sorted([d for d in stage_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
    if not stages:
        print("No stages found!")
        sys.exit(1)
    stage_id = stages[-1].name

print(f"Checking results for stage: {stage_id}")
entries_file = Path(f"benchmarking/save/stage1/{stage_id}/primitive/entries.yaml")

if not entries_file.exists():
    print(f"File not found: {entries_file}")
    sys.exit(1)

with open(entries_file) as f:
    data = yaml.safe_load(f)

x_hat = data.get('loss/online/x_hat/E100', {}) or {}
rel_l1 = data.get('loss/online/rel_l1/E100', {}) or {}
rel_l2 = data.get('loss/online/rel_l2/E100', {}) or {}

if not x_hat:
    print("No training data found!")
    sys.exit(1)

epochs = sorted([int(k) for k in x_hat.keys()])

print("\n" + "="*70)
print("TRAINING RESULTS")
print("="*70)
print(f"Total epochs: {epochs[-1] if epochs else 0}")
print("\nKey milestones:")
for e in [100, 500, 1000, 2000, 3000, 4000, 5000]:
    if e in epochs:
        rl1 = rel_l1.get(e, 0)
        rl2 = rel_l2.get(e, 0)
        print(f"  Epoch {e:5d}: Loss={x_hat.get(e, 0):.8f}, "
              f"Rel L1={rl1:.8f} ({rl1*100:.4f}%), "
              f"Rel L2={rl2:.8f} ({rl2*100:.4f}%)")

if epochs:
    final_e = epochs[-1]
    final_loss = x_hat.get(final_e, 0)
    final_rl1 = rel_l1.get(final_e, 1)
    final_rl2 = rel_l2.get(final_e, 1)
    
    print("\n" + "="*70)
    print(f"FINAL RESULTS (Epoch {final_e})")
    print("="*70)
    print(f"  MSE Loss: {final_loss:.8f}")
    print(f"  Rel L1:   {final_rl1:.8f} ({final_rl1*100:.4f}%)")
    print(f"  Rel L2:   {final_rl2:.8f} ({final_rl2*100:.4f}%)")
    print(f"\n  Target:   Rel L1 < 0.001 (0.1%)")
    
    if final_rl1 < 0.001:
        print("\n  ✓✓✓ SUCCESS! TARGET ACHIEVED (< 0.1%)! ✓✓✓")
    else:
        gap = final_rl1 - 0.001
        factor = final_rl1 / 0.001
        print(f"\n  ⚠ Not yet reached:")
        print(f"     Current: {final_rl1*100:.4f}%")
        print(f"     Gap: {gap:.8f}")
        print(f"     Need {factor:.1f}x more improvement")
        
        # Check trend
        if len(epochs) >= 2:
            recent_rl1 = rel_l1.get(epochs[-2], 1)
            if final_rl1 < recent_rl1:
                improvement = (recent_rl1 - final_rl1) / recent_rl1 * 100
                print(f"     Recent improvement: {improvement:.2f}%")
                est_epochs_needed = (final_rl1 / (improvement/100)) if improvement > 0 else float('inf')
                print(f"     Estimated epochs needed: ~{est_epochs_needed:.0f}")
