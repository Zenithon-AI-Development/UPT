"""
Analyze quadtree node count distribution across TRL2D dataset.

This helps determine optimal max_nodes and understand truncation frequency.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm

from collators.quadtree_collator import QuadtreeCollator
from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset


def analyze_node_distribution():
    """Analyze node counts across dataset"""
    
    print("=" * 80)
    print("ANALYZING QUADTREE NODE DISTRIBUTION FOR TRL2D")
    print("=" * 80)
    
    # === Load Dataset ===
    print(f"\n Loading dataset...")
    
    dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=None,  # Load all training data
    )
    
    print(f"  Total samples: {len(dataset)}")
    
    # === Quadtree Collator ===
    # Use high max_nodes to not truncate (for counting)
    collator_for_counting = QuadtreeCollator(
        max_nodes=100000,  # Very high to avoid truncation
        k=2,
        min_size=4,
        max_size=16,
        deterministic=True,
    )
    
    # === Collect Node Counts ===
    print(f"\n Analyzing node counts...")
    
    node_counts = []
    
    # Temporarily disable warnings in collator
    import warnings
    warnings.filterwarnings('ignore')
    
    for idx in tqdm(range(min(len(dataset), 100))):  # Analyze first 100 samples
        sample = dataset[idx]
        
        # Count nodes by doing quadtree partition
        try:
            batch = collator_for_counting([sample])
            # Count actual non-zero nodes
            valid_mask = (batch['node_feat'].abs().sum(dim=1) > 1e-6)
            num_nodes = valid_mask.sum().item()
            node_counts.append(num_nodes)
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            continue
    
    warnings.filterwarnings('default')
    
    # === Statistics ===
    print(f"\n{'=' * 80}")
    print("NODE COUNT STATISTICS")
    print(f"{'=' * 80}")
    
    node_counts = np.array(node_counts)
    
    print(f"\nSamples analyzed: {len(node_counts)}")
    print(f"\nNode count distribution:")
    print(f"  Minimum:     {node_counts.min():,}")
    print(f"  Maximum:     {node_counts.max():,}")
    print(f"  Mean:        {node_counts.mean():,.0f}")
    print(f"  Median:      {np.median(node_counts):,.0f}")
    print(f"  Std:         {node_counts.std():,.0f}")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(node_counts, p)
        print(f"  {p:2d}th: {val:>8,.0f}")
    
    # === Truncation Analysis ===
    print(f"\n{'=' * 80}")
    print("TRUNCATION ANALYSIS")
    print(f"{'=' * 80}")
    
    max_nodes_options = [1024, 2048, 4096, 8192, 16384]
    
    print(f"\n{'max_nodes':<12} | {'Coverage':<10} | {'Truncated':<12} | {'Avg Waste'}")
    print("-" * 60)
    
    for max_nodes in max_nodes_options:
        truncated = (node_counts > max_nodes).sum()
        coverage = (1 - truncated / len(node_counts)) * 100
        avg_waste = ((max_nodes - node_counts[node_counts <= max_nodes].mean()) / max_nodes * 100) if truncated < len(node_counts) else 0
        
        print(f"{max_nodes:>10,} | {coverage:>8.1f}% | {truncated:>6} / {len(node_counts):>4} | {avg_waste:>6.1f}%")
    
    # === Recommendations ===
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")
    
    # Find max_nodes that covers 95% of samples
    max_nodes_95 = np.percentile(node_counts, 95)
    max_nodes_99 = np.percentile(node_counts, 99)
    
    print(f"\nTo cover 95% of samples: max_nodes ≥ {max_nodes_95:,.0f}")
    print(f"To cover 99% of samples: max_nodes ≥ {max_nodes_99:,.0f}")
    
    # Round up to power of 2
    max_nodes_95_pow2 = 2 ** int(np.ceil(np.log2(max_nodes_95)))
    max_nodes_99_pow2 = 2 ** int(np.ceil(np.log2(max_nodes_99)))
    
    print(f"\nRounded to power of 2:")
    print(f"  95% coverage: max_nodes = {max_nodes_95_pow2:,}")
    print(f"  99% coverage: max_nodes = {max_nodes_99_pow2:,}")
    
    print(f"\nFinal recommendation:")
    if max_nodes_95_pow2 <= 4096:
        print(f"  ✓ max_nodes = 4096 is sufficient (covers {coverage:.1f}% of samples)")
    elif max_nodes_99_pow2 <= 8192:
        print(f"  ⚠ Consider max_nodes = 8192 for better coverage")
    else:
        print(f"  ⚠ Node counts are very high, consider:")
        print(f"    - Adjusting quadtree parameters (increase min_size or max_size)")
        print(f"    - Using coarsening strategy instead of truncation")
    
    return node_counts


if __name__ == "__main__":
    node_counts = analyze_node_distribution()

