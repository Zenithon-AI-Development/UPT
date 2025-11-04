"""
Quick and correct analysis of actual quadtree node counts.

Directly counts nodes from quadtree_partition_parallel output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm

from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset
from amr_transformer.tree import quadtree_partition_parallel, pad_to_power_of_k


def analyze_actual_node_counts(num_samples=200):
    """
    Directly analyze quadtree node counts without collator overhead.
    """
    
    print("=" * 80)
    print("EFFICIENT NODE COUNT ANALYSIS")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset...")
    dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=None,
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"Analyzing first {num_samples} samples...")
    
    # Quadtree parameters (matching collator)
    k = 2
    min_size = 4
    max_size = 16
    
    node_counts = []
    
    for idx in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[idx]
        input_grid = sample["input_grid"]  # (C, H, W)
        target_grid = sample["target_grid"]  # (C, H, W)
        
        # Convert to (H, W, C)
        C, H, W = input_grid.shape
        input_hwc = input_grid.permute(1, 2, 0)
        target_hwc = target_grid.permute(1, 2, 0)
        
        # Pad to power of k
        input_padded = pad_to_power_of_k(input_hwc.unsqueeze(0), 0.0, k)[0]
        target_padded = pad_to_power_of_k(target_hwc.unsqueeze(0), 0.0, k)[0]
        
        mask = torch.ones(input_padded.shape[0], input_padded.shape[1], 1)
        feature_field = input_padded[:, :, 2:4] if C >= 4 else input_padded[:, :, :2]
        
        # Run quadtree partitioning
        regions, patches_tensor, labels_tensor = quadtree_partition_parallel(
            inputs=input_padded.cuda(),
            labels=target_padded.cuda(),
            mask=mask.cuda(),
            feature_field=feature_field.cuda(),
            k=k,
            min_size=min_size,
            max_size=max_size,
            common_refine_threshold=0.4,
            integral_refine_threshold=0.1,
            vorticity_threshold=0.5,
            momentum_threshold=0.5,
            shear_threshold=0.5,
            condition_type='grad',
        )
        
        # patches_tensor shape: (N_total_patches, features)
        # Need to reshape to (N_nodes, k*k, features)
        # N_nodes = N_total_patches / (k*k)
        
        N_total_patches = patches_tensor.shape[0]
        N_nodes = N_total_patches // (k * k)
        
        node_counts.append(N_nodes)
    
    # === Statistics ===
    node_counts = np.array(node_counts)
    
    print(f"\n{'=' * 80}")
    print("NODE COUNT STATISTICS")
    print(f"{'=' * 80}")
    
    print(f"\nSamples analyzed: {len(node_counts)}")
    print(f"\nNode count distribution:")
    print(f"  Minimum:     {node_counts.min():,}")
    print(f"  Maximum:     {node_counts.max():,}")
    print(f"  Mean:        {node_counts.mean():,.1f}")
    print(f"  Median:      {np.median(node_counts):,.0f}")
    print(f"  Std:         {node_counts.std():,.1f}")
    
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99, 100]:
        val = np.percentile(node_counts, p)
        print(f"  {p:3d}th: {val:>8,.0f} nodes")
    
    # === Coverage Analysis ===
    print(f"\n{'=' * 80}")
    print("COVERAGE WITH DIFFERENT max_nodes")
    print(f"{'=' * 80}")
    
    max_nodes_options = [4032, 4096, 5120, 6144, 8192]
    
    print(f"\n{'max_nodes':<12} | {'Covered':<8} | {'Truncated':<10} | {'Lost Info':<10}")
    print("-" * 60)
    
    for max_nodes in max_nodes_options:
        covered = (node_counts <= max_nodes).sum()
        truncated = (node_counts > max_nodes).sum()
        avg_lost = ((node_counts[node_counts > max_nodes] - max_nodes).mean() / node_counts[node_counts > max_nodes].mean() * 100) if truncated > 0 else 0
        
        print(f"{max_nodes:>10,} | {covered:>3}/{len(node_counts):<4} | {truncated:>4}/{len(node_counts):<5} | {avg_lost:>6.1f}%")
    
    # === Final Recommendation ===
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    
    p99 = np.percentile(node_counts, 99)
    p100 = node_counts.max()
    
    # Round up to nice number
    recommended_max = int(np.ceil(p100 / 256) * 256)  # Round to nearest 256
    
    print(f"\n99th percentile: {p99:,.0f}")
    print(f"100th percentile (max): {p100:,.0f}")
    print(f"\nRecommended max_nodes: {recommended_max:,}")
    print(f"  (Covers 100% of samples with minimal padding)")
    
    # Check memory implications
    print(f"\nMemory comparison:")
    print(f"  UPT default (2048 supernodes):     Baseline")
    print(f"  Quadtree ({recommended_max} nodes): {recommended_max/2048:.2f}x")
    
    return node_counts, recommended_max


if __name__ == "__main__":
    node_counts, recommended = analyze_actual_node_counts(num_samples=200)

