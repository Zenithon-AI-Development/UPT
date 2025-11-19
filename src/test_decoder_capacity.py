#!/usr/bin/env python3
"""
Test if the decoder alone can fit the data (bypass encoder/latent).
This will tell us if the issue is in the decoder or earlier components.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from configs.static_config import StaticConfig
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from datasets import dataset_from_kwargs
from models import model_from_kwargs
from torch_geometric.nn import radius_graph


class MiniDC:
    def __init__(self, ds):
        self._ds = ds
    def get_dataset(self, *_, **__):
        return self._ds


def test_decoder_only():
    """Test if decoder alone can fit the data."""
    print("="*70)
    print("TESTING DECODER CAPACITY (Bypass encoder/latent)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    repo_root = Path(__file__).parent.parent
    static = StaticConfig(uri=str(Path(__file__).parent / "static_config.yaml"))
    
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths=static.get_global_dataset_paths(),
        local_dataset_path=static.get_local_dataset_path(),
        data_source_modes=static.get_data_source_modes(),
    )
    path_provider = PathProvider(
        output_path=static.output_path,
        model_path=static.model_path,
        stage_name="decoder_test",
        stage_id="decoder_test",
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
    
    # Get sample
    target = dataset.getitem_target(0).to(device)
    query_pos = dataset.getitem_query_pos(0).to(device)
    
    print(f"\nTarget shape: {target.shape}")
    print(f"Target stats: mean={target.mean():.4f}, std={target.std():.4f}")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    
    # Create simple MLP decoder (no transformer, just direct mapping)
    print("\n" + "="*70)
    print("TEST 1: Simple MLP (position -> values)")
    print("="*70)
    
    class SimpleMLP(nn.Module):
        def __init__(self, hidden_dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),  # 4 output channels
            )
        
        def forward(self, pos):
            return self.net(pos)
    
    mlp = SimpleMLP(hidden_dim=512).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    for epoch in range(200):
        optimizer.zero_grad()
        pred = mlp(query_pos)
        loss = ((pred - target) ** 2).mean()
        rel_l1 = (pred - target).abs().sum() / (target.abs().sum() + 1e-12)
        
        loss.backward()
        optimizer.step()
        
        if epoch in [0, 1, 5, 10, 20, 50, 100, 199]:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.6f}, Rel L1={rel_l1.item():.6f} ({rel_l1.item()*100:.1f}%), "
                  f"Pred std={pred.std().item():.4f}")
    
    final_pred = mlp(query_pos)
    final_rel_l1 = (final_pred - target).abs().sum() / (target.abs().sum() + 1e-12)
    
    if final_rel_l1.item() < 0.01:
        print(f"\n✓ Simple MLP CAN fit the data! (rel_l1={final_rel_l1.item():.6f})")
        print("  → Problem is in encoder/latent/decoder architecture, not the data!")
    else:
        print(f"\n⚠ Simple MLP also struggles (rel_l1={final_rel_l1.item():.6f})")
        print("  → May be a fundamental data/task difficulty issue")
    
    # Test direct lookup (memorization)
    print("\n" + "="*70)
    print("TEST 2: Learnable lookup table (pure memorization)")
    print("="*70)
    
    # Create a learnable tensor that directly memorizes the target
    lookup_table = nn.Parameter(torch.randn_like(target))
    optimizer2 = torch.optim.Adam([lookup_table], lr=1e-2)
    
    for epoch in range(100):
        optimizer2.zero_grad()
        loss = ((lookup_table - target) ** 2).mean()
        rel_l1 = (lookup_table - target).abs().sum() / (target.abs().sum() + 1e-12)
        
        loss.backward()
        optimizer2.step()
        
        if epoch in [0, 1, 5, 10, 20, 50, 99]:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.8f}, Rel L1={rel_l1.item():.8f} ({rel_l1.item()*100:.4f}%)")
    
    final_rel_l1_lookup = (lookup_table - target).abs().sum() / (target.abs().sum() + 1e-12)
    print(f"\nLookup table final rel_l1: {final_rel_l1_lookup.item():.10f}")
    
    if final_rel_l1_lookup.item() < 1e-6:
        print("✓ Pure memorization works perfectly!")
        print("  → Issue is architectural, not optimizer or data")
    
    return final_rel_l1.item(), final_rel_l1_lookup.item()


if __name__ == "__main__":
    test_decoder_only()



