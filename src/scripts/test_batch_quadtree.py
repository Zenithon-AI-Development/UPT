"""
Test quadtree-UPT with small batch (2-4 samples) to verify batching works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.encoder_amr_quadtree import EncoderAMRQuadtree
from models.quadtree_upt import QuadtreeUPT
from collators.quadtree_collator import QuadtreeCollator
from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../upt-tutorial'))
from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.conditioner_timestep import ConditionerTimestep


def test_batching():
    """Test quadtree-UPT with batching"""
    
    print("=" * 80)
    print("TESTING QUADTREE-UPT WITH BATCHING")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 4
    max_nodes = 4096
    enc_dim = 128
    num_heads = 4
    input_dim = 4
    output_dim = 4
    
    print(f"\nConfiguration:")
    print(f"  batch_size: {batch_size}")
    print(f"  max_nodes: {max_nodes}")
    
    # === Load Dataset ===
    print(f"\n{'=' * 80}")
    print("LOADING DATASET")
    print(f"{'=' * 80}")
    
    dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=10,  # Load 10 samples for batching
    )
    
    # === Collator ===
    collator = QuadtreeCollator(
        max_nodes=max_nodes,
        k=2,
        min_size=4,
        max_size=16,
        deterministic=True,
    )
    
    # === DataLoader ===
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    print(f"\nDataLoader created:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {len(dataloader)}")
    
    # === Build Model ===
    print(f"\n{'=' * 80}")
    print("BUILDING MODEL")
    print(f"{'=' * 80}")
    
    conditioner = ConditionerTimestep(dim=enc_dim, num_timesteps=101)
    
    encoder = EncoderAMRQuadtree(
        input_dim=input_dim,
        ndim=2,
        enc_dim=enc_dim,
        enc_depth=4,
        enc_num_heads=num_heads,
        perc_dim=enc_dim,
        perc_num_heads=num_heads,
        num_latent_tokens=512,
        cond_dim=conditioner.cond_dim,
        max_depth=10,
    )
    
    approximator = Approximator(
        input_dim=enc_dim,
        dim=enc_dim,
        depth=4,
        num_heads=num_heads,
        cond_dim=conditioner.cond_dim,
    )
    
    decoder = DecoderPerceiver(
        input_dim=enc_dim,
        output_dim=output_dim,
        ndim=2,
        dim=enc_dim,
        num_heads=num_heads,
        depth=4,
        unbatch_mode="dense_to_sparse_unpadded",
        cond_dim=conditioner.cond_dim,
    )
    
    model = QuadtreeUPT(
        conditioner=conditioner,
        encoder=encoder,
        approximator=approximator,
        decoder=decoder,
    ).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # === Test Batches ===
    print(f"\n{'=' * 80}")
    print("TESTING BATCHES")
    print(f"{'=' * 80}")
    
    model.eval()
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}/{len(dataloader)}:")
        print(f"  node_feat: {batch['node_feat'].shape}")
        print(f"  node_pos: {batch['node_pos'].shape}")
        print(f"  depth: {batch['depth'].shape}")
        print(f"  batch_idx: {batch['batch_idx'].shape}")
        print(f"  output_pos: {batch['output_pos'].shape}")
        print(f"  output_feat: {batch['output_feat'].shape}")
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        try:
            with torch.no_grad():
                pred = model(
                    node_feat=batch["node_feat"],
                    node_pos=batch["node_pos"],
                    depth=batch["depth"],
                    batch_idx=batch["batch_idx"],
                    output_pos=batch["output_pos"],
                    timestep=batch["timestep"],
                )
            
            print(f"  ✓ Forward pass successful")
            print(f"  Prediction shape: {pred.shape}")
            
            # Compute loss
            batch_size_actual = batch["output_pos"].shape[0]
            num_output_points = batch["output_pos"].shape[1]
            pred_flat = pred.reshape(batch_size_actual * num_output_points, output_dim)
            loss = F.mse_loss(pred_flat, batch["output_feat"])
            print(f"  Loss: {loss.item():.6f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'=' * 80}")
    print("✓ ALL BATCHES PROCESSED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print(f"\nBatching works correctly with variable node counts.")
    
    return True


if __name__ == "__main__":
    success = test_batching()
    sys.exit(0 if success else 1)

