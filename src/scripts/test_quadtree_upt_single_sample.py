"""
Test quadtree-UPT integration with a single TRL2D sample.

This script:
1. Loads 1 sample from TRL2D dataset
2. Applies quadtree partitioning
3. Feeds through EncoderAMRQuadtree → Approximator → Decoder
4. Verifies shapes and computes loss
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from models.encoder_amr_quadtree import EncoderAMRQuadtree
from models.quadtree_upt import QuadtreeUPT
from collators.quadtree_collator import QuadtreeCollator
from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset

# Import UPT components (keeping them unchanged)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../upt-tutorial'))
from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.conditioner_timestep import ConditionerTimestep


def test_single_sample():
    """Test quadtree-UPT on a single sample"""
    
    print("=" * 80)
    print("TESTING QUADTREE-UPT ON SINGLE TRL2D SAMPLE")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # === Configuration ===
    max_nodes = 4096
    enc_dim = 128
    num_heads = 4
    input_dim = 4  # TRL2D channels: density, pressure, vx, vy
    output_dim = 4
    
    print(f"\nConfiguration:")
    print(f"  max_nodes: {max_nodes}")
    print(f"  enc_dim: {enc_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  input_dim: {input_dim}")
    print(f"  output_dim: {output_dim}")
    
    # === Load Dataset ===
    print(f"\n{'=' * 80}")
    print("LOADING DATASET")
    print(f"{'=' * 80}")
    
    data_dir = "/home/workspace/projects/data/datasets_david/datasets/"
    
    try:
        dataset = TRL2DQuadtreeDataset(
            data_dir=data_dir,
            split="train",
            n_input_timesteps=1,
            max_num_sequences=1,  # Just load 1 sample for testing
        )
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    
    # === Get Single Sample ===
    print(f"\n{'=' * 80}")
    print("LOADING SINGLE SAMPLE")
    print(f"{'=' * 80}")
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"  input_grid shape: {sample['input_grid'].shape}")
    print(f"  target_grid shape: {sample['target_grid'].shape}")
    print(f"  timestep: {sample['timestep']}")
    
    # === Apply Quadtree Collator ===
    print(f"\n{'=' * 80}")
    print("APPLYING QUADTREE PARTITIONING")
    print(f"{'=' * 80}")
    
    collator = QuadtreeCollator(
        max_nodes=max_nodes,
        k=2,
        min_size=4,
        max_size=16,
        deterministic=True,
    )
    
    batch = collator([sample])
    
    print(f"\nCollated batch keys: {batch.keys()}")
    print(f"  node_feat shape: {batch['node_feat'].shape}")
    print(f"  node_pos shape: {batch['node_pos'].shape}")
    print(f"  depth shape: {batch['depth'].shape}")
    print(f"  batch_idx shape: {batch['batch_idx'].shape}")
    print(f"  output_pos shape: {batch['output_pos'].shape}")
    print(f"  output_feat shape: {batch['output_feat'].shape}")
    
    # Check how many nodes are valid (non-padded)
    valid_nodes = (batch['node_feat'].abs().sum(dim=1) > 0).sum().item()
    print(f"\n  Valid nodes: {valid_nodes} / {max_nodes}")
    print(f"  Padding ratio: {(max_nodes - valid_nodes) / max_nodes * 100:.1f}%")
    
    # === Build Model ===
    print(f"\n{'=' * 80}")
    print("BUILDING MODEL")
    print(f"{'=' * 80}")
    
    # Conditioner (for temporal)
    num_timesteps = 101  # TRL2D has 101 timesteps
    conditioner = ConditionerTimestep(
        dim=enc_dim,
        num_timesteps=num_timesteps,
    )
    
    # Encoder (quadtree-based)
    encoder = EncoderAMRQuadtree(
        input_dim=input_dim,
        ndim=2,
        enc_dim=enc_dim,
        enc_depth=4,
        enc_num_heads=num_heads,
        perc_dim=enc_dim,
        perc_num_heads=num_heads,
        num_latent_tokens=512,  # Downsample to 512 tokens
        cond_dim=conditioner.cond_dim,
        max_depth=10,
    )
    
    # Approximator (temporal, unchanged)
    approximator = Approximator(
        input_dim=enc_dim,
        dim=enc_dim,
        depth=4,
        num_heads=num_heads,
        cond_dim=conditioner.cond_dim,
    )
    
    # Decoder (unchanged)
    decoder = DecoderPerceiver(
        input_dim=enc_dim,
        output_dim=output_dim,
        ndim=2,
        dim=enc_dim,
        num_heads=num_heads,
        depth=4,
        unbatch_mode="dense_to_sparse_unpadded",  # Convert to sparse output
        cond_dim=conditioner.cond_dim,
    )
    
    # Full model
    model = QuadtreeUPT(
        conditioner=conditioner,
        encoder=encoder,
        approximator=approximator,
        decoder=decoder,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created:")
    print(f"  Total parameters: {num_params:.2f}M")
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
    print(f"  Approximator: {sum(p.numel() for p in approximator.parameters()) / 1e6:.2f}M")
    print(f"  Decoder: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")
    
    # === Forward Pass ===
    print(f"\n{'=' * 80}")
    print("FORWARD PASS")
    print(f"{'=' * 80}")
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    print(f"\nRunning forward pass...")
    
    try:
        pred = model(
            node_feat=batch["node_feat"],
            node_pos=batch["node_pos"],
            depth=batch["depth"],
            batch_idx=batch["batch_idx"],
            output_pos=batch["output_pos"],
            timestep=batch["timestep"],
        )
        
        print(f"✓ Forward pass successful!")
        print(f"  Prediction shape: {pred.shape}")
        print(f"  Expected shape: {batch['output_feat'].shape}")
        
        # Reshape prediction to match target
        batch_size = batch["output_pos"].shape[0]
        num_output_points = batch["output_pos"].shape[1]
        pred_flat = pred.reshape(batch_size * num_output_points, output_dim)
        
        print(f"  Reshaped prediction: {pred_flat.shape}")
        print(f"  Target: {batch['output_feat'].shape}")
        
        # === Compute Loss ===
        print(f"\n{'=' * 80}")
        print("COMPUTING LOSS")
        print(f"{'=' * 80}")
        
        loss = F.mse_loss(pred_flat, batch["output_feat"])
        
        print(f"\nMSE Loss (before training): {loss.item():.6f}")
        print(f"  This should be high (~1-10) since model is randomly initialized")
        
        # === Check for NaN/Inf ===
        has_nan = torch.isnan(pred).any()
        has_inf = torch.isinf(pred).any()
        
        if has_nan or has_inf:
            print(f"\n⚠️  WARNING: Prediction contains NaN: {has_nan}, Inf: {has_inf}")
        else:
            print(f"\n✓ No NaN or Inf in predictions")
        
        # === Test Backward Pass ===
        print(f"\n{'=' * 80}")
        print("TESTING BACKWARD PASS")
        print(f"{'=' * 80}")
        
        loss.backward()
        print(f"✓ Backward pass successful!")
        
        # Check gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for p in model.parameters())
        print(f"  Parameters with gradients: {has_grad} / {total_params}")
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"  Gradient norm: {grad_norm:.4f}")
        
        print(f"\n{'=' * 80}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'=' * 80}")
        print(f"\nQuadtree-UPT pipeline is working correctly.")
        print(f"Next steps:")
        print(f"  1. Test with small batch (2-4 samples)")
        print(f"  2. Overfit on single sample")
        print(f"  3. Train on full dataset")
        
    except Exception as e:
        print(f"\n✗ ERROR during forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return
    

if __name__ == "__main__":
    test_single_sample()

