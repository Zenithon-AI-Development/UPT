"""
Overfit quadtree-UPT on a single sample to verify learning works.

This tests that the model can learn to predict one sample perfectly,
which validates the entire training pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from models.encoder_amr_quadtree import EncoderAMRQuadtree
from models.quadtree_upt import QuadtreeUPT
from collators.quadtree_collator import QuadtreeCollator
from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../upt-tutorial'))
from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.conditioner_timestep import ConditionerTimestep


def overfit_single_sample():
    """Overfit on single sample"""
    
    print("=" * 80)
    print("OVERFITTING QUADTREE-UPT ON SINGLE SAMPLE")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === Configuration ===
    max_nodes = 4096
    enc_dim = 128
    num_heads = 4
    input_dim = 4
    output_dim = 4
    num_steps = 2000
    lr = 1e-3
    
    print(f"\nConfiguration:")
    print(f"  max_nodes: {max_nodes}")
    print(f"  enc_dim: {enc_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  Training steps: {num_steps}")
    print(f"  Learning rate: {lr}")
    
    # === Load Dataset ===
    print(f"\n{'=' * 80}")
    print("LOADING DATASET")
    print(f"{'=' * 80}")
    
    dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=1,  # Single sample
    )
    
    sample = dataset[0]
    
    # === Collator ===
    collator = QuadtreeCollator(
        max_nodes=max_nodes,
        k=2,
        min_size=4,
        max_size=16,
        deterministic=True,
    )
    
    # === Build Model ===
    print(f"\n{'=' * 80}")
    print("BUILDING MODEL")
    print(f"{'=' * 80}")
    
    num_timesteps = 101
    conditioner = ConditionerTimestep(dim=enc_dim, num_timesteps=num_timesteps)
    
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
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.2f}M parameters")
    
    # === Optimizer ===
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # === Training Loop ===
    print(f"\n{'=' * 80}")
    print("TRAINING")
    print(f"{'=' * 80}")
    
    model.train()
    losses = []
    
    print(f"\nStep | Loss")
    print("-" * 30)
    
    for step in range(num_steps):
        # Collate sample (same sample every time for overfitting)
        batch = collator([sample])
        
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward
        pred = model(
            node_feat=batch["node_feat"],
            node_pos=batch["node_pos"],
            depth=batch["depth"],
            batch_idx=batch["batch_idx"],
            output_pos=batch["output_pos"],
            timestep=batch["timestep"],
        )
        
        # Reshape and compute loss
        batch_size = batch["output_pos"].shape[0]
        num_output_points = batch["output_pos"].shape[1]
        pred_flat = pred.reshape(batch_size * num_output_points, output_dim)
        
        loss = F.mse_loss(pred_flat, batch["output_feat"])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # Log
        if step % 100 == 0 or step == num_steps - 1:
            print(f"{step:4d} | {loss.item():.6f}")
    
    # === Results ===
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Reduction:    {reduction:.1f}%")
    
    if final_loss < 0.01:
        print(f"\n✓ SUCCESS! Model can overfit (loss < 0.01)")
        print(f"  This confirms the learning pipeline works correctly.")
    elif final_loss < 0.1:
        print(f"\n⚠ PARTIAL SUCCESS: Loss reduced but not converged (loss < 0.1)")
        print(f"  Consider training longer or adjusting learning rate.")
    else:
        print(f"\n✗ FAILURE: Model did not overfit (loss > 0.1)")
        print(f"  There may be an issue with the model or training setup.")
    
    return losses


if __name__ == "__main__":
    losses = overfit_single_sample()

