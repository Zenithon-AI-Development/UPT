"""
Train quadtree-UPT on full TRL2D dataset.

Based on UPT's training configuration for TRL2D but using quadtree encoding.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder_amr_quadtree import EncoderAMRQuadtree
from models.quadtree_upt import QuadtreeUPT
from collators.quadtree_collator import QuadtreeCollator
from datasets.trl2d_quadtree_dataset import TRL2DQuadtreeDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../upt-tutorial'))
from upt.models.approximator import Approximator
from upt.models.decoder_perceiver import DecoderPerceiver
from upt.models.conditioner_timestep import ConditionerTimestep


def train():
    """Train quadtree-UPT on full TRL2D dataset"""
    
    print("=" * 80)
    print("TRAINING QUADTREE-UPT ON TRL2D DATASET")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === Configuration ===
    max_nodes = 5376  # Exact maximum from analysis (covers 100% of samples)
    enc_dim = 128
    num_heads = 4
    input_dim = 4
    output_dim = 4
    batch_size = 4
    num_epochs = 50
    lr = 5e-5
    weight_decay = 0.01
    
    print(f"\nConfiguration:")
    print(f"  max_nodes: {max_nodes}")
    print(f"  enc_dim: {enc_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_epochs: {num_epochs}")
    print(f"  learning_rate: {lr}")
    print(f"  device: {device}")
    
    # === Load Datasets ===
    print(f"\n{'=' * 80}")
    print("LOADING DATASETS")
    print(f"{'=' * 80}")
    
    train_dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="train",
        n_input_timesteps=1,
        max_num_sequences=None,  # Use all training data
    )
    
    val_dataset = TRL2DQuadtreeDataset(
        data_dir="/home/workspace/projects/data/datasets_david/datasets/",
        split="valid",
        n_input_timesteps=1,
        max_num_sequences=None,
    )
    
    print(f"\nDatasets loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Valid: {len(val_dataset)} samples")
    
    # === Collator ===
    train_collator = QuadtreeCollator(
        max_nodes=max_nodes,
        k=2,
        min_size=4,
        max_size=16,
        deterministic=False,  # Random augmentation during training
    )
    
    val_collator = QuadtreeCollator(
        max_nodes=max_nodes,
        k=2,
        min_size=4,
        max_size=16,
        deterministic=True,  # Deterministic for validation
    )
    
    # === DataLoaders ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=0,
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Valid batches: {len(val_loader)}")
    
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
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel created: {num_params:.2f}M parameters")
    
    # === Optimizer & Scheduler ===
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # === Training Loop ===
    print(f"\n{'=' * 80}")
    print("TRAINING")
    print(f"{'=' * 80}")
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
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
            
            # Compute loss
            batch_size_actual = batch["output_pos"].shape[0]
            num_output_points = batch["output_pos"].shape[1]
            pred_flat = pred.reshape(batch_size_actual * num_output_points, output_dim)
            loss = F.mse_loss(pred_flat, batch["output_feat"])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # === Validation ===
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for batch in pbar:
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
                
                # Compute loss
                batch_size_actual = batch["output_pos"].shape[0]
                num_output_points = batch["output_pos"].shape[1]
                pred_flat = pred.reshape(batch_size_actual * num_output_points, output_dim)
                loss = F.mse_loss(pred_flat, batch["output_feat"])
                
                val_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # === Logging ===
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Valid Loss: {avg_val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = "quadtree_upt_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.6f})")
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Final checkpoint saved to: quadtree_upt_best.pt")


if __name__ == "__main__":
    train()

