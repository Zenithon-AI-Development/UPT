#!/usr/bin/env python3
"""
One-command training script for UPT-baseline.

Usage:
    python -m upt.train_baseline --config configs/upt_baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from upt.core import UPTModel, BaseEncoder, BaseLatentCore, BaseDecoder, BaseConditioner
from upt.benchmarking import MetricsCollector


def parse_args():
    parser = argparse.ArgumentParser(description="Train UPT-baseline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking (timing, memory, metrics)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    return parser.parse_args()


def load_config(config_path: str):
    """Load config from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str) -> UPTModel:
    """
    Create UPT model from config.
    
    Config structure:
        model:
            encoder: {...}
            latent_core: {...}
            decoder: {...}
            conditioner: {...}  # optional
    """
    # For now, we'll create a simple example
    # In practice, you'd load actual encoder/decoder implementations
    from upt.core import TransformerLatentCore
    
    # Create components
    encoder = create_encoder(config['model']['encoder'])
    latent_core = create_latent_core(config['model']['latent_core'])
    decoder = create_decoder(config['model']['decoder'])
    conditioner = None
    if 'conditioner' in config['model']:
        conditioner = create_conditioner(config['model']['conditioner'])
    
    # Create model
    model = UPTModel(
        encoder=encoder,
        latent_core=latent_core,
        decoder=decoder,
        conditioner=conditioner,
    ).to(device)
    
    return model


def create_encoder(config: dict) -> BaseEncoder:
    """Create encoder from config."""
    # This is a placeholder - you'd implement actual encoders
    # For now, return a dummy encoder
    class DummyEncoder(BaseEncoder):
        def forward(self, x, condition=None, extra_token_conditioning=None, **kwargs):
            B = x.shape[0] if x.ndim > 0 else 1
            M = self.num_latent_tokens or 128
            d = self.latent_dim
            return torch.randn(B, M, d, device=x.device if isinstance(x, torch.Tensor) else None)
    
    return DummyEncoder(
        input_dim=config.get('input_dim', 7),
        latent_dim=config.get('latent_dim', 128),
        num_latent_tokens=config.get('num_latent_tokens', 128),
    )


def create_latent_core(config: dict) -> BaseLatentCore:
    """Create latent core from config."""
    from upt.core import TransformerLatentCore
    
    return TransformerLatentCore(
        dim=config.get('dim', 128),
        depth=config.get('depth', 4),
        num_heads=config.get('num_heads', 4),
        cond_dim=config.get('cond_dim', None),
        drop_path_rate=config.get('drop_path_rate', 0.0),
    )


def create_decoder(config: dict) -> BaseDecoder:
    """Create decoder from config."""
    # This is a placeholder - you'd implement actual decoders
    class DummyDecoder(BaseDecoder):
        def forward(self, latent, query_pos=None, condition=None, extra_token_conditioning=None, **kwargs):
            B, M, d = latent.shape
            output_dim = self.output_dim
            # Simple linear projection for now
            if query_pos is not None:
                num_outputs = query_pos.shape[1] if query_pos.ndim > 1 else query_pos.shape[0]
            else:
                num_outputs = M
            return torch.randn(B, num_outputs, output_dim, device=latent.device)
    
    return DummyDecoder(
        latent_dim=config.get('latent_dim', 128),
        output_dim=config.get('output_dim', 7),
    )


def create_conditioner(config: dict) -> BaseConditioner:
    """Create conditioner from config."""
    # This is a placeholder - you'd implement actual conditioners
    class DummyConditioner(BaseConditioner):
        def forward(self, timestep=None, velocity=None, params=None, **kwargs):
            B = timestep.shape[0] if timestep is not None else 1
            return torch.randn(B, self.cond_dim)
    
    return DummyConditioner(
        cond_dim=config.get('cond_dim', 128),
    )


def train_epoch(
    model: UPTModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    metrics: MetricsCollector = None,
    epoch: int = 0,
):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            batch = batch.to(device)
        
        # Forward pass
        if metrics:
            metrics.timing.start('forward')
            metrics.memory.record('before_forward')
        
        # Extract inputs (this depends on your dataset format)
        x = batch.get('x', batch.get('input', batch))
        target = batch.get('target', batch.get('y', None))
        query_pos = batch.get('query_pos', batch.get('pos', None))
        timestep = batch.get('timestep', torch.zeros(x.shape[0], device=device, dtype=torch.long))
        
        outputs = model(
            x=x,
            query_pos=query_pos,
            timestep=timestep,
        )
        
        pred = outputs['output']
        
        if metrics:
            metrics.timing.end('forward')
            metrics.memory.record('after_forward')
        
        # Compute loss
        if target is not None:
            loss = criterion(pred, target)
            
            # Backward pass
            if metrics:
                metrics.timing.start('backward')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if metrics:
                metrics.timing.end('backward')
                metrics.memory.record('after_backward')
            
            # Update metrics
            if metrics and metrics.accuracy:
                metrics.accuracy.update(pred, target)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config(args.config)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, args.device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Create dataset and dataloader
    # This is a placeholder - you'd load your actual dataset
    logger.info("Creating dataset...")
    # dataset = create_dataset(config['dataset'])
    # dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # For now, create a dummy dataloader
    class DummyDataset:
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {
                'x': torch.randn(1, 7, 64, 64),
                'target': torch.randn(1, 7, 64, 64),
                'query_pos': torch.randn(64*64, 2),
                'timestep': torch.tensor(0),
            }
    
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', 4), shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.01),
    )
    
    # Create loss function
    criterion = nn.MSELoss()
    
    # Create metrics collector
    metrics = MetricsCollector(
        enable_timing=args.benchmark,
        enable_memory=args.benchmark,
        enable_accuracy=args.benchmark,
        enable_stability=args.benchmark,
    ) if args.benchmark else None
    
    # Training loop
    num_epochs = config.get('num_epochs', 10)
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if metrics:
            metrics.reset()
        
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=args.device,
            metrics=metrics,
            epoch=epoch,
        )
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        if metrics:
            metrics.print_summary()
    
    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

