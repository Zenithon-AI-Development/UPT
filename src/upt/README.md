# UPT Core - Stage 0 Baseline

Clean modular implementation of Universal Physics Transformers for Stage 0 baseline.

## Structure

```
upt/
├── core/              # Core modules
│   ├── encoder.py    # Base encoder interface: data → [B, M, d]
│   ├── latent_core.py # Latent core: transformer over M tokens
│   ├── decoder.py    # Base decoder interface: [B, M, d] → outputs
│   ├── conditioner.py # Conditioners: time, params, etc.
│   └── model.py      # Clean UPT model that composes modules
├── benchmarking/      # Benchmarking infrastructure
│   ├── timing.py     # Wall-clock timing per step
│   ├── memory.py     # Peak GPU memory tracking
│   └── metrics.py    # Accuracy and rollout stability metrics
└── train_baseline.py # One-command training script
```

## Features

### ✅ Modular Architecture

- **Encoder**: `data → [B, M, d]` - Compresses input to latent space
- **Latent Core**: `[B, M, d] → [B, M, d]` - Temporal propagation
- **Decoder**: `[B, M, d] → outputs` - Reconstructs output
- **Conditioner**: Provides global/per-token conditioning

### ✅ Hooks for Extensibility

- **Extra per-token conditioning**: Hook for grid-structure embeddings
- **Multiple encoders/decoders**: Support via config flag
- **Clean API**: Easy to swap components

### ✅ Standardized Benchmarks

- **Wall-clock timing**: Per-step timing for encoder/latent/decoder
- **Peak GPU memory**: Memory tracking per component
- **Accuracy metrics**: L2, L1, spectral, relative errors
- **Rollout stability**: Metrics for long-term rollout stability

## Usage

### One-Command Training

```bash
python -m upt.train_baseline --config configs/upt_baseline.yaml --benchmark
```

### Creating a Model

```python
from upt.core import UPTModel, BaseEncoder, TransformerLatentCore, BaseDecoder, BaseConditioner

# Create components
encoder = MyEncoder(input_dim=7, latent_dim=128, num_latent_tokens=128)
latent_core = TransformerLatentCore(dim=128, depth=4, num_heads=4)
decoder = MyDecoder(latent_dim=128, output_dim=7)
conditioner = MyConditioner(cond_dim=128)

# Create model
model = UPTModel(
    encoder=encoder,
    latent_core=latent_core,
    decoder=decoder,
    conditioner=conditioner,
)
```

### Using Hooks

```python
# Register extra per-token conditioning hook
def grid_structure_hook(latent, condition, grid_pos, **kwargs):
    # Compute grid-structure embeddings
    B, M, d = latent.shape
    grid_embeddings = compute_grid_embeddings(grid_pos)  # [B, M, cond_dim]
    return grid_embeddings

model.register_extra_token_conditioning_hook(grid_structure_hook)
```

### Multiple Encoders/Decoders

```python
# Create multiple encoders/decoders
encoders = [encoder1, encoder2, encoder3]
decoders = [decoder1, decoder2]

model = UPTModel(
    encoder=encoders[0],
    latent_core=latent_core,
    decoder=decoders[0],
    use_multiple_encoders=True,
    use_multiple_decoders=True,
    encoders=encoders,
    decoders=decoders,
)

# Use specific encoder/decoder
outputs = model(x, encoder_idx=1, decoder_idx=0)
```

### Benchmarking

```python
from upt.benchmarking import MetricsCollector

# Create metrics collector
metrics = MetricsCollector(
    enable_timing=True,
    enable_memory=True,
    enable_accuracy=True,
    enable_stability=True,
)

# During training
metrics.timing.start('forward')
outputs = model(x)
metrics.timing.end('forward')

metrics.accuracy.update(pred, target)

# Print summary
metrics.print_summary()
```

## API Reference

### BaseEncoder

```python
class BaseEncoder(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode input data to latent representation.
        
        Args:
            x: Input data (format depends on encoder type)
            condition: Global conditioning [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
            **kwargs: Additional encoder-specific arguments
        
        Returns:
            latent: [B, M, d] latent representation
        """
```

### BaseLatentCore

```python
class BaseLatentCore(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Transform latent representation.
        
        Args:
            x: [B, M, d] latent tokens
            condition: Global conditioning [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
            **kwargs: Additional arguments
        
        Returns:
            x: [B, M, d] transformed latent tokens
        """
```

### BaseDecoder

```python
class BaseDecoder(nn.Module):
    def forward(
        self,
        latent: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        extra_token_conditioning: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent representation to output.
        
        Args:
            latent: [B, M, d] latent tokens
            query_pos: Query positions for output (format depends on decoder type)
            condition: Global conditioning [B, cond_dim] (optional)
            extra_token_conditioning: Per-token conditioning [B, M, cond_dim] (optional)
            **kwargs: Additional decoder-specific arguments
        
        Returns:
            output: Decoded output (format depends on decoder type)
        """
```

### BaseConditioner

```python
class BaseConditioner(nn.Module):
    def forward(
        self,
        timestep: Optional[torch.Tensor] = None,
        velocity: Optional[torch.Tensor] = None,
        params: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute global conditioning vector.
        
        Args:
            timestep: [B] timestep indices
            velocity: [B] or [B, ...] velocity or other scalar/vector features
            params: Dict of additional parameters
            **kwargs: Additional arguments
        
        Returns:
            condition: [B, cond_dim] global conditioning vector
        """
```

## Next Steps

1. **Implement concrete encoders/decoders**: Wrap existing encoders/decoders to match the base interfaces
2. **Add dataset loaders**: Create dataset loaders for your target datasets
3. **Reproduce baseline**: Train on target dataset(s) and reproduce strong baseline
4. **Extend with hooks**: Add grid-structure embeddings and other per-token conditioning

## Milestone: UPT-baseline

✅ Clean modular structure  
✅ Hooks for extensibility  
✅ Standardized benchmarks  
✅ One-command training script  
✅ Clean API for encoder/decoder/conditioners  

**Status**: Ready for implementation of concrete encoders/decoders and dataset loaders.

