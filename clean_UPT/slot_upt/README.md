# Slot-Based UPT: Stage 1 Implementation

This module implements Stage 1 of the slot-based UPT architecture, which wraps vanilla UPT with a fixed `[M, N]` slot layer for computational efficiency and time-varying data support.

## Overview

Stage 1 inserts a **fixed slot layer between the raw mesh/particles and the encoder**, and the inverse operation before the decoder's output is scattered back to the physical domain. Everything in the UPT latent-core (encoder transformer+Perceiver, approximator, decoder Perceiver, conditioners, training losses) stays conceptually identical.

### Architecture

```
Raw Cells → Slot Assignment → [B, T, M, N, C]
                              ↓
                    Slot Aggregator
                              ↓
                    [B, T, M, d_latent]
                              ↓
                    Encoder Transformer + Perceiver
                              ↓
                    [B*T, n_latent, d]
                              ↓
                    Latent Core (unchanged)
                              ↓
                    [B*T, n_latent, d]
                              ↓
                    Decoder Transformer + Perceiver
                              ↓
                    [B, T, M, d_latent]
                              ↓
                    Slot Splitter
                              ↓
                    [B, T, M, N, C_out]
                              ↓
                    Scatter to Cells
                              ↓
                    [B*K, T*C_out]
```

## Components

### 1. Slot Assignment (`slot_assignment.py`)

Assigns cells to slots using voxel grid partitioning:

- **`assign_cells_to_slots_voxel_grid()`**: Assigns cells to `[M, N]` slot structure
  - Input: `mesh_pos` [B*K, d_x], `features` [B*K, C]
  - Output: `subnode_feats` [B, T, M, N, C], `subnode_mask` [B, T, M, N], `slot2cell` [B, T, M, N]
  
- **`get_slot_positions()`**: Computes canonical positions for each slot (m, n)
  
- **`scatter_slots_to_cells()`**: Scatters slot features back to original cell positions

### 2. Slot Aggregator (`slot_aggregator.py`)

Aggregates slots to supernodes (encoder side):

- **`MaskedMeanSlotAggregator`**: Masked mean aggregation (baseline)
  - Input: `[B, T, M, N, C]`
  - Output: `[B, T, M, d_latent]`
  - Formula: `Z_m = sum(μ_{m,n} * X_{m,n}) / (ε + sum(μ_{m,n}))`

- **`SlotAggregatorBase`**: Base class for future aggregators (MLP, attention)

### 3. Slot Splitter (`slot_splitter.py`)

Splits supernodes back to slots (decoder side):

- **`SlotSplitter`**: MLP-based splitter with positional embeddings
  - Input: `[B, T, M, d_latent]`
  - Output: `[B, T, M, N, C_out]`
  - Uses positional embeddings for slot positions

### 4. Slot-Based Encoder (`encoders/cfd_slot_pool_transformer_perceiver.py`)

Replaces `CfdPool` with slot aggregator:

- **`CfdSlotPoolTransformerPerceiver`**: Slot aggregator + transformer + Perceiver
  - Handles time-varying data by flattening time dimension
  - Reuses existing transformer/Perceiver blocks

### 5. Slot-Based Decoder (`decoders/cfd_slot_transformer_perceiver.py`)

Replaces query-based decoding with slot splitter:

- **`CfdSlotTransformerPerceiver`**: Transformer + query at supernodes + slot splitter
  - Queries latent at supernode positions
  - Applies slot splitter to generate per-slot outputs

### 6. Composite Model (`models/cfd_slot_simformer_model.py`)

Main model that composes all components:

- **`CfdSlotSimformerModel`**: Based on `CfdSimformerModel`
  - Uses slot-based encoder/decoder
  - Latent core and conditioner unchanged
  - Handles time-varying forward pass and rollouts

### 7. Collator (`collators/cfd_slot_simformer_collator.py`)

Assigns cells to slots during collation:

- **`CfdSlotSimformerCollator`**: Based on `CfdSimformerCollator`
  - Assigns cells to slots using voxel grid
  - Creates slot representation (`subnode_feats`, `subnode_mask`, `slot2cell`)

## Usage

### Configuration

```yaml
model:
  _target_: slot_upt.models.cfd_slot_simformer_model.CfdSlotSimformerModel
  M: 256  # Number of supernodes
  N: 16   # Number of slots per supernode
  encoder:
    _target_: slot_upt.encoders.cfd_slot_pool_transformer_perceiver.CfdSlotPoolTransformerPerceiver
    gnn_dim: 128
    enc_dim: 256
    perc_dim: 256
    enc_depth: 4
    enc_num_attn_heads: 8
    perc_num_attn_heads: 8
    num_latent_tokens: 128
    M: 256
    N: 16
  decoder:
    _target_: slot_upt.decoders.cfd_slot_transformer_perceiver.CfdSlotTransformerPerceiver
    dim: 256
    depth: 4
    num_attn_heads: 8
    perc_dim: 256
    perc_num_attn_heads: 8
    M: 256
    N: 16
    ndim: 2
  latent:
    # ... (same as baseline)
  conditioner:
    # ... (same as baseline)

collator:
  _target_: slot_upt.collators.cfd_slot_simformer_collator.CfdSlotSimformerCollator
  M: 256
  N: 16
  ndim: 2
```

### Forward Pass

```python
from slot_upt.models import CfdSlotSimformerModel

model = CfdSlotSimformerModel(...)

# Forward pass
outputs = model(
    subnode_feats=subnode_feats,      # [B, T, M, N, C]
    subnode_mask=subnode_mask,         # [B, T, M, N]
    slot2cell=slot2cell,               # [B, T, M, N]
    timestep=timestep,                  # [B] or [B*T]
    velocity=velocity,                  # [B] or [B*T]
    batch_idx=batch_idx,               # [B*T*M]
    unbatch_idx=None,
    unbatch_select=None,
)

x_hat = outputs["x_hat"]  # [B*K, T*C_out] scattered to original cells
```

### Rollout

```python
# Autoregressive rollout
predictions = model.rollout(
    subnode_feats=subnode_feats,
    subnode_mask=subnode_mask,
    slot2cell=slot2cell,
    velocity=velocity,
    batch_idx=batch_idx,
    num_rollout_timesteps=10,
    mode="latent",  # or "image"
)
```

## Design Decisions

1. **Time-varying support**: All tensors include T dimension `[B, T, ...]`. Flatten to `[B*T, ...]` before transformer blocks, reshape after.

2. **Modularity**: Each component (aggregator, splitter, assignment) is independent and can be swapped.

3. **Reuse**: Leverages existing transformer blocks, Perceiver blocks, conditioner from UPT.

4. **Slot assignment**: Done in collator (preprocessing) to keep model forward pass clean.

5. **Masking**: Consistent mask handling throughout (zero features, ignore in aggregation).

## Future Extensions

- **Stage 2**: Grid-structure conditioning (add grid embeddings to conditioner)
- **Stage 3**: Learned functional mapping (predict masks from latent)
- **Stage 4**: Generalization & scaling (non-AMR data, scaling studies)
- **Stage 5**: Clean-up & ablations (different aggregators, assignment strategies)

## Notes

- Slot assignment is currently done in the collator. For rollouts, cells need to be re-assigned to slots at each timestep (simplified version in current implementation).
- The slot splitter uses positional embeddings for slot positions. These are computed from the voxel grid structure.
- Mask handling ensures empty slots don't contribute to aggregation and are zeroed out in outputs.

