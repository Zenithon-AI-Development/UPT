# Files to Copy for Fourier Attention Neural Operator

This document identifies the minimal files needed to port the Fourier Attention mechanism to your UPT repository.

## Core File (Required)

### 1. `models/transformer_custom.py` 
**Extract only the following classes (lines 486-645):**

#### Essential Classes for Fourier Attention:
1. **`SpectralConv2d_Attention`** (lines 486-533)
   - The Fourier convolution layer that transforms Q/K/V using FFT
   - Key method: `compl_mul2d()` for complex multiplication in frequency domain
   
2. **`MultiheadAttention_Conv`** (lines 535-569)
   - Multi-head attention wrapper that uses `SpectralConv2d_Attention` for Q/K/V projection
   - Combines multiple attention heads
   
3. **`ScaledDotProductAttention_Conv`** (lines 571-588)
   - **THE CONTINUUM ATTENTION MECHANISM**
   - Computes attention scores using einsum: `"bnpxyd,bnqxyd->bnpq"`
   - This is where patch-to-patch attention is computed across spatial dimensions
   
4. **`TransformerEncoderLayer_Conv`** (lines 590-645)
   - Complete encoder layer with Fourier attention
   - Includes feedforward network, layer norms, residual connections
   
5. **`TransformerEncoder_Operator`** (lines 296-304) - **Also needed!**
   - Simple encoder stack that applies multiple layers
   - Used by FANO implementation

## Required Imports (from transformer_custom.py top)

You'll need these imports at the top of your file:
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
```

## Dependencies

All dependencies are standard PyTorch:
- `torch.fft.rfft2()` - Real-valued 2D FFT
- `torch.fft.irfft2()` - Inverse real-valued 2D FFT
- `torch.cfloat` - Complex float dtype
- `torch.einsum()` - Einstein summation for attention computation

## Optional but Helpful Reference

### `models/FANO/FANO_pytorch.py`
- Example of how to use `TransformerEncoderLayer_Conv`
- Shows input preprocessing (patch chopping)
- Demonstrates the full forward pass

## Key Design Points

### Input Shape Convention:
The Fourier attention expects input of shape:
```
(batch, num_patches, patch_size, patch_size, d_model)
```

### Attention Computation:
The continuum attention mechanism computes:
- Query, Key, Value via Fourier transforms (FFT → multiply → iFFT)
- Attention scores: `scores = einsum("bnpxyd,bnqxyd->bnpq", query, key) / scale`
- Attention weights: `softmax(scores, dim=-1)` - attention between patches
- Output: `einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)`

The scale factor is: `scale = sqrt(im_size^4)`

### Fourier Modes:
- `modes1`, `modes2`: Number of Fourier modes to keep (typically `patch_size//2+1`)
- Only low-frequency modes are used for efficiency

## Minimal Copy Strategy

**Recommended:** Extract just the 4 classes listed above (lines 486-645) plus `TransformerEncoder_Operator` (lines 296-304) into a new file in your UPT repo, e.g.:
- `upt/models/fourier_attention.py`

Then adapt the input/output shape handling to match your UPT model's conventions.

## Notes

- The continuum attention mechanism is specifically in `ScaledDotProductAttention_Conv.forward()` (line 579-588)
- The Fourier Q/K/V transformation happens in `SpectralConv2d_Attention.forward()` (line 509-533)
- The attention operates at the **patch level**, not pixel level - patches attend to other patches
- Each patch has spatial dimensions (patch_size × patch_size), and attention weights are computed per patch pair

