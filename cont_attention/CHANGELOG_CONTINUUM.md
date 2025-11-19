# Changelog: Continuum Attention Integration

This document tracks all changes made to integrate FANO (Fourier Attention Neural Operator) continuum attention mechanisms into the UPT codebase.

**Based on:** "Continuum Attention for Neural Operators" (Calvello et al., 2024)  
**Paper:** https://arxiv.org/abs/2406.06486  
**Original Code:** https://github.com/EdoCalvello/TransformerNeuralOperators

---

## 📅 Initial Integration

### **New Modules Added**

#### 1. **`modules/attention/`** (NEW folder)
Fourier attention mechanisms from FANO paper.

**Files:**
- `__init__.py` - Module exports
- `fano_attention.py` - Complete FANO implementation
  - `TransformerEncoder_Operator` - Layer stacking
  - `SpectralConv2d_Attention` - Fourier Q/K/V projection per head
  - `MultiheadAttention_Conv` - Multi-head wrapper
  - `ScaledDotProductAttention_Conv` - **Continuum attention core** (patch-to-patch)
  - `TransformerEncoderLayer_Conv` - Complete FANO encoder layer

**Purpose:** Pure FANO implementation from paper, no modifications.

---

#### 2. **`modules/pooling/`** (NEW folder)
Strategies for converting FANO patches to tokens for perceiver.

**Files:**
- `__init__.py` - Module exports
- `patch_pooling.py` - Five pooling strategies:
  
  **Direct Pooling (Custom):**
  - `SpatialAvgPooling` - Spatial average per patch
  - `LearnedPooling` - Learnable probe MLP
  - `AttentionPooling` - Query-based pooling
  
  **Grid Reconstruction (Faithful to FANO):**
  - `GridUniformSampling` - Reshape to full grid → uniform subsample
  - `GridAdaptiveSampling` - Reshape to full grid → learned top-K

**Purpose:** Bridge between FANO (continuous field) and UPT perceiver (discrete tokens).

---

#### 3. **`utils/verify_grid_ordering.py`** (NEW file)
Verification script to check data ordering assumptions.

**Purpose:**
- Verify zpinch/trl2d data is in proper grid order
- Test patch decomposition and reconstruction
- Confirm simple reshape operations will work

---

### **New Experiment Configurations**

#### 4. **`yamls/continuum/`** (NEW folder)
Experiment configurations for FANO encoder testing.

**Files:**
- `README_EXPERIMENTS.md` - Documentation of experimental setup
- `zpinch_fano_baseline.yaml` - Z-pinch with direct pooling
- `zpinch_fano_faithful.yaml` - Z-pinch with grid reconstruction
- `trl2d_fano_baseline.yaml` - TRL2D with direct pooling
- `trl2d_fano_faithful.yaml` - TRL2D with grid reconstruction

**Key Configuration Parameters:**
```yaml
vars:
  # FANO architecture
  num_fano_layers: 2          # Number of FANO blocks
  fano_num_heads: 6           # Attention heads
  fourier_modes: 9            # Fourier modes per dimension
  
  # Patching
  num_patches_h: 8            # 8×8 = 64 patches
  num_patches_w: 8
  
  # Pooling strategy
  patch_pooling_type: spatial_avg        # or grid_uniform
  num_grid_samples: 64                   # for grid reconstruction
  
  # Disable graph components
  num_supernodes: null        # ← Disables random sampling
  radius_graph_r: null
```

---

### **Modified Files**

None yet! All changes are additive (new files only).

---

## 🎯 **Experimental Design**

### **Two Pooling Philosophies:**

**A) Direct Pooling** (`*_baseline.yaml`)
```
Patches (64, 16×16, d) → Pool each patch → 64 tokens → Perceiver → 256 latent
```
- Simple, efficient
- Custom (not from FANO paper)

**B) Grid Reconstruction** (`*_faithful.yaml`)
```
Patches (64, 16×16, d) → Reshape to full grid (128×128) → 
  Subsample → 64 points → Perceiver → 256 latent
```
- Faithful to FANO paper design
- Reconstructs continuous field first

### **Key Research Question:**
Does grid reconstruction (FANO-faithful) improve over direct pooling?

---

## 📋 **Next Steps (Not Yet Implemented)**

### **To Complete Integration:**

1. **`modules/gno/patch_builder.py`** (TODO)
   - Convert flat mesh points → 2D grid → patches
   - Assumes row-major `(h w)` ordering (verified in datasets)

2. **`models/encoders/cfd_fano_perceiver.py`** (TODO)
   - Main FANO encoder orchestration
   - Patch builder → FANO blocks → Pooling → Perceiver
   - Switchable pooling via config

3. **Registration** (TODO)
   - Update `modules/gno/__init__.py`
   - Update `models/encoders/__init__.py`

4. **Testing** (TODO)
   - Forward pass test with dummy data
   - Single-batch overfit test
   - Full training comparison vs baseline

---

## 🔬 **Datasets Verified**

### **Z-Pinch (128×128 grid)**
- ✅ Data is row-major ordered `(h w)`
- ✅ No randomness (when `num_supernodes=null`)
- ✅ 4 timesteps × 7 channels = 28 input features
- ✅ Ready for grid-based FANO

### **TRL2D (192×256 grid - to verify)**
- ✅ Data is row-major ordered `(h w)`
- ✅ No randomness (when `num_supernodes=null`)
- ⚠️ Grid dimensions to verify: H=192, W=256 (placeholder)
- ✅ Ready for grid-based FANO

---

## 📚 **References**

```bibtex
@article{Calvello2024Continuum,
  title={Continuum Attention for Neural Operators},
  author={Calvello, Edoardo and Kovachki, Nikola B and Levine, Matthew E and Stuart, Andrew M},
  journal={arXiv preprint arXiv:2406.06486},
  year={2024}
}
```

---

## 🗂️ **File Structure Summary**

```
cont_attention/
├── modules/
│   ├── attention/           [NEW] FANO attention mechanisms
│   │   ├── __init__.py
│   │   └── fano_attention.py
│   └── pooling/             [NEW] Patch pooling strategies
│       ├── __init__.py
│       └── patch_pooling.py
├── yamls/
│   └── continuum/           [NEW] FANO experiment configs
│       ├── README_EXPERIMENTS.md
│       ├── zpinch_fano_baseline.yaml
│       ├── zpinch_fano_faithful.yaml
│       ├── trl2d_fano_baseline.yaml
│       └── trl2d_fano_faithful.yaml
├── utils/
│   └── verify_grid_ordering.py  [NEW] Data ordering verification
├── FANO/                    [Reference only - not used in code]
│   ├── fourier_attention_standalone.py
│   ├── fourier_attention_exact_copy.py
│   └── FOURIER_ATTENTION_FILES_TO_COPY.md
└── CHANGELOG_CONTINUUM.md   [NEW] This file
```

---

## ⚠️ **Important Notes**

1. **FANO folder is for reference only** - The actual code is in `modules/attention/`
2. **All changes are additive** - No existing UPT code modified
3. **Experiments are isolated** - Can run baseline and FANO side-by-side
4. **Dataset compatibility** - Only works with gridded data (zpinch, trl2d)
5. **Set `num_supernodes: null`** - Required to preserve grid structure

---

## 🔧 **Configuration Tips**

### **To switch between pooling strategies:**
```yaml
# Direct pooling
pooling_type: spatial_avg
# No num_grid_samples needed

# Grid reconstruction (faithful)
pooling_type: grid_uniform
num_grid_samples: 64
```

### **To change patch resolution:**
```yaml
# Coarser patches (16 total, 32×32 each)
num_patches_h: 4
num_patches_w: 4

# Finer patches (256 total, 8×8 each)
num_patches_h: 16
num_patches_w: 16
```

### **To adjust FANO depth:**
```yaml
# Shallow (faster)
num_fano_layers: 2

# Deep (more capacity)
num_fano_layers: 4
```

