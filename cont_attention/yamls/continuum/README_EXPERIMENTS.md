# FANO Continuum Attention Experiments

This folder contains configurations for testing different FANO encoder architectures.

## 🎯 Two Pooling Philosophies

### **A) Direct Pooling** (Custom - NOT from FANO paper)
**Files:** `*_baseline.yaml`

**Pipeline:**
```
Patches (64, 16, 16, 192)
  ↓ Pool each patch directly (spatial avg, learned, or attention)
(batch, 64, 192) tokens
  ↓ Perceiver cross-attention
(batch, 256, 192) latent
```

**Pros:**
- Simple, efficient
- Direct compression from patches to tokens
- No reconstruction overhead

**Cons:**
- Not faithful to FANO paper
- May lose information within patches


### **B) Grid Reconstruction** (Faithful to FANO paper)
**Files:** `*_faithful.yaml`

**Pipeline:**
```
Patches (64, 16, 16, 192)
  ↓ Reshape back to full grid (like original FANO)
(batch, 128, 128, 192) full field
  ↓ Uniform grid subsampling (every Nth point)
(batch, 64, 192) tokens
  ↓ Perceiver cross-attention
(batch, 256, 192) latent
```

**Pros:**
- Faithful to FANO paper's design philosophy
- Reconstructs continuous field representation
- Can sample flexibly (64, 256, or more points)

**Cons:**
- Extra reshape operations
- Slight computational overhead


---

## 📁 Configuration Files

### **Z-Pinch Dataset (128×128 grid)**

#### 1. `zpinch_fano_baseline.yaml` 
- **Pooling:** Direct (`spatial_avg`)
- **Patches:** 8×8 = 64 patches, each 16×16
- **FANO layers:** 2
- **d_model:** 192
- **Latent:** 256 tokens

#### 2. `zpinch_fano_faithful.yaml` 
- **Pooling:** Grid reconstruction (`grid_uniform`)
- **Grid samples:** 64 (one per patch)
- **Patches:** 8×8 = 64 patches, each 16×16
- **FANO layers:** 2
- **d_model:** 192
- **Latent:** 256 tokens

### **TRL2D Dataset (192×256 grid)** 

#### 3. `trl2d_fano_baseline.yaml`
- **Pooling:** Direct (`spatial_avg`)
- **Patches:** 8×8 = 64 patches
- **FANO layers:** 2
- **d_model:** 256 (higher than zpinch)
- **Latent:** 256 tokens

#### 4. `trl2d_fano_faithful.yaml`
- **Pooling:** Grid reconstruction (`grid_uniform`)
- **Grid samples:** 64
- **Patches:** 8×8 = 64 patches
- **FANO layers:** 2
- **d_model:** 256
- **Latent:** 256 tokens

---

## 🔬 Experimental Design

### **Primary Comparison:**
**Baseline vs Faithful** (for each dataset)
- Same FANO blocks, same patch count
- Different pooling strategy
- Tests: "Does grid reconstruction help?"

### **Secondary Comparisons:**
- vs. Standard UPT encoder (graph-based, 2048 tokens)
- Different patch counts: 64 vs 256
- Different FANO depths: 2 vs 4 layers

---

## 🎚️ Key Hyperparameters to Tune

All controllable via YAML `vars` section:

### **FANO Architecture:**
- `num_fano_layers`: 2, 4, 6
- `fano_num_heads`: 6, 8, 12
- `fourier_modes`: 9 (default for 16×16 patches)

### **Patching:**
- `num_patches_h`, `num_patches_w`: Controls patch count
  - 8×8 = 64 patches (baseline)
  - 16×16 = 256 patches (high-res)
  - 4×4 = 16 patches (coarse)

### **Pooling (for faithful configs):**
- `num_grid_samples`: 64, 256, 1024
  - More samples = more information to perceiver
  - More samples = more expensive attention

### **Pooling Type:**
- Direct: `spatial_avg`, `learned`, `attention`
- Grid reconstruction: `grid_uniform`, `grid_adaptive`

---

## 🚀 How to Run

### **Single experiment:**
```bash
python main_train.py --config yamls/continuum/zpinch_fano_baseline.yaml
```

### **Compare baseline vs faithful:**
```bash
# Direct pooling
python main_train.py --config yamls/continuum/zpinch_fano_baseline.yaml

# Grid reconstruction (faithful to FANO)
python main_train.py --config yamls/continuum/zpinch_fano_faithful.yaml
```

### **Quick test (single-batch overfit):**
```bash
python test_minimal_overfit.py --config yamls/continuum/zpinch_fano_baseline.yaml
```

---

## 📊 What to Compare

### **Metrics:**
1. **Accuracy:** `loss/online/x_hat/E1` (rollout error)
2. **Speed:** Time per epoch
3. **Memory:** Peak GPU usage
4. **Stability:** Training loss curves

### **Key Questions:**
1. Does grid reconstruction help? (faithful vs baseline)
2. How does FANO compare to standard UPT? (graph pooling vs patches)
3. What's the optimal patch count? (64 vs 256)
4. Do we need more FANO layers? (2 vs 4)

---

## 📝 Notes

### **Grid Dimensions:**
- **Z-pinch:** 128×128 (verified)
- **TRL2D:** 192×256 (placeholder - **verify actual dimensions!**)

### **Attention Module:**
The `modules/attention/fano_attention.py` is **pure FANO** from the continuum attention paper - unchanged regardless of pooling strategy.

### **Pooling Module:**
The `modules/pooling/patch_pooling.py` contains both approaches:
- Direct pooling classes (custom)
- Grid reconstruction classes (faithful to FANO)

### **Future Experiments:**
Can easily add:
- Different Fourier modes (5, 7, 12)
- Hybrid: CfdPool → FANO
- Dense perceiver (all 16K grid points)
- Adaptive grid sampling

