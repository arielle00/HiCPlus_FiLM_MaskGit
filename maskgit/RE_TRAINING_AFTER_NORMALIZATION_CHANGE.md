# Re-Training After Normalization Change

## Summary

After updating `hic_to_npy.py` to match the normalization in `hicplus/trainConvNet.py`, you need to **re-generate all data files and re-train all models** because:

1. **Data normalization changed**: The `.npy` files now use different preprocessing (clipping to 100, LR scaling by 16)
2. **VQGAN depends on data**: VQGAN was trained on the old normalization, so it needs retraining
3. **MaskGIT depends on VQGAN**: MaskGIT uses VQGAN encodings, so it also needs retraining

## Pipeline Order

The training pipeline has these dependencies:
```
.hic files
    ↓
hic_to_npy.py → chrXX_hr.npy, chrXX_lr.npy (per chromosome)
    ↓
combine_npy.py → hic_vqgan_train_hr.npy, hic_vqgan_train_lr.npy (combined)
    ↓
train_vqgan_*.py → VQGAN checkpoint (e.g., vae.best.pt)
    ↓
train_maskgit_*.py → MaskGIT checkpoint
```

## Step-by-Step Re-Training

### Step 1: Re-generate Per-Chromosome .npy Files

For each chromosome you want to train on, run:

```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 hic_to_npy.py \
  --hic /path/to/your/file.hic \
  --chrom <CHROMOSOME> \
  --resolution 10000 \
  --tile-size 128 \
  --stride 64 \
  --norm KR \
  --frac 0.10 \
  --output chr_data_lowres/chr<CHROMOSOME>_lr.npy \
  --output-hr chr_data_highres/chr<CHROMOSOME>_hr.npy
```

**Important**: The normalization now matches `trainConvNet.py`:
- **HR tiles**: `raw_counts → clip(100) → log1p()`
- **LR tiles**: `raw_counts → multiply(16) → clip(100) → log1p()`

**Files affected**: All `chr*_hr.npy` and `chr*_lr.npy` files in your data directories.

### Step 2: Re-combine Chromosome Files

Combine all per-chromosome files into single training files:

```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 combine_npy.py \
  --hr-dir chr_data_highres \
  --lr-dir chr_data_lowres \
  --mode both \
  --out-hr hic_vqgan_train_hr.npy \
  --out-lr hic_vqgan_train_lr.npy
```

**Files affected**: 
- `hic_vqgan_train_hr.npy` (will be overwritten)
- `hic_vqgan_train_lr.npy` (will be overwritten)

### Step 3: Re-generate Coordinate Files (if needed)

If you use coordinate-based filtering, regenerate coordinate files:

```bash
cd /home/012002744/hicplus_thesis/maskgit

# For HR coordinates
python3 create_coords_for_combined.py \
  --npy-file hic_vqgan_train_hr.npy \
  --chr-dir chr_data_highres \
  --kind hr

# For LR coordinates
python3 create_coords_for_combined.py \
  --npy-file hic_vqgan_train_lr.npy \
  --chr-dir chr_data_lowres \
  --kind lr
```

**Files affected**:
- `hic_vqgan_train_hr.npy.coords.npy`
- `hic_vqgan_train_lr.npy.coords.npy`

### Step 4: Re-train VQGAN

VQGAN must be retrained because it was trained on data with the old normalization.

**Option A: Standard VQGAN training**
```bash
cd /home/012002744/hicplus_thesis/maskgit/muse_vqgan

python3 train_vqgan_maskgit.py
```

**Option B: VQGAN with improved sampling**
```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_vqgan_improved.py
```

**Checkpoints affected**: 
- `vqgan_results_2layers_4096_removedOEsignal/vae.best.pt` (or similar)
- Any other VQGAN checkpoints you're using

### Step 5: Re-train MaskGIT

MaskGIT must be retrained because it depends on VQGAN encodings, which changed when VQGAN was retrained.

**Option A: Conditional MaskGIT**
```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_maskgit_cond.py
```

**Option B: Conditional MaskGIT with improved sampling**
```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_maskgit_cond_improved.py
```

**Option C: Unconditional MaskGIT**
```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_maskgit_uncond.py
```

**Checkpoints affected**: All MaskGIT checkpoints that use the retrained VQGAN.

## Important: Inference Normalization

**Critical**: For inference, you use HiCPlus+FiLM prediction outputs (`.npy` files) as inputs to MaskGIT. However:

1. **HiCPlus+FiLM outputs are in counts space**: The `pred_chromosome.py` script converts predictions from log1p space back to counts space before saving (via `_postprocess_pred_tensor`).

2. **MaskGIT expects log1p space**: The inference script `infer_gated_maskgit_log1p.py` expects all inputs in log1p space (see line 12: "All arrays are assumed log1p(counts)").

3. **Conversion needed**: Before feeding HiCPlus+FiLM predictions to MaskGIT, you must convert them:
   ```python
   # HiCPlus+FiLM output (counts space)
   proposal_counts = np.load("pred_chr20_patches.npy")  # [N, 1, H, W] in counts
   
   # Convert to MaskGIT expected space (matching training normalization)
   proposal_log1p = np.log1p(np.clip(proposal_counts, 0, 100))  # clip to 100, then log1p
   ```

4. **After retraining**: Once you retrain MaskGIT with the new normalization, the conversion must match:
   - **HR/Proposal**: `counts → clip(100) → log1p()`
   - **LR**: `counts → multiply(16) → clip(100) → log1p()` (if you're creating LR from counts)

**Note**: If your inference pipeline already handles this conversion, make sure it uses the correct normalization constants (clip_max=100, down_sample_ratio=16) to match the retrained MaskGIT.

## What You Can Skip

- **HiCPlus+FiLM models**: These use their own data pipeline (`hicplus/trainConvNet.py`) and are not affected by changes to `maskgit/hic_to_npy.py`. However, their **outputs** need proper conversion before feeding to MaskGIT (see above).

## Verification

After re-training, verify the normalization is consistent:

1. Check that data ranges match expectations:
   ```python
   import numpy as np
   hr = np.load("hic_vqgan_train_hr.npy")
   lr = np.load("hic_vqgan_train_lr.npy")
   print(f"HR range: [{hr.min():.3f}, {hr.max():.3f}]")
   print(f"LR range: [{lr.min():.3f}, {lr.max():.3f}]")
   ```

2. Verify HR values are clipped: `hr.max() <= np.log1p(100) ≈ 4.615`
3. Verify LR values account for scaling: LR should generally be higher than HR (due to ×16 scaling before clipping)

## Notes

- **Backup old files**: Consider backing up old `.npy` files and checkpoints before regenerating
- **Training time**: VQGAN and MaskGIT training can take significant time
- **Consistency**: Make sure all models in your pipeline use the same normalization to ensure compatibility
