# Training with Improved Patch Sampling - Step-by-Step Guide

This guide explains how to use the improved patch sampling features for VQGAN and MaskGit training.

## Overview

The improved patch sampling includes:
1. **Diagonal filtering**: Only train on patches with `|i-j| <= diag_max_sep_bins` (default: 512)
2. **Position jitter**: Random shift augmentation (default: ±8 bins)
3. **Diagonal-masked loss**: Exclude main diagonal band from loss (offsets < 2-4 bins)

## Prerequisites

1. **Coordinate files**: You need `.coords.npy` files for your training data
   - `hic_vqgan_train_hr.npy.coords.npy`
   - `hic_vqgan_train_lr.npy.coords.npy` (for MaskGit training)

2. **If coordinates don't exist**, generate them:
   ```bash
   cd /home/012002744/hicplus_thesis/maskgit
   python3 create_coords_for_combined.py \
     --npy-file hic_vqgan_train_hr.npy \
     --chr-dir chr_data \
     --kind hr
   ```

## Step-by-Step Training Order

### Step 1: Test the Improved Sampling (Optional)

First, verify the improved sampling works with your data:

```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 improved_patch_sampling.py \
  --npy hic_vqgan_train_hr.npy \
  --coords hic_vqgan_train_hr.npy.coords.npy \
  --diag-max-sep 512 \
  --jitter \
  --jitter-max 8
```

This will:
- Load your data
- Filter patches by diagonal distance
- Test jitter functionality
- Test the masked loss function

### Step 2: Train VQGAN with Improved Sampling

Train the VQGAN encoder/decoder with filtered patches:

```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_vqgan_improved.py
```

**What this does:**
- Creates a filtered dataset (only patches with `|i-j| <= 512`)
- Applies position jitter during training
- Saves filtered patches to `hic_vqgan_train_hr_filtered.npy`
- Trains VQGAN on the filtered data
- Saves results to `./vqgan_results_improved_sampling/`

**Expected output:**
- Filtered dataset size (should be smaller than original)
- Training progress with loss values
- Checkpoints saved every 1000 steps
- Best model saved as `vae.best.pt`

### Step 3: Train Conditional MaskGit with Improved Sampling

Train MaskGit conditioned on LR images:

```bash
cd /home/012002744/hicplus_thesis/maskgit

python3 train_maskgit_cond_improved.py
```

**What this does:**
- Loads the trained VQGAN from Step 2
- Creates filtered paired HR+LR dataset
- Applies diagonal filtering and jitter to both HR and LR
- Trains MaskGit transformer
- Saves best model as `maskgit_cond_improved.best.pt`

**Expected output:**
- Filtered dataset statistics
- Training progress with train/val loss
- Early stopping when validation loss stops improving
- Best model checkpoint

### Step 4: Inference with Improved Models

Use the trained models for inference:

```bash
cd /home/012002744/hicplus_thesis/maskgit/muse_vqgan

python3 infer_gated_maskgit_log1p.py \
  --vqgan_ckpt /home/012002744/hicplus_thesis/maskgit/vqgan_results_improved_sampling/vae.best.pt \
  --maskgit_ckpt /home/012002744/hicplus_thesis/maskgit/maskgit_cond_improved.best.pt \
  --proposal_npy /home/012002744/hicplus_thesis/pred_chr20_patches.npy \
  --lr_npy /home/012002744/hicplus_thesis/low_res_chr20_patches.npy \
  --hr_npy /home/012002744/hicplus_thesis/ref_chr20_patches.npy \
  --out_dir cond_samples_gated_chr20_improved \
  --gate_mode topk_diff \
  --topk_frac 0.2 \
  --timesteps 12 \
  --temperature 1.0 \
  --do_merge \
  --merge_in_counts \
  --save_npy \
  --max_diagonal_dist 512 \
  --min_ref_sum 100
```

## Configuration Parameters

You can adjust these parameters in the training scripts:

### Diagonal Filtering
- `DIAG_MAX_SEP_BINS = 512`: Maximum `|i-j|` distance to include
  - Smaller = more restrictive (only very close interactions)
  - Larger = less restrictive (includes more off-diagonal)

### Position Jitter
- `ENABLE_JITTER = True`: Enable/disable jitter
- `JITTER_MAX = 8`: Maximum random shift in bins
  - Helps with data augmentation
  - Can improve generalization

### Loss Masking
- `MIN_DIAG_OFFSET = 2`: Ignore diagonal offsets < 2 bins in loss
  - Prevents overfitting to diagonal ridge artifacts
  - Forces model to learn off-diagonal interactions
  - Recommended: 2-4 bins

## File Structure

After training, you should have:

```
maskgit/
├── improved_patch_sampling.py          # Core module
├── train_vqgan_improved.py             # VQGAN training script
├── train_maskgit_cond_improved.py      # MaskGit training script
├── vqgan_results_improved_sampling/    # VQGAN outputs
│   ├── vae.best.pt                     # Best VQGAN model
│   └── ...
├── maskgit_cond_improved.best.pt       # Best MaskGit model
└── hic_vqgan_train_hr_filtered.npy    # Filtered training data
```

## Troubleshooting

### "No coordinates file found"
- Generate coordinates using `create_coords_for_combined.py`
- Or set `coords_path=None` (filtering won't work, but training will proceed)

### "HR and LR datasets have different lengths"
- The script will use the minimum length
- Ensure HR and LR are aligned (same number of patches)

### "Filtered dataset is too small"
- Increase `DIAG_MAX_SEP_BINS` (e.g., 1024 instead of 512)
- Check that coordinates are correct

### Training loss is NaN
- Reduce learning rate
- Check data normalization
- Verify data doesn't contain NaN/Inf values

## Comparison: Standard vs Improved Sampling

| Feature | Standard Training | Improved Sampling |
|---------|------------------|-------------------|
| Patch filtering | All patches | Only `|i-j| <= 512` |
| Data augmentation | None | Position jitter |
| Loss masking | Full image | Exclude main diagonal |
| Training focus | All interactions | Off-diagonal emphasis |

## Next Steps

After training with improved sampling:
1. Compare results with standard training
2. Adjust parameters based on validation performance
3. Use for inference on new chromosomes
4. Analyze which patches benefit most from filtering
