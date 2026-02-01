#!/usr/bin/env python3
"""
Improved patch sampling for VQGAN/MaskGit training.

Features:
1. Filter patches with |i-j| <= diag_max_sep_bins (diagonal only)
2. Exclude main diagonal band from loss (ignore offsets < min_diag_offset bins)
3. Add random jitter in patch position (data augmentation)

Usage:
    from improved_patch_sampling import FilteredHiCDataset, DiagonalMaskedLoss
    
    # Create filtered dataset
    dataset = FilteredHiCDataset(
        npy_path="hic_vqgan_train_hr.npy",
        coords_path="hic_vqgan_train_hr.npy.coords.npy",
        diag_max_sep_bins=512,
        enable_jitter=True,
        jitter_max=8
    )
    
    # Use masked loss in training
    loss_fn = DiagonalMaskedLoss(min_diag_offset=2)
    loss = loss_fn(pred, target)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class FilteredHiCDataset(Dataset):
    """
    Hi-C dataset with diagonal filtering and optional position jitter.
    
    Filters patches to only include those where |i-j| <= diag_max_sep_bins.
    Optionally adds random jitter to patch positions for data augmentation.
    """
    
    def __init__(
        self,
        npy_path: str,
        coords_path: Optional[str] = None,
        diag_max_sep_bins: Optional[int] = None,
        enable_jitter: bool = False,
        jitter_max: int = 8,
        seed: Optional[int] = None,
    ):
        """
        Args:
            npy_path: Path to .npy file with patches [N, 1, H, W]
            coords_path: Path to .coords.npy file with coordinates [N, 2] (i_bin, j_bin)
                        If None, tries to auto-detect as npy_path + ".coords.npy"
            diag_max_sep_bins: Maximum |i-j| distance to include (None = no filtering)
            enable_jitter: If True, add random jitter to patch positions during training
            jitter_max: Maximum jitter in bins (default: 8)
            seed: Random seed for jitter (None = no seed)
        """
        self.data = np.load(npy_path)
        if self.data.ndim != 4 or self.data.shape[1] != 1:
            raise ValueError(f"Expected [N,1,H,W], got {self.data.shape}")
        
        # Load or auto-detect coordinates
        if coords_path is None:
            coords_path = str(Path(npy_path).with_suffix(Path(npy_path).suffix + ".coords.npy"))
        
        if Path(coords_path).exists():
            self.coords = np.load(coords_path)
            if self.coords.shape[0] != self.data.shape[0]:
                raise ValueError(
                    f"Coords count ({self.coords.shape[0]}) != data count ({self.data.shape[0]})"
                )
        else:
            print(f"[WARN] No coordinates file found at {coords_path}")
            print(f"       Creating dummy coordinates. Filtering/jitter will not work correctly.")
            # Create dummy coords (sequential)
            n = self.data.shape[0]
            self.coords = np.stack([
                np.arange(n, dtype=np.int32),
                np.arange(n, dtype=np.int32)
            ], axis=1)
        
        # Filter by diagonal distance
        self.valid_indices = None
        if diag_max_sep_bins is not None:
            distances = np.abs(self.coords[:, 0] - self.coords[:, 1])
            self.valid_indices = np.where(distances <= diag_max_sep_bins)[0]
            print(f"[INFO] Filtered {len(self.valid_indices)}/{len(self.coords)} patches "
                  f"with |i-j| <= {diag_max_sep_bins}")
        else:
            self.valid_indices = np.arange(len(self.data))
            print(f"[INFO] No diagonal filtering (using all {len(self.data)} patches)")
        
        self.enable_jitter = enable_jitter
        self.jitter_max = jitter_max
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        
        if enable_jitter:
            print(f"[INFO] Position jitter enabled (max={jitter_max} bins)")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns a patch, optionally with random jitter applied.
        """
        actual_idx = self.valid_indices[idx]
        patch = self.data[actual_idx].copy()  # [1, H, W]
        
        # Apply jitter if enabled
        if self.enable_jitter and self.jitter_max > 0:
            patch = self._apply_jitter(patch)
        
        return torch.from_numpy(patch).float()
    
    def _apply_jitter(self, patch: np.ndarray) -> np.ndarray:
        """
        Apply random jitter by shifting the patch.
        Jitter is applied in both i and j directions.
        """
        H, W = patch.shape[1], patch.shape[2]
        
        # Random jitter in [-jitter_max, jitter_max]
        di = self.rng.randint(-self.jitter_max, self.jitter_max + 1)
        dj = self.rng.randint(-self.jitter_max, self.jitter_max + 1)
        
        if di == 0 and dj == 0:
            return patch
        
        # Create shifted patch
        jittered = np.zeros_like(patch)
        
        # Source and destination indices
        if di >= 0:
            src_i_start, src_i_end = 0, H - di
            dst_i_start, dst_i_end = di, H
        else:
            src_i_start, src_i_end = -di, H
            dst_i_start, dst_i_end = 0, H + di
        
        if dj >= 0:
            src_j_start, src_j_end = 0, W - dj
            dst_j_start, dst_j_end = dj, W
        else:
            src_j_start, src_j_end = -dj, W
            dst_j_start, dst_j_end = 0, W + dj
        
        # Copy valid region
        if src_i_end > src_i_start and src_j_end > src_j_start:
            jittered[:, dst_i_start:dst_i_end, dst_j_start:dst_j_end] = \
                patch[:, src_i_start:src_i_end, src_j_start:src_j_end]
        
        return jittered


class DiagonalMaskedLoss:
    """
    Loss function that masks out the main diagonal band to prevent overfitting
    to diagonal ridge artifacts.
    
    For Hi-C data, the main diagonal (offset=0) and near-diagonal (offset < min_diag_offset)
    are very strong signals that can dominate training. This loss function ignores
    those regions, forcing the model to learn off-diagonal interactions.
    """
    
    def __init__(
        self,
        min_diag_offset: int = 2,
        loss_fn: Optional[torch.nn.Module] = None,
    ):
        """
        Args:
            min_diag_offset: Minimum diagonal offset to include in loss.
                            Offsets < min_diag_offset are masked out.
                            Recommended: 2-4 bins
            loss_fn: Base loss function (default: L1Loss)
        """
        self.min_diag_offset = min_diag_offset
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.L1Loss(reduction='none')
        
        # Pre-compute mask (will be created on first use with correct size)
        self._mask_cache = {}
    
    def _get_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create diagonal mask for given dimensions."""
        key = (H, W)
        if key not in self._mask_cache:
            # Create mask: 1 where we want to compute loss, 0 where we mask out
            mask = torch.ones(H, W, dtype=torch.float32)
            
            # Mask out main diagonal band
            for i in range(H):
                for j in range(W):
                    offset = abs(i - j)
                    if offset < self.min_diag_offset:
                        mask[i, j] = 0.0
            
            self._mask_cache[key] = mask.to(device)
        
        return self._mask_cache[key]
    
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked loss.
        
        Args:
            pred: Predicted tensor [B, C, H, W] or [B, 1, H, W]
            target: Target tensor [B, C, H, W] or [B, 1, H, W]
        
        Returns:
            Scalar loss value
        """
        # Compute per-pixel loss
        per_pixel_loss = self.loss_fn(pred, target)  # [B, C, H, W]
        
        # Get mask (assume square patches, use H dimension)
        B, C, H, W = per_pixel_loss.shape
        mask = self._get_mask(H, W, per_pixel_loss.device)  # [H, W]
        
        # Expand mask to match batch and channel dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply mask
        masked_loss = per_pixel_loss * mask
        
        # Compute mean over non-masked pixels
        num_valid = mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            # Fallback if all pixels are masked
            return per_pixel_loss.mean()


def create_filtered_dataset(
    npy_path: str,
    coords_path: Optional[str] = None,
    diag_max_sep_bins: Optional[int] = 512,
    enable_jitter: bool = True,
    jitter_max: int = 8,
) -> FilteredHiCDataset:
    """
    Convenience function to create a filtered Hi-C dataset.
    
    Args:
        npy_path: Path to patches .npy file
        coords_path: Path to coordinates .npy file (auto-detected if None)
        diag_max_sep_bins: Maximum |i-j| distance (default: 512)
        enable_jitter: Enable position jitter (default: True)
        jitter_max: Maximum jitter in bins (default: 8)
    
    Returns:
        FilteredHiCDataset instance
    """
    return FilteredHiCDataset(
        npy_path=npy_path,
        coords_path=coords_path,
        diag_max_sep_bins=diag_max_sep_bins,
        enable_jitter=enable_jitter,
        jitter_max=jitter_max,
    )


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test improved patch sampling"
    )
    parser.add_argument("--npy", type=str, required=True, help="Path to patches .npy")
    parser.add_argument("--coords", type=str, default=None, help="Path to coords .npy")
    parser.add_argument("--diag-max-sep", type=int, default=512, help="Max |i-j| distance")
    parser.add_argument("--jitter", action="store_true", help="Enable jitter")
    parser.add_argument("--jitter-max", type=int, default=8, help="Max jitter in bins")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = FilteredHiCDataset(
        npy_path=args.npy,
        coords_path=args.coords,
        diag_max_sep_bins=args.diag_max_sep,
        enable_jitter=args.jitter,
        jitter_max=args.jitter_max,
    )
    
    print(f"\n[INFO] Dataset size: {len(dataset)}")
    print(f"[INFO] Sample shape: {dataset[0].shape}")
    
    # Test a few samples
    print("\n[INFO] Testing samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: shape={sample.shape}, min={sample.min():.3f}, max={sample.max():.3f}")
    
    # Test masked loss
    print("\n[INFO] Testing diagonal masked loss...")
    loss_fn = DiagonalMaskedLoss(min_diag_offset=2)
    
    # Create dummy pred/target
    B, C, H, W = 2, 1, 128, 128
    pred = torch.randn(B, C, H, W)
    target = torch.randn(B, C, H, W)
    
    loss = loss_fn(pred, target)
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Masked out diagonal offsets < 2 bins")
