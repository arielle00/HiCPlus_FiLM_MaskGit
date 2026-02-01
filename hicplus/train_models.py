from __future__ import print_function
import argparse as ap
from math import log10
import torch
from pathlib import Path

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required but not available. Exiting.")

device = torch.device("cuda")

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
from hicplus import utils
#import model
import argparse
from hicplus import trainConvNet
import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

def main(args):
    # Get diag_max_sep_bins with backward compatibility
    diag_max_sep_bins = getattr(args, 'diag_max_sep_bins', None)
    lr_npy = getattr(args, 'lr_npy', None)
    hr_npy = getattr(args, 'hr_npy', None)
    
    # Check if we're using .npy files or legacy .hic file mode
    use_npy_mode = (lr_npy is not None) or (hr_npy is not None)
    
    if use_npy_mode:
        # New mode: Load from .npy files with .coords.npy
        if lr_npy is None or hr_npy is None:
            raise ValueError("Both --lr-npy and --hr-npy must be provided when using .npy file mode.")
        
        print('[INFO] Loading patches from .npy files...')
        print(f'  LR file: {lr_npy}')
        print(f'  HR file: {hr_npy}')
        
        # Load patches
        lowres_sub = np.load(lr_npy)  # Expected [N, 1, H, W] or [N, H, W]
        highres_sub = np.load(hr_npy)  # Expected [N, 1, H, W] or [N, H, W]
        
        # Ensure 4D format [N, 1, H, W]
        if lowres_sub.ndim == 3:
            lowres_sub = lowres_sub[:, None, :, :]
        if highres_sub.ndim == 3:
            highres_sub = highres_sub[:, None, :, :]
        
        if lowres_sub.shape[0] != highres_sub.shape[0]:
            raise ValueError(f"LR and HR must have same number of patches. LR: {lowres_sub.shape[0]}, HR: {highres_sub.shape[0]}")
        
        print(f'  Loaded {lowres_sub.shape[0]} patches')
        print(f'  LR shape: {lowres_sub.shape}')
        print(f'  HR shape: {highres_sub.shape}')
        
        # Load or auto-detect coordinates
        lr_coords_path = str(Path(lr_npy).with_suffix(Path(lr_npy).suffix + ".coords.npy"))
        hr_coords_path = str(Path(hr_npy).with_suffix(Path(hr_npy).suffix + ".coords.npy"))
        
        # Try to load coords (prefer HR coords, fallback to LR coords)
        coords = None
        if Path(hr_coords_path).exists():
            coords = np.load(hr_coords_path)
            print(f'  Loaded HR coords from: {hr_coords_path}')
        elif Path(lr_coords_path).exists():
            coords = np.load(lr_coords_path)
            print(f'  Loaded LR coords from: {lr_coords_path}')
        else:
            print(f'  [WARN] No .coords.npy file found at {hr_coords_path} or {lr_coords_path}')
            print(f'         Diagonal filtering will not work. Please provide .coords.npy files.')
            if diag_max_sep_bins is not None:
                raise ValueError(f"--diag-max-sep-bins was specified but no .coords.npy file found.")
        
        # Filter by diagonal distance if requested
        if diag_max_sep_bins is not None and coords is not None:
            if coords.shape[0] != lowres_sub.shape[0]:
                raise ValueError(f"Coords count ({coords.shape[0]}) != patch count ({lowres_sub.shape[0]})")
            
            # Compute separation: sep = abs(i - j)
            if coords.shape[1] < 2:
                raise ValueError(f"Coords must have at least 2 columns (i, j). Got shape {coords.shape}")
            
            separations = np.abs(coords[:, 0] - coords[:, 1])
            valid_mask = separations <= diag_max_sep_bins
            
            n_before = lowres_sub.shape[0]
            lowres_sub = lowres_sub[valid_mask]
            highres_sub = highres_sub[valid_mask]
            n_after = lowres_sub.shape[0]
            
            print(f'  [INFO] Diagonal filtering: |i-j| <= {diag_max_sep_bins} bins')
            print(f'  [INFO] Filtered {n_before} -> {n_after} patches ({n_after/n_before*100:.1f}% kept)')
        
        print(f'Final HR shape: {highres_sub.shape}')
        print(f'Final LR shape: {lowres_sub.shape}')
    
    else:
        # Legacy mode: Extract from .hic file
        # Handle both single chromosome (backward compatibility) and multiple chromosomes
        if isinstance(args.chromosome, int):
            chromosomes = [args.chromosome]
        else:
            chromosomes = args.chromosome
        
        print(f'[INFO] Training on chromosomes: {chromosomes}')
        
        # Get diagonal filtering parameter
        diag_max_sep_bins = getattr(args, 'diag_max_sep_bins', None)
        
        # Extract and concatenate data from all chromosomes
        highres_list = []
        lowres_list = []
        MAX_PER_CHR = 50000
        
        for chr_num in chromosomes:
            print(f'[INFO] Processing chromosome {chr_num}...')
            highres = utils.train_matrix_extract(chr_num, 10000, args.inputfile)
            
            print(f'  Dividing, filtering chromosome {chr_num}...')
            highres_sub, index_hr = utils.train_divide(highres)
            print(f'  Chromosome {chr_num} HR shape: {highres_sub.shape}')
            
            lowres = utils.genDownsample(highres, 1/float(args.scalerate))
            lowres_sub, index_lr = utils.train_divide(lowres)
            print(f'  Chromosome {chr_num} LR shape: {lowres_sub.shape}')
            
            # Apply diagonal filtering if requested
            if diag_max_sep_bins is not None:
                # Extract coordinates from index: index contains (tag, i, j) tuples
                # Convert to numpy array of (i, j) coordinates
                if len(index_hr) > 0 and len(index_hr[0]) >= 3:
                    coords_hr = np.array([(idx[1], idx[2]) for idx in index_hr], dtype=np.int32)
                else:
                    # Fallback: create dummy coords if index format is unexpected
                    print(f'  [WARN] Unexpected index format, skipping diagonal filtering for chr {chr_num}')
                    coords_hr = None
                
                if coords_hr is not None and coords_hr.shape[0] == highres_sub.shape[0]:
                    # Compute separation: sep = abs(i - j)
                    separations = np.abs(coords_hr[:, 0] - coords_hr[:, 1])
                    valid_mask = separations <= diag_max_sep_bins
                    
                    n_before = highres_sub.shape[0]
                    highres_sub = highres_sub[valid_mask]
                    lowres_sub = lowres_sub[valid_mask]
                    n_after = highres_sub.shape[0]
                    
                    print(f'  [INFO] Diagonal filtering: |i-j| <= {diag_max_sep_bins} bins')
                    print(f'  [INFO] Filtered {n_before} -> {n_after} patches ({n_after/n_before*100:.1f}% kept)')
            
            n = highres_sub.shape[0]
            if n > MAX_PER_CHR:
                idx = np.random.choice(n, MAX_PER_CHR, replace=False)
                highres_sub = highres_sub[idx]
                lowres_sub  = lowres_sub[idx]
                print(f'  Subsampled chromosome {chr_num}: {n} -> {MAX_PER_CHR}')

            highres_list.append(highres_sub)
            lowres_list.append(lowres_sub)
        
        # Concatenate all chromosomes
        print('Concatenating data from all chromosomes...')
        highres_sub = np.concatenate(highres_list, axis=0)
        lowres_sub = np.concatenate(lowres_list, axis=0)
        
        print(f'Final HR shape: {highres_sub.shape}')
        print(f'Final LR shape: {lowres_sub.shape}')

    print('start training...')
    print("----------------------------------",args.outmodel)
    
    # Get loss weighting parameters with backward compatibility
    loss_weight_mode = getattr(args, 'loss_weight_mode', None)
    loss_weight_k = getattr(args, 'loss_weight_k', 10)
    lr_tau_counts = getattr(args, 'lr_tau_counts', 0.0)
    offdiag_band = getattr(args, 'offdiag_band', None)
    lambda_bg = getattr(args, 'lambda_bg', 0.005)
    
    trainConvNet.train(
        lowres_sub,
        highres_sub,
        args.outmodel,
        arch=args.arch,
        width=args.width,
        depth=args.depth,
        use_aux=(not args.no_aux),
        space=args.space,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        amp=(not args.no_amp),
        accum_steps=args.accum_steps,
        log_every=args.log_every,
        patience=args.patience,
        loss_weight_mode=loss_weight_mode,
        loss_weight_k=loss_weight_k,
        lr_tau_counts=lr_tau_counts,
        offdiag_band=offdiag_band,
        lambda_bg=lambda_bg,
    )




    print('finished...')
