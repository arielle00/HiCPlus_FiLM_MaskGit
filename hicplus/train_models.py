from __future__ import print_function
import argparse as ap
from math import log10
import torch

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
    # Handle both single chromosome (backward compatibility) and multiple chromosomes
    if isinstance(args.chromosome, int):
        chromosomes = [args.chromosome]
    else:
        chromosomes = args.chromosome
    
    print(f'[INFO] Training on chromosomes: {chromosomes}')
    
    # Extract and concatenate data from all chromosomes
    highres_list = []
    lowres_list = []
    MAX_PER_CHR = 50000
    
    for chr_num in chromosomes:
        print(f'[INFO] Processing chromosome {chr_num}...')
        highres = utils.train_matrix_extract(chr_num, 10000, args.inputfile)
        
        print(f'  Dividing, filtering chromosome {chr_num}...')
        highres_sub, index = utils.train_divide(highres)
        print(f'  Chromosome {chr_num} HR shape: {highres_sub.shape}')
        
        lowres = utils.genDownsample(highres, 1/float(args.scalerate))
        lowres_sub, index = utils.train_divide(lowres)
        print(f'  Chromosome {chr_num} LR shape: {lowres_sub.shape}')
        
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
    )




    print('finished...')
