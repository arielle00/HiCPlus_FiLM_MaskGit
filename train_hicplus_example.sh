#!/bin/bash
# Example script to train HiCPlus with 4DNFI7H4II2V.hic

# Single chromosome example:
hicplus train \
    -i /home/012002744/hicplus_thesis/4DNFI7H4II2V.hic \
    -r 10 \
    --arch hicplus \
    -c 21 \
    -o test_hicplus_4dnfi7h4ii2v_chr21.model \
    --optimizer adamw \
    --lr 0.002 \
    --space log1p \
    --no-amp

# Multiple chromosomes example (now supported!):
# hicplus train \
#     -i /home/012002744/hicplus_thesis/4DNFI7H4II2V.hic \
#     -r 10 \
#     --arch hicplus \
#     -c 1 2 3 19 21 \
#     -o test_hicplus_4dnfi7h4ii2v_multi.model \
#     --optimizer adamw \
#     --lr 0.002 \
#     --space log1p \
#     --no-amp

# Parameters explained:
# -i, --inputfile: Path to your .hic file (required)
# -r, --scalerate: Downsampling rate for low-res training data
#                  - Use 10 to match maskgit (frac=0.10, 10x downsampling)
#                  - Use 16 for 200-300M reads data
#                  - Use 40 for lower depth data
# --arch: Model architecture - "hicplus" (default) or "film"
# -c, --chromosome: Chromosome number(s) to train on (default: 21)
#                   - Single: -c 21
#                   - Multiple: -c 1 2 3 19 21 (space-separated)
#                   - Data from all specified chromosomes will be combined
# -o, --outmodel: Output model checkpoint path
# --optimizer: Optimizer - "adamw" (default) or "sgd"
# --lr: Learning rate (default: 2e-3, example uses 0.002)
# --space: Training space - "log1p" (default, recommended) or "counts"
# --no-amp: Disable mixed precision training (use if you encounter issues)

