import os, sys
import torch

# ------------------------------------------------
# ensure local muse_maskgit_pytorch is used
# ------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import muse_maskgit_pytorch
print("[INFO] muse_maskgit_pytorch loaded from:", muse_maskgit_pytorch.__file__)

# ------------------------------------------------
# imports
# ------------------------------------------------

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.trainers import VQGanVAETrainer

# ------------------------------------------------
# config
# ------------------------------------------------

HIGHRES_NPY = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy"
RESULTS_DIR = "./vqgan_results_2layers_4096_removedOEsignal"

BATCH_SIZE = 4
GRAD_ACCUM = 8
NUM_STEPS  = 20_000
IMAGE_SIZE = 128

# ------------------------------------------------
# VQGAN (official MaskGit version ONLY)
# ------------------------------------------------

vae = VQGanVAE(
    dim=256,
    channels=1,
    layers=2,
    codebook_size=4096,
    lookup_free_quantization=False,   # <-- big change
    use_vgg_and_gan=False,
    vq_kwargs=dict(
        codebook_dim=64,              # try 64 or 128
        decay=0.95,
        commitment_weight=1.0,
        kmeans_init=True,
        use_cosine_sim=True,
    ),
)

# ------------------------------------------------
# trainer (Hi-C compatible)
# ------------------------------------------------

trainer = VQGanVAETrainer(
    vae=vae,
    folder=HIGHRES_NPY,          # .npy → HiCDataset automatically
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    grad_accum_every=GRAD_ACCUM,
    num_train_steps=NUM_STEPS,
    results_folder=RESULTS_DIR,
)

# ------------------------------------------------
# train
# ------------------------------------------------

trainer.train()
