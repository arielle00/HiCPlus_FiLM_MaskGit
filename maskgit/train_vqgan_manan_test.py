import os, sys
import torch

# Add the current directory to Python path to ensure we import from local folder, not pip package
# This ensures we use the local muse_maskgit_pytorch folder instead of any pip-installed version
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# custom VQGAN (your weighted-loss variant)
from muse_maskgit_pytorch.newVqgan_vae import VQGanVAE
# trainer from the package (ensure its type hint was changed to nn.Module as we discussed)
from muse_maskgit_pytorch.trainers import VQGanVAETrainer

# Verify we're using the local version, not the pip package
import muse_maskgit_pytorch
local_package_path = os.path.join(script_dir, 'muse_maskgit_pytorch')
if hasattr(muse_maskgit_pytorch, '__file__'):
    actual_path = os.path.dirname(muse_maskgit_pytorch.__file__)
    if os.path.abspath(actual_path) != os.path.abspath(local_package_path):
        print(f"[WARNING] Importing from {actual_path} instead of local {local_package_path}")
        print(f"[WARNING] This may cause issues. Consider uninstalling pip package: pip uninstall muse-maskgit-pytorch")
    else:
        print(f"[INFO] Using local muse_maskgit_pytorch from {actual_path}")
else:
    print(f"[INFO] Using local muse_maskgit_pytorch (namespace package)")

# ---------------- config ----------------
HIGHRES_NPY   = '/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train.npy'
RESULTS_DIR   = './2layers_4096_commit0.20_image128'

# Set USE_WANDB = False to disable wandb logging
USE_WANDB = False  # Change to True to enable wandb

WANDB_PROJECT = 'vqgan-hic-training' if USE_WANDB else None
WANDB_RUN     = 'vqgan-highres-3layers-codebook-2048_asinc_No_hicweighting' if USE_WANDB else None

vae = VQGanVAE(
    dim = 256,
    channels = 1,

    # ↓↓↓ change here ↓↓↓
    layers = 2,                   # 512 / 2^4 = 32 → latent 32x32 (1024 tokens)
    codebook_size = 4096,         # match latent positions to force capacity bottleneck
    lookup_free_quantization = False,

    vq_kwargs = dict(
        codebook_dim      = 256,
        commitment_weight = 0.20, # keep same so the only big change is capacity
        decay             = 0.995
    ),

    l2_recon_loss = True,
    use_vgg_and_gan = False,
)

trainer = VQGanVAETrainer(
    vae = vae,
    image_size = 128,
    folder = HIGHRES_NPY,

    batch_size = 4,
    grad_accum_every = 8,

    # if you just want a quick “degrades” demo, you can shorten a bit
    num_train_steps = 20_000,

    results_folder = RESULTS_DIR,

    use_hic_weighted_loss = False,
    hic_weight_alpha = 0.55,

    codebook_log_interval = 200,
    wandb_project = WANDB_PROJECT,
    wandb_run_name = WANDB_RUN
)

trainer.train()
