from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial

from beartype import beartype
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

#from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
# Import the HiC-weighted VAE from the root directory
# if TYPE_CHECKING:
#     import sys
#     import os
#     sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#     from vqgan_vae_hic_weighted import VQGanVAE

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from ema_pytorch import EMA

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import wandb
from scipy.stats import pearsonr

def find_and_pop(items, condition, default):
    """Find and remove an item from list matching condition, or return default."""
    for i, item in enumerate(items):
        if condition(item):
            return items.pop(i)
    return default()

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def noop(*args, **kwargs):
    pass

def find_index(arr, cond):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return None

def find_and_pop(arr, cond, default = None):
    ind = find_index(arr, cond)

    if exists(ind):
        return arr.pop(ind)

    if callable(default):
        return default()

    return default

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# image related helpers fnuctions and dataset

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

class HiCDataset(Dataset):
    def __init__(self, npy_path):
        super().__init__()
        self.data = np.load(npy_path)
        if self.data.ndim == 3:
            self.data = self.data[:, None, :, :]  # ensure shape [N, 1, 256, 256]
        print(f"Loaded Hi-C dataset with shape: {self.data.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

@beartype
class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        grad_accum_every = 1,
        max_grad_norm = None,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict(),
        use_hic_weighted_loss = False,  # New argument
        hic_weight_alpha = 1.0,         # New argument
        wandb_project = None,           # New argument
        wandb_run_name = None,          # New argument
        codebook_log_interval = 200     # New argument
    ):
        super().__init__()

        # instantiate accelerator

        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        self.accelerator = Accelerator(**accelerate_kwargs)

        # vae

        self.vae = vae

        # training params

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(vae.parameters())
        if vae.discr is None:
            discr_parameters = None
            vae_parameters = all_parameters
            self.discr_optim = None
        else:
            discr_parameters = set(vae.discr.parameters())
            vae_parameters = all_parameters - discr_parameters
            self.discr_optim = Adam(discr_parameters, lr = lr)

        self.vae_parameters = vae_parameters
        self.optim = Adam(vae_parameters, lr = lr)

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        #self.ds = ImageDataset(folder, image_size)
        self.ds = HiCDataset(folder)  # where folder is the path to your .npy file

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # prepare with accelerator

        (
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        )

        self.use_ema = use_ema

        if use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)
            self.ema_vae = self.accelerator.prepare(self.ema_vae)

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        self.use_hic_weighted_loss = use_hic_weighted_loss
        self.hic_weight_alpha = hic_weight_alpha
        self.codebook_log_interval = codebook_log_interval
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self._wandb_initialized = False
        self._wandb = None
        self.best_pearson_corr = -float('inf')
        self.best_pearson_corr_ema = -float('inf')
        self.best_pearson_corr_step = 0
        self.best_pearson_corr_ema_step = 0

    @staticmethod
    def compute_pearson(x, y):
        # x, y: [B, C, H, W] torch tensors
        # Compute mean Pearson correlation over batch
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        pearson_corrs = []
        for i, (xb, yb) in enumerate(zip(x, y)):
            xb = xb.flatten()
            yb = yb.flatten()
            
            # Check for constant arrays
            if np.std(xb) == 0 or np.std(yb) == 0:
                print(f"[Warning] Constant array detected in batch {i}: x_std={np.std(xb):.6f}, y_std={np.std(yb):.6f}")
                print(f"[Warning] x_range=[{xb.min():.6f}, {xb.max():.6f}], y_range=[{yb.min():.6f}, {yb.max():.6f}]")
                pearson_corrs.append(0.0)  # Assign 0 correlation for constant arrays
                continue
                
            try:
                pearson_corr = pearsonr(xb, yb)[0]
                pearson_corrs.append(pearson_corr if pearson_corr is not None else 0.0)
            except Exception as e:
                print(f"[Warning] Pearson correlation computation failed for batch {i}: {e}")
                pearson_corrs.append(0.0)
        
        mean_pearson = float(np.mean(pearson_corrs))
        print(f"[Pearson Debug] Batch Pearson correlations: {pearson_corrs}, Mean: {mean_pearson:.4f}")
        return mean_pearson


    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        vae = self.accelerator.unwrap_model(self.vae)
        vae.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def _init_wandb(self):
        if not self._wandb_initialized and self.wandb_project is not None:
            try:
                wandb.init(project=self.wandb_project, name=self.wandb_run_name)
                self._wandb = wandb
                self._wandb_initialized = True
            except Exception as e:
                print(f"[Warning] wandb could not be initialized: {e}")
                self._wandb = None
                self._wandb_initialized = True

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        logs = {}

        self._init_wandb()
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        self.vae.train()
        discr = self.vae.module.discr if self.is_distributed else self.vae.discr
        if self.use_ema:
            ema_vae = self.ema_vae.module if self.is_distributed else self.ema_vae

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl_iter)
            img = img.to(device)

            with self.accelerator.autocast():
                loss = self.vae(
                    img,
                    add_gradient_penalty = apply_grad_penalty,
                    return_loss = True,
                    use_hic_weighted_loss = self.use_hic_weighted_loss,
                    hic_weight_alpha = self.hic_weight_alpha
                )

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(discr):
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

            # log

            self.print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        # update exponential moving averaged generator

        if self.use_ema:
            ema_vae.update()

        # sample results every so often

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)
                valid_data = valid_data.to(device)

                recons = model(valid_data, return_recons = True)

                # else save a grid of images

                imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

                # --- Compute and log Pearson correlation ---
                pearson_corr = self.compute_pearson(recons, valid_data)
                logs['pearson_corr'] = pearson_corr
                
                # Add debugging information
                print(f"[Debug] Step {steps}: Recons range=[{recons.min():.6f}, {recons.max():.6f}], Valid range=[{valid_data.min():.6f}, {valid_data.max():.6f}]")
                print(f"[Debug] Step {steps}: Recons std={recons.std():.6f}, Valid std={valid_data.std():.6f}")
                
                if self._wandb is not None:
                    self._wandb.log({f"valid/pearson_corr_{filename}": pearson_corr, "step": steps})
                # Track best Pearson correlation and save best model
                if filename == str(steps):
                    # Regular model
                    if pearson_corr > self.best_pearson_corr:
                        prev_best = self.best_pearson_corr
                        prev_best_step = self.best_pearson_corr_step
                        self.best_pearson_corr = pearson_corr
                        self.best_pearson_corr_step = steps
                        # Save best model
                        state_dict = self.accelerator.unwrap_model(model).state_dict()
                        model_path = str(self.results_folder / f'vae.best_pearson_corr.pt')
                        self.accelerator.save(state_dict, model_path)
                        if prev_best > -float('inf'):
                            self.print(f"[Step {steps}] ⭐ NEW BEST MODEL ⭐ - Pearson correlation: {pearson_corr:.4f} (Previous best: {prev_best:.4f} at step {prev_best_step})")
                        else:
                            self.print(f"[Step {steps}] ⭐ NEW BEST MODEL ⭐ - Pearson correlation: {pearson_corr:.4f} (First evaluation)")
                    else:
                        self.print(f"[Step {steps}] Pearson correlation: {pearson_corr:.4f} (Best: {self.best_pearson_corr:.4f} at step {self.best_pearson_corr_step})")
                elif filename.endswith('.ema'):
                    # EMA model
                    if pearson_corr > self.best_pearson_corr_ema:
                        prev_best_ema = self.best_pearson_corr_ema
                        prev_best_ema_step = self.best_pearson_corr_ema_step
                        self.best_pearson_corr_ema = pearson_corr
                        self.best_pearson_corr_ema_step = steps
                        state_dict = self.accelerator.unwrap_model(model).state_dict()
                        model_path = str(self.results_folder / f'vae.best_pearson_corr.ema.pt')
                        self.accelerator.save(state_dict, model_path)
                        if prev_best_ema > -float('inf'):
                            self.print(f"[Step {steps}] ⭐ NEW BEST EMA MODEL ⭐ - Pearson correlation: {pearson_corr:.4f} (Previous best: {prev_best_ema:.4f} at step {prev_best_ema_step})")
                        else:
                            self.print(f"[Step {steps}] ⭐ NEW BEST EMA MODEL ⭐ - Pearson correlation: {pearson_corr:.4f} (First evaluation)")
                    else:
                        self.print(f"[Step {steps}] EMA Pearson correlation: {pearson_corr:.4f} (Best: {self.best_pearson_corr_ema:.4f} at step {self.best_pearson_corr_ema_step})")

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # --- wandb logging ---
        if self._wandb is not None:
            self._wandb.log({"train/loss": logs.get('loss', 0.0), "step": steps})
        # --- codebook usage logging ---
        if steps % self.codebook_log_interval == 0:
            try:
                self.vae.eval()
                with torch.no_grad():
                    img = next(self.dl_iter).to(device)
                    _, indices, _ = self.vae.encode(img)
                    flat_indices = indices.cpu().numpy().flatten()
                    hist, bins = np.histogram(flat_indices, bins=self.vae.codebook_size, range=(0, self.vae.codebook_size))
                    usage = (hist > 0).sum()
                    entropy = -np.sum((hist / hist.sum() + 1e-8) * np.log(hist / hist.sum() + 1e-8))
                    
                    # Always print to console
                    self.print(f"[Step {steps}] Codebook usage: {usage}/{self.vae.codebook_size} unique tokens ({100.0 * usage / self.vae.codebook_size:.2f}%), Entropy: {entropy:.2f}")
                    
                    # Log to wandb if enabled
                    if self._wandb is not None:
                        self._wandb.log({
                            "codebook/unique_codes": int(usage),
                            "codebook/entropy": float(entropy),
                            "codebook/histogram": self._wandb.Histogram(flat_indices),
                            "step": steps
                        })
            except Exception as e:
                self.print(f"[Warning] Codebook logging failed: {e}")
        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device
        self._init_wandb()
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)
        if self._wandb is not None:
            self._wandb.finish()
        self.print('training complete')
        self.print('=' * 80)
        self.print('TRAINING SUMMARY - Best Results:')
        self.print('=' * 80)
        if self.best_pearson_corr > -float('inf'):
            self.print(f'Best Regular Model: Step {self.best_pearson_corr_step} with Pearson correlation: {self.best_pearson_corr:.4f}')
        else:
            self.print('No evaluations performed for regular model')
        if self.use_ema:
            if self.best_pearson_corr_ema > -float('inf'):
                self.print(f'Best EMA Model: Step {self.best_pearson_corr_ema_step} with Pearson correlation: {self.best_pearson_corr_ema:.4f}')
            else:
                self.print('No evaluations performed for EMA model')
        self.print('=' * 80)