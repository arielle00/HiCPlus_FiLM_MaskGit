#!/usr/bin/env python3
"""
infer_gated_maskgit_log1p_aligned.py

Key fix vs your current script:
- Align LR / Proposal / HR patches by coords (*.coords.npy) before sampling.
- Better visualization: shared vmin/vmax across LR/Proposal/Refined/Final/HR.
- Enforces VAE + MaskGit on same device as inputs (fixes cuda/cpu mismatch).

Expected:
- proposal_npy, lr_npy, hr_npy are log1p(counts) with shape [N,128,128] or [N,1,128,128]
- coords files exist as:
    <file>.npy.coords.npy  OR  <file>.coords.npy
  and contain an array shaped [N, K] where each row identifies a patch uniquely
  (we treat each row as a tuple key).
"""

import os
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE


# -------------------------
# small utilities
# -------------------------
def ensure_nchw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected [N,H,W] or [N,1,H,W], got {x.shape}")
    return x.astype(np.float32, copy=False)

def load_state_dict_flex(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    return ckpt

def default_coords_path(npy_path: str) -> str:
    p = Path(npy_path)
    cand1 = str(p) + ".coords.npy"              # foo.npy.coords.npy
    cand2 = str(p.with_suffix("")) + ".coords.npy"  # foo.coords.npy
    if Path(cand1).exists():
        return cand1
    if Path(cand2).exists():
        return cand2
    return ""

def load_coords_or_none(npy_path: str, explicit: str = None):
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"coords file not found: {explicit}")
        coords = np.load(explicit, allow_pickle=True)
        return coords, explicit

    auto = default_coords_path(npy_path)
    if auto and Path(auto).exists():
        coords = np.load(auto, allow_pickle=True)
        return coords, auto

    return None, ""

def coords_keys(coords: np.ndarray):
    """
    Turn coords array into hashable keys.
    Works for numeric arrays or object arrays.
    """
    coords = np.asarray(coords)
    if coords.ndim == 1:
        # already something like array of tuples/strings
        return [tuple([c]) if not isinstance(c, (tuple, list)) else tuple(c) for c in coords]

    keys = []
    for i in range(coords.shape[0]):
        row = coords[i]
        # convert to python scalars
        row = [r.item() if hasattr(r, "item") else r for r in row.tolist()]
        keys.append(tuple(row))
    return keys

@torch.no_grad()
def decode_ids_vq_safe(vae: VQGanVAE, ids_hw: torch.Tensor) -> torch.Tensor:
    q = vae.quantizer

    cb = None
    if hasattr(q, "codebook"):
        cb = getattr(q, "codebook")
        if hasattr(cb, "embed"):
            cb = cb.embed
    if cb is None and hasattr(q, "embed"):
        cb = q.embed
    if cb is None and hasattr(q, "embedding"):
        cb = q.embedding
    if cb is None and hasattr(q, "_codebook") and hasattr(q._codebook, "embed"):
        cb = q._codebook.embed
    if cb is None:
        raise AttributeError("Could not locate VQ codebook inside vae.quantizer")

    if isinstance(cb, torch.nn.Embedding):
        codes = cb(ids_hw)          # [B,Ht,Wt,D]
    else:
        codes = cb[ids_hw]          # [B,Ht,Wt,D]

    if hasattr(q, "project_out"):
        fmap = q.project_out(codes)
    else:
        fmap = codes

    fmap = rearrange(fmap, "b h w c -> b c h w")
    return vae.decode(fmap)

# -------------------------
# gating (token-space)
# -------------------------
@torch.no_grad()
def downsample_to_tokens_logspace(x: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    # x: [B,1,H,W] log1p
    B, C, H, W = x.shape
    assert C == 1
    assert H % token_h == 0 and W % token_w == 0
    kH = H // token_h
    kW = W // token_w
    return F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))

@torch.no_grad()
def compute_gate(
    lr_log: torch.Tensor,
    proposal_log: torch.Tensor,
    token_h: int,
    token_w: int,
    gate_mode: str,
    lr_tau_counts: float,
    topk_frac: float,
    topk_only_where_low_evidence: bool,
    max_edit_frac: float,
):
    """
    gate_mode:
      - lr_zero: editable where lr_counts <= tau
      - topk_diff: editable top-k by |proposal - lr| (optionally restricted to low-evidence)
    """
    lr_tok = downsample_to_tokens_logspace(lr_log, token_h, token_w).squeeze(1)       # [B,th,tw]
    p_tok  = downsample_to_tokens_logspace(proposal_log, token_h, token_w).squeeze(1) # [B,th,tw]

    # low-evidence defined in COUNT space
    lr_counts_tok = torch.expm1(lr_tok).clamp(min=0)
    low_ev = (lr_counts_tok <= lr_tau_counts)

    B = lr_tok.shape[0]
    T = token_h * token_w

    if gate_mode == "lr_zero":
        gate = low_ev

    elif gate_mode == "topk_diff":
        diff = (p_tok - lr_tok).abs()  # [B,th,tw]
        if topk_only_where_low_evidence:
            diff = diff.masked_fill(~low_ev, -1e9)

        flat = diff.view(B, -1)
        k = max(int(topk_frac * flat.shape[1]), 1)
        gate_flat = torch.zeros_like(flat, dtype=torch.bool)
        topk_idx = flat.topk(k, dim=-1).indices
        gate_flat.scatter_(1, topk_idx, True)
        gate = gate_flat.view(B, token_h, token_w)

    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    # enforce max_edit_frac cap
    if max_edit_frac is not None and max_edit_frac > 0:
        cap = max(int(max_edit_frac * T), 1)
        gate2 = torch.zeros((B, T), dtype=torch.bool, device=gate.device)
        gflat = gate.view(B, T)
        for b in range(B):
            idxs = torch.nonzero(gflat[b], as_tuple=False).squeeze(-1)
            if idxs.numel() > cap:
                idxs = idxs[:cap]
            if idxs.numel() > 0:
                gate2[b, idxs] = True
        gate = gate2.view(B, token_h, token_w)

    low_ev_frac = low_ev.float().mean().item()
    gate_frac = gate.float().mean().item()
    gate_counts = gate.view(B, -1).sum(dim=-1).tolist()
    print(f"[INFO] low-evidence frac (token space): {low_ev_frac:.4f}")
    print(f"[INFO] gate frac (token space): {gate_frac:.4f}, per-sample gate counts: {gate_counts}")

    return gate

@torch.no_grad()
def upsample_gate_to_image(gate_hw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    gate = gate_hw.float().unsqueeze(1)  # [B,1,th,tw]
    return F.interpolate(gate, size=(H, W), mode="nearest")

# -------------------------
# gated conditional generate
# -------------------------
@torch.no_grad()
def generate_gated_conditional(
    model: MaskGit,
    lr_log: torch.Tensor,
    proposal_log: torch.Tensor,
    gate_hw: torch.Tensor,
    timesteps: int,
    temperature: float,
):
    device = next(model.parameters()).device
    fmap = model.vae.get_encoded_fmap_size(model.image_size)
    seq_len = fmap * fmap
    B = lr_log.shape[0]
    assert gate_hw.shape == (B, fmap, fmap)

    _, ids_lr_grid, _ = model.vae.encode(lr_log.to(device))
    ids_lr = rearrange(ids_lr_grid, "b h w -> b (h w)")

    _, ids_p_grid, _ = model.vae.encode(proposal_log.to(device))
    ids_init = rearrange(ids_p_grid, "b h w -> b (h w)")

    gate_flat = gate_hw.view(B, seq_len)

    ids = ids_init.clone()
    scores = torch.zeros((B, seq_len), device=device, dtype=torch.float)
    scores = scores.masked_fill(~gate_flat, -1e5)
    editable_count = gate_flat.sum(dim=-1).clamp(min=0)

    for t in torch.linspace(0, 1, timesteps, device=device):
        mask_prob = model.noise_schedule(t)
        num_mask = (mask_prob * editable_count.float()).long().clamp(min=1)
        num_mask = torch.minimum(num_mask, editable_count.clamp(min=1))

        masked_idx_list = []
        max_k = 0
        for b in range(B):
            if editable_count[b].item() == 0:
                masked_idx = torch.empty((0,), device=device, dtype=torch.long)
            else:
                k = int(num_mask[b].item())
                masked_idx = scores[b].topk(k, dim=-1).indices
            masked_idx_list.append(masked_idx)
            max_k = max(max_k, masked_idx.numel())

        if max_k == 0:
            break

        idx_pad = torch.zeros((B, max_k), device=device, dtype=torch.long)
        for b in range(B):
            mi = masked_idx_list[b]
            if mi.numel() > 0:
                idx_pad[b, :mi.numel()] = mi

        ids = ids.scatter(1, idx_pad, model.transformer.mask_id)

        logits = model.transformer(ids, cond=ids_lr)
        probs = F.softmax(logits / temperature, dim=-1)

        B2, N, V = probs.shape
        sampled = torch.multinomial(probs.reshape(B2 * N, V), 1).reshape(B2, N)

        is_mask = ids == model.transformer.mask_id
        ids = torch.where(is_mask, sampled, ids)

        conf = probs.gather(2, sampled[..., None]).squeeze(-1)
        new_scores = 1.0 - conf
        scores = torch.where(is_mask, new_scores, scores)
        scores = scores.masked_fill(~gate_flat, -1e5)

    ids_hw = ids.view(B, fmap, fmap)
    refined = decode_ids_vq_safe(model.vae, ids_hw)
    # Clamp to >= 0 to ensure valid log1p space (log1p(x) >= 0 for x >= 0)
    refined = refined.clamp(min=0.0)
    return refined

# -------------------------
# mass calibration
# -------------------------
@torch.no_grad()
def mass_calibrate_to_hr(final_log1p: torch.Tensor, hr_log1p: torch.Tensor, eps: float = 1e-8):
    final_c = torch.expm1(final_log1p).clamp(min=0)
    hr_c = torch.expm1(hr_log1p).clamp(min=0)
    s_final = final_c.sum(dim=(-2, -1), keepdim=True)
    s_hr = hr_c.sum(dim=(-2, -1), keepdim=True)
    alpha = (s_hr + eps) / (s_final + eps)
    final_c2 = final_c * alpha
    return torch.log1p(final_c2), alpha.squeeze(-1).squeeze(-1).squeeze(-1)

# -------------------------
# plotting (no normalization artifacts)
# -------------------------
def save_comparison_figure(out_path: Path, lr, prop, refined, gate_up, hr=None):
    """
    lr/prop/refined/hr: [B,1,H,W] tensors on CPU, already log1p
    gate_up: [B,1,H,W] float 0/1
    Note: refined is already the final output (combines proposal + MaskGit-refined via gating)
    """
    B = lr.shape[0]
    cols = 5 if hr is not None else 4

    # choose a shared color scale across LR/Prop/Refined/(HR)
    stack = [lr, prop, refined]
    if hr is not None:
        stack.append(hr)
    allv = torch.cat([x.reshape(B, -1) for x in stack], dim=1)
    vmin = 0.0
    vmax = float(torch.quantile(allv, 0.995).item() + 1e-8)

    fig_h = max(2.0 * B, 4.0)
    fig, axes = plt.subplots(B, cols, figsize=(3.2 * cols, fig_h), squeeze=False)

    titles = ["LR", "Proposal (HiCPlus+FiLM)", "MaskGit refined (final)", "Gate (upsampled)"]
    if hr is not None:
        titles.append("HR ref")

    for r in range(B):
        panels = [lr[r, 0], prop[r, 0], refined[r, 0], gate_up[r, 0]]
        if hr is not None:
            panels.append(hr[r, 0])

        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)

            if titles[c].startswith("Gate"):
                ax.imshow(panels[c].numpy(), cmap="gray", vmin=0.0, vmax=1.0, origin="lower")
            else:
                ax.imshow(panels[c].numpy(), cmap="Reds", vmin=vmin, vmax=vmax, origin="lower")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

def save_gate_token_image(out_path: Path, gate_hw: torch.Tensor):
    # gate_hw: [B,th,tw] bool
    g = gate_hw.float().cpu().numpy()
    # stack rows vertically
    img = np.concatenate([g[i] for i in range(g.shape[0])], axis=0)
    plt.figure(figsize=(8, 2))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()

# -------------------------
# main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vqgan_ckpt", required=True)
    p.add_argument("--maskgit_ckpt", required=True)

    p.add_argument("--proposal_npy", required=True)
    p.add_argument("--lr_npy", required=True)
    p.add_argument("--hr_npy", default=None)

    p.add_argument("--proposal_coords", default=None)
    p.add_argument("--lr_coords", default=None)
    p.add_argument("--hr_coords", default=None)
    p.add_argument("--align_by_coords", action="store_true",
                   help="Align arrays by coords before sampling (RECOMMENDED: ensures proper LR/proposal/HR matching)")
    p.add_argument("--max_diagonal_dist", type=int, default=512,
                   help="Maximum distance from diagonal for coordinate filtering (in bp). Only coordinates where |x-y| <= this are selected.")
    p.add_argument("--min_ref_sum", type=float, default=50.0,
                   help="Minimum sum of counts in reference (HR) patch for filtering non-empty patches. Patches with sum < this are excluded.")

    p.add_argument("--out_dir", default="cond_samples_gated")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--codebook_size", type=int, default=4096)

    p.add_argument("--timesteps", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.7)

    # gating options requested
    p.add_argument("--gate_mode", choices=["lr_zero", "topk_diff"], default="topk_diff")
    p.add_argument("--lr_tau_counts", type=float, default=0.0)
    p.add_argument("--topk_frac", type=float, default=0.05)
    p.add_argument("--topk_only_where_low_evidence", action="store_true")
    p.add_argument("--max_edit_frac", type=float, default=0.10)

    # merge controls
    p.add_argument("--do_merge", action="store_true")
    p.add_argument("--merge_in_counts", action="store_true")
    p.add_argument("--mass_calibrate", action="store_true")

    p.add_argument("--save_npy", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load arrays
    # -------------------------
    proposal_all = torch.from_numpy(ensure_nchw(np.load(args.proposal_npy))).float()
    lr_all = torch.from_numpy(ensure_nchw(np.load(args.lr_npy))).float()
    hr_all = None
    if args.hr_npy and Path(args.hr_npy).exists():
        hr_all = torch.from_numpy(ensure_nchw(np.load(args.hr_npy))).float()

    Np, Nl = proposal_all.shape[0], lr_all.shape[0]
    if hr_all is not None:
        Nh = hr_all.shape[0]
    else:
        Nh = None

    # -------------------------
    # Align by coords (CRITICAL)
    # -------------------------
    if args.align_by_coords:
        prop_coords, prop_cpath = load_coords_or_none(args.proposal_npy, args.proposal_coords)
        lr_coords, lr_cpath = load_coords_or_none(args.lr_npy, args.lr_coords)
        hr_coords, hr_cpath = (None, "")
        if hr_all is not None:
            hr_coords, hr_cpath = load_coords_or_none(args.hr_npy, args.hr_coords)

        if prop_coords is None or lr_coords is None or (hr_all is not None and hr_coords is None):
            raise RuntimeError(
                "align_by_coords requires coords files for proposal and lr (and hr if provided). "
                "I couldn't find them automatically. Provide --proposal_coords/--lr_coords/--hr_coords."
            )

        prop_keys = coords_keys(prop_coords)
        lr_keys = coords_keys(lr_coords)
        prop_map = {k: i for i, k in enumerate(prop_keys)}
        lr_map = {k: i for i, k in enumerate(lr_keys)}

        if hr_all is not None:
            hr_keys = coords_keys(hr_coords)
            hr_map = {k: i for i, k in enumerate(hr_keys)}
            common = sorted(set(prop_map.keys()) & set(lr_map.keys()) & set(hr_map.keys()))
        else:
            common = sorted(set(prop_map.keys()) & set(lr_map.keys()))

        # Filter to only diagonal/near-diagonal coordinates (|x-y| <= max_diagonal_dist)
        # For Hi-C, valid interactions should be near the diagonal
        max_diagonal_dist = args.max_diagonal_dist
        diagonal_common = []
        for coord in common:
            if isinstance(coord, (tuple, list, np.ndarray)) and len(coord) >= 2:
                x, y = float(coord[0]), float(coord[1])
                dist = abs(x - y)
                if dist <= max_diagonal_dist:
                    diagonal_common.append(coord)
            else:
                # If coord format is unknown, include it (better than excluding everything)
                diagonal_common.append(coord)
        
        print(f"[INFO] Filtered to {len(diagonal_common)}/{len(common)} coordinates near diagonal (max_dist={max_diagonal_dist})")
        
        # Filter to non-empty patches (ref_sum >= threshold)
        # Use HR as reference if available, otherwise use proposal
        min_ref_sum = args.min_ref_sum
        non_empty_common = []
        
        if hr_all is not None:
            # Use HR as reference for filtering
            ref_data = hr_all
            ref_map = hr_map
            ref_name = "HR"
        else:
            # Use proposal as reference
            ref_data = proposal_all
            ref_map = prop_map
            ref_name = "proposal"
        
        print(f"[INFO] Filtering non-empty patches using {ref_name} (min_sum={min_ref_sum})...")
        for coord in diagonal_common:
            if coord not in ref_map:
                continue
            idx_ref = ref_map[coord]
            patch = ref_data[idx_ref]  # [1, H, W] or [H, W]
            
            # Convert log1p to counts and sum
            # Handle both numpy and torch
            if isinstance(patch, torch.Tensor):
                patch_np = patch.cpu().numpy()
            else:
                patch_np = np.asarray(patch)
            
            # Ensure it's 2D or 3D
            if patch_np.ndim == 3:
                patch_np = patch_np[0]  # Take first channel if [1, H, W]
            elif patch_np.ndim == 4:
                patch_np = patch_np[0, 0]  # Take first sample, first channel
            
            # Convert log1p to counts and sum
            patch_counts = np.expm1(patch_np).clip(min=0)
            patch_sum = float(patch_counts.sum())
            
            if patch_sum >= min_ref_sum:
                non_empty_common.append(coord)
        
        if len(non_empty_common) < args.batch_size:
            print(f"[WARN] Only {len(non_empty_common)} non-empty diagonal coords (out of {len(diagonal_common)} diagonal); need batch_size={args.batch_size}")
            if len(non_empty_common) == 0:
                raise RuntimeError(f"No non-empty coordinates found (min_sum={min_ref_sum}). "
                                 f"Try decreasing --min_ref_sum or check data.")
            # Use what we have
            args.batch_size = min(args.batch_size, len(non_empty_common))
        
        print(f"[INFO] Filtered to {len(non_empty_common)}/{len(diagonal_common)} non-empty diagonal coordinates")

        # Select first N patches (deterministic, not random) from filtered list
        chosen = non_empty_common[:args.batch_size]

        idx_prop = torch.tensor([prop_map[k] for k in chosen], dtype=torch.long)
        idx_lr = torch.tensor([lr_map[k] for k in chosen], dtype=torch.long)
        if hr_all is not None:
            idx_hr = torch.tensor([hr_map[k] for k in chosen], dtype=torch.long)

        # record chosen coords
        with open(out_dir / "selected_coords.txt", "w") as f:
            f.write(f"proposal_coords: {prop_cpath}\n")
            f.write(f"lr_coords: {lr_cpath}\n")
            if hr_all is not None:
                f.write(f"hr_coords: {hr_cpath}\n")
            f.write("chosen_coords:\n")
            for k in chosen:
                f.write(str(k) + "\n")

        proposal = proposal_all[idx_prop]
        lr = lr_all[idx_lr]
        hr = hr_all[idx_hr] if hr_all is not None else None

    else:
        # fallback: require coordinate alignment for proper matching
        print("[WARN] --align_by_coords not set, but coordinate alignment is required for proper LR/proposal/HR matching!")
        print("[WARN] Falling back to index-based alignment (assumes identical ordering).")
        print("[WARN] This may cause misalignment if arrays were generated separately.")
        N = min(Np, Nl) if hr_all is None else min(Np, Nl, Nh)
        
        # Still apply diagonal and non-empty filtering even without coords
        # Filter by checking if patches are near diagonal (approximate)
        # and non-empty
        valid_indices = []
        for i in range(N):
            # Check if patch is non-empty (use HR if available, else proposal)
            if hr_all is not None:
                patch = hr_all[i]
            else:
                patch = proposal_all[i]
            
            # Convert to numpy for processing
            if isinstance(patch, torch.Tensor):
                patch_np = patch.cpu().numpy()
            else:
                patch_np = np.asarray(patch)
            
            # Ensure it's 2D
            if patch_np.ndim == 3:
                patch_np = patch_np[0]  # Take first channel if [1, H, W]
            elif patch_np.ndim == 4:
                patch_np = patch_np[0, 0]  # Take first sample, first channel
            
            # Convert log1p to counts and check sum
            patch_counts = np.expm1(patch_np).clip(min=0)
            patch_sum = float(patch_counts.sum())
            
            if patch_sum >= args.min_ref_sum:
                valid_indices.append(i)
        
        if len(valid_indices) < args.batch_size:
            print(f"[WARN] Only {len(valid_indices)} non-empty patches; need batch_size={args.batch_size}")
            if len(valid_indices) == 0:
                raise RuntimeError(f"No non-empty patches found (min_sum={args.min_ref_sum}). "
                                 f"Try decreasing --min_ref_sum.")
            args.batch_size = min(args.batch_size, len(valid_indices))
        
        # Select deterministically (not random)
        idx = torch.tensor(valid_indices[:args.batch_size], dtype=torch.long)
        proposal = proposal_all[idx]
        lr = lr_all[idx]
        hr = hr_all[idx] if hr_all is not None else None

    # move to device
    proposal = proposal.to(device)
    lr = lr.to(device)
    hr = hr.to(device) if hr is not None else None

    # -------------------------
    # Load VQGAN (same device!)
    # -------------------------
    vae = VQGanVAE(
        dim=256,
        channels=1,
        layers=2,
        codebook_size=args.codebook_size,
        lookup_free_quantization=False,
        use_vgg_and_gan=False,
        vq_kwargs=dict(
            codebook_dim=64,
            decay=0.95,
            commitment_weight=1.0,
            kmeans_init=True,
            use_cosine_sim=True,
        ),
    ).to(device)

    vae.load_state_dict(load_state_dict_flex(args.vqgan_ckpt, device), strict=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # infer token grid
    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        _, ids0, _ = vae.encode(dummy)
        token_h, token_w = ids0.shape[1:]
        seq_len = token_h * token_w
    print(f"[INFO] Token grid: {token_h} x {token_w} => seq_len={seq_len}")

    # -------------------------
    # Build MaskGit (same device!)
    # -------------------------
    transformer = Transformer(
        num_tokens=vae.codebook_size,
        seq_len=seq_len,
        dim=512,
        depth=8,
        heads=8,
        dim_head=64,
        self_cond=False,
        cond_tokens=True,
    ).to(device)

    maskgit = MaskGit(
        image_size=args.image_size,
        transformer=transformer,
        vae=vae,
        no_mask_token_prob=0.1,
    ).to(device)

    maskgit.load_state_dict(torch.load(args.maskgit_ckpt, map_location=device), strict=False)
    maskgit.eval()

    # -------------------------
    # gate + refine
    # -------------------------
    gate_hw = compute_gate(
        lr_log=lr,
        proposal_log=proposal,
        token_h=token_h,
        token_w=token_w,
        gate_mode=args.gate_mode,
        lr_tau_counts=args.lr_tau_counts,
        topk_frac=args.topk_frac,
        topk_only_where_low_evidence=args.topk_only_where_low_evidence,
        max_edit_frac=args.max_edit_frac,
    )

    refined = generate_gated_conditional(
        model=maskgit,
        lr_log=lr,
        proposal_log=proposal,
        gate_hw=gate_hw,
        timesteps=args.timesteps,
        temperature=args.temperature,
    )

    # -------------------------
    # Note: refined is already the final output
    # MaskGit selectively refines tokens where gate=True, keeping proposal tokens where gate=False
    # The additional merge step (--do_merge) is redundant and not recommended
    # -------------------------
    gate_up = upsample_gate_to_image(gate_hw, args.image_size, args.image_size)

    # Ensure outputs are in valid log1p space (>= 0)
    refined = refined.clamp(min=0.0)
    
    # Handle optional merge (not recommended - refined is already final)
    if args.do_merge:
        print("[WARN] --do_merge is enabled, but refined output already combines proposal + MaskGit via gating.")
        print("[WARN] Additional merge may be redundant. Consider using refined directly as final output.")
        if not args.merge_in_counts:
            # log-space merge (not recommended)
            final = (1.0 - gate_up) * proposal + gate_up * refined
        else:
            prop_c = torch.expm1(proposal).clamp(min=0)
            ref_c  = torch.expm1(refined).clamp(min=0)
            final_c = (1.0 - gate_up) * prop_c + gate_up * ref_c
            final = torch.log1p(final_c)
        final = final.clamp(min=0.0)
        
        if args.mass_calibrate:
            if hr is None:
                raise RuntimeError("--mass_calibrate requires --hr_npy")
            final, alpha = mass_calibrate_to_hr(final, hr)
            alpha_list = alpha.detach().cpu().numpy().tolist()
            print(f"[INFO] mass_calibrate alpha (per-sample): {alpha_list}")
    else:
        # refined is already the final output (no additional merge needed)
        final = refined
        if args.mass_calibrate:
            if hr is None:
                raise RuntimeError("--mass_calibrate requires --hr_npy")
            final, alpha = mass_calibrate_to_hr(final, hr)
            alpha_list = alpha.detach().cpu().numpy().tolist()
            print(f"[INFO] mass_calibrate alpha (per-sample): {alpha_list}")

    # -------------------------
    # save visuals
    # -------------------------
    save_gate_token_image(out_dir / "gate_tokens.png", gate_hw)

    # upsampled gate preview
    gate_up_cpu = gate_up.detach().cpu()
    plt.figure(figsize=(10, 2))
    img = torch.cat([gate_up_cpu[i, 0] for i in range(gate_up_cpu.shape[0])], dim=1).numpy()
    plt.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_dir / "gate_upsampled.png"), dpi=200)
    plt.close()

    # comparison (refined is already the final output)
    lr_cpu = lr.detach().cpu()
    prop_cpu = proposal.detach().cpu()
    refined_cpu = refined.detach().cpu()  # This is already the final output
    hr_cpu = hr.detach().cpu() if hr is not None else None

    save_comparison_figure(
        out_dir / "gated_comparison.png",
        lr_cpu, prop_cpu, refined_cpu,
        gate_up_cpu, hr=hr_cpu
    )

    print(f"[OK] Saved: {out_dir / 'gated_comparison.png'}")
    print(f"[OK] Saved: {out_dir / 'gate_tokens.png'}")
    print(f"[OK] Saved: {out_dir / 'gate_upsampled.png'}")

    # -------------------------
    # save arrays (optional)
    # -------------------------
    if args.save_npy:
        np.save(out_dir / "proposal_log1p.npy", prop_cpu.numpy())
        np.save(out_dir / "lr_log1p.npy", lr_cpu.numpy())
        np.save(out_dir / "refined_log1p.npy", refined_cpu.numpy())
        np.save(out_dir / "final_log1p.npy", final_cpu.numpy())
        np.save(out_dir / "gate_hw.npy", gate_hw.detach().cpu().numpy())
        np.save(out_dir / "gate_up.npy", gate_up_cpu.numpy())
        if hr_cpu is not None:
            np.save(out_dir / "hr_log1p.npy", hr_cpu.numpy())
        print(f"[OK] Saved .npy outputs to: {out_dir}")

if __name__ == "__main__":
    main()
