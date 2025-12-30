import torch

@torch.no_grad()
class MaskGitConditioner:
    def __init__(self, maskgit):
        """
        maskgit: pretrained MaskGIT model (already loaded)
        """
        self.maskgit = maskgit.eval()

    def __call__(self, x_lr):
        """
        x_lr: [B,1,H,W] in SAME normalized space as MaskGIT training
        returns:
            M: [B,1,f,f] ∈ [0,1]
        """
        # encode to latent tokens
        _, ids, _ = self.maskgit.vae.encode(x_lr)
        B, f, _ = ids.shape
        ids = ids.view(B, f * f)

        cond_ids = None
        if self.maskgit.resize_image_for_cond_image:
            _, cond_ids, _ = self.maskgit.cond_vae.encode(x_lr)
            cond_ids = cond_ids.view(B, -1)

        logits = self.maskgit.transformer.forward_with_cond_scale(
            ids,
            conditioning_token_ids=cond_ids,
            cond_scale=1.0
        )

        probs = logits.softmax(-1)
        conf  = probs.max(dim=-1).values   # [B,S]
        unc   = 1.0 - conf

        return unc.view(B, 1, f, f).clamp_(0.0, 1.0)
