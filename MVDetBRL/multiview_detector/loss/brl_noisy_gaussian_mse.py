import torch

from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE


class BRLNoisyGaussianMSE(BRLGaussianMSE):
    """
    BRLGaussianMSE with a learned per-pixel noise scale.

    If sigma is None, this behaves exactly like BRLGaussianMSE. This keeps
    per-view losses and test-time comparison aligned with the baseline.

    noise_scope:
      "confuse": use noisy loss only on confusing background pixels.
      "all": use noisy loss on every pixel.

    noise_mode:
      "nll": Gaussian negative log-likelihood.
      "prob": MSE(x + sigma * eps, y), useful as an ablation.
    """

    def __init__(self, pos_thr=0.1, confuse_pred_thr=0.3, beta=0.1, mirror=True,
                 noise_mode="nll", noise_scope="confuse", n_samples=1):
        super().__init__(pos_thr=pos_thr, confuse_pred_thr=confuse_pred_thr, beta=beta, mirror=mirror)
        assert noise_mode in ("nll", "prob")
        assert noise_scope in ("confuse", "all")
        self.noise_mode = noise_mode
        self.noise_scope = noise_scope
        self.n_samples = n_samples

    def forward(self, x, target, kernel, sigma=None):
        if sigma is None:
            return super().forward(x, target, kernel)
        assert sigma.shape == x.shape, (sigma.shape, x.shape)

        soft_gt = self._traget_transform(x, target, kernel)
        pos_mask = soft_gt >= self.pos_thr
        confuse_mask = (~pos_mask) & (x.detach() >= self.confuse_pred_thr)
        noisy_mask = confuse_mask if self.noise_scope == "confuse" else torch.ones_like(pos_mask)
        clean_mask = ~noisy_mask

        loss = x.new_zeros(())
        if clean_mask.any():
            loss = loss + (x[clean_mask] - soft_gt[clean_mask]).pow(2).sum()
        if noisy_mask.any():
            loss = loss + self._noisy_loss(x[noisy_mask], soft_gt[noisy_mask], sigma[noisy_mask])
        return loss / x.numel()

    def _noisy_loss(self, x, y, sigma):
        if self.noise_mode == "nll":
            var = sigma.pow(2)
            return (0.5 * ((x - y).pow(2) / var + var.log())).sum()
        total = x.new_zeros(())
        for _ in range(self.n_samples):
            total = total + (x + sigma * torch.randn_like(x) - y).pow(2).sum()
        return total / self.n_samples
