import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)


class FourierLoss(nn.Module):
    """L1 distance between 2D FFT amplitude spectra."""

    def forward(self, pred, target):
        device = pred.device
        pred_f = torch.fft.fft2(pred.cpu().float(), norm="ortho").abs()
        target_f = torch.fft.fft2(target.cpu().float(), norm="ortho").abs()
        return F.l1_loss(pred_f, target_f).to(device)


class SSIMLoss(nn.Module):
    """1 - SSIM using a separable Gaussian window."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _gaussian_window(self, device):
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        return g.outer(g).unsqueeze(0).unsqueeze(0)

    def forward(self, pred, target):
        B, T, H, W = pred.shape
        pred = pred.reshape(B * T, 1, H, W)
        target = target.reshape(B * T, 1, H, W)
        pad = self.window_size // 2
        window = self._gaussian_window(pred.device)

        mu_p = F.conv2d(pred, window, padding=pad)
        mu_t = F.conv2d(target, window, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_pp = F.conv2d(pred * pred, window, padding=pad) - mu_pp
        sigma_tt = F.conv2d(target * target, window, padding=pad) - mu_tt
        sigma_pt = F.conv2d(pred * target, window, padding=pad) - mu_pt

        ssim = ((2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)) / \
               ((mu_pp + mu_tt + self.C1) * (sigma_pp + sigma_tt + self.C2))
        return 1 - ssim.mean()


class DeltaCriterion:
    """Combined loss for delta-prediction models.

    L = delta_weight   * L1(Δ_pred, Δ_true)
      + recon_weight   * L1(baseline + Δ_pred, target)
      + fourier_weight  * L1(|FFT(pred)|, |FFT(target)|)
      + ssim_weight    * (1 - SSIM(pred, target))
      + temporal_weight * L1(Δframe_pred, Δframe_target)
    """

    def __init__(
        self,
        model,
        delta_weight=1.0,
        recon_weight=0.3,
        fourier_weight=0.2,
        ssim_weight=0.1,
        temporal_weight=0.05,
    ):
        self.model = model
        self.l1 = nn.L1Loss()
        self.fourier_loss = FourierLoss()
        self.ssim_loss = SSIMLoss()
        self.delta_weight = delta_weight
        self.recon_weight = recon_weight
        self.fourier_weight = fourier_weight
        self.ssim_weight = ssim_weight
        self.temporal_weight = temporal_weight
        self._needs_context = True

    def __call__(self, tec_in, aux_in, tec_gt):
        delta_pred = self.model.forward_delta(tec_in, aux_in)
        output_length = tec_gt.shape[1]
        input_baseline = tec_in[:, -output_length:, :, :]
        delta_true = tec_gt - input_baseline

        recon_pred = input_baseline + delta_pred

        delta_loss = self.l1(delta_pred, delta_true)
        recon_loss = self.l1(recon_pred, tec_gt)
        fourier_loss = self.fourier_loss(recon_pred, tec_gt)
        ssim_loss = self.ssim_loss(recon_pred, tec_gt)

        if output_length > 1:
            temporal_loss = self.l1(
                recon_pred[:, 1:] - recon_pred[:, :-1],
                tec_gt[:, 1:] - tec_gt[:, :-1],
            )
        else:
            temporal_loss = torch.tensor(0.0, device=tec_in.device)

        return (
            self.delta_weight * delta_loss
            + self.recon_weight * recon_loss
            + self.fourier_weight * fourier_loss
            + self.ssim_weight * ssim_loss
            + self.temporal_weight * temporal_loss
        )
