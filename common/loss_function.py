import torch
import torch.nn.functional as F
import torch.nn as nn



def vae_loss(recon_x, x, mu, logvar):
    # 重构损失（BCE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)


class DeltaCriterion:
    """Supervise delta-supervised models without changing model code.

    When called as criterion(tec_in, aux_in, tec_gt), this returns a
    combined loss over delta prediction plus optional weak reconstruction.
    """

    def __init__(self, model, delta_weight=1.0, recon_weight=0.1):
        self.model = model
        self.l1 = nn.L1Loss()
        self.delta_weight = delta_weight
        self.recon_weight = recon_weight
        self._needs_context = True

    def __call__(self, tec_in, aux_in, tec_gt):
        delta_pred = self.model.forward_delta(tec_in, aux_in)
        delta_true = tec_gt - tec_in[:, -12:, :, :]
        delta_loss = self.l1(delta_pred, delta_true)

        recon_pred = tec_in[:, -12:, :, :] + delta_pred
        recon_loss = self.l1(recon_pred, tec_gt)

        return self.delta_weight * delta_loss + self.recon_weight * recon_loss
