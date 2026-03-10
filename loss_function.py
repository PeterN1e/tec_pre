import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    # 重构损失（BCE）
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)