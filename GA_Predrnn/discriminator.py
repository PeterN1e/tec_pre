import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  building blocks
# ------------------------------------------------------------------ #

class SpaceToDepth(nn.Module):
    """
    Rearrange spatial pixels into channels with stride r.
    (B, C, H, W) -> (B, C*r*r, ceil(H/r), ceil(W/r))
    Input is padded so H, W are divisible by r.
    """
    def __init__(self, r: int = 2):
        super().__init__()
        self.r = r

    def forward(self, x):
        B, C, H, W = x.shape
        r = self.r
        pad_h = (r - H % r) % r
        pad_w = (r - W % r) % r
        x = F.pad(x, [0, pad_w, 0, pad_h])
        _, _, H_p, W_p = x.shape
        x = x.view(B, C, H_p // r, r, W_p // r, r)
        x = x.permute(0, 1, 3, 5, 2, 4)              # (B, C, r, r, H/r, W/r)
        return x.reshape(B, C * r * r, H_p // r, W_p // r)


class D3Block(nn.Module):
    """Conv + BN + ReLU (no spectral norm, used early in discriminator)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DBlock(nn.Module):
    """Conv (spectral-normed) + BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------------------ #
#  full discriminator
# ------------------------------------------------------------------ #

class Discriminator(nn.Module):
    """
    Patch-style discriminator for TEC map pairs.
    Architecture (matching paper):
        SpaceToDepth(r=2)
        -> D3Block x 2  (no spectral norm, downsample)
        -> DBlock  x 3  (spectral norm, downsample)
        -> DBlock  x 1  (spectral norm, identity stride=1)
        -> AdaptiveMaxPool -> ReLU -> FC -> score

    Input:  (B, 1, H, W)   a single TEC frame
    Output: (B, 1)         realism score
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            SpaceToDepth(r=2),                        # (4, H/2, W/2)
            D3Block(4,       ch,    stride=2),        # (ch,   H/4, W/4)
            D3Block(ch,      ch*2,  stride=2),        # (ch*2, H/8, W/8)
            DBlock(ch*2,     ch*4,  stride=2),        # (ch*4, H/16, W/16)
            DBlock(ch*4,     ch*8,  stride=2),        # (ch*8, H/32, W/32)
            DBlock(ch*8,     ch*8,  stride=2),        # (ch*8, H/64, W/64)
            DBlock(ch*8,     ch*8,  stride=1),        # (ch*8, same)
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc   = nn.Linear(ch * 8, 1)

    def forward(self, x):
        """
        x: (B, 1, H, W)
        returns: (B, 1)
        """
        x = self.net(x)
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(x, inplace=True)
        return self.fc(x)


# ------------------------------------------------------------------ #
#  hinge losses (standalone functions)
# ------------------------------------------------------------------ #

def hinge_loss_d(score_real, score_fake):
    """Discriminator hinge loss: E[relu(1-D(real))] + E[relu(1+D(fake))]"""
    return F.relu(1.0 - score_real).mean() + F.relu(1.0 + score_fake).mean()


def hinge_loss_g(score_fake):
    """Generator hinge loss (used inside predictor loss): -E[D(fake)]"""
    return -score_fake.mean()
