import torch.nn as nn
import torch

class FilmFusion(nn.Module):
    def __init__(self, aux_dim, channel,out_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.ReLU(),
            nn.Linear(32, channel * 2)  # out: gamma + beta
        )
        self.channel = channel
        self.out_dim = out_dim
    def forward(self, feat_map, aux):
        # feat_map: (B, T, C, H, W), aux: (B, T, D)
        B, T, C, H, W = feat_map.shape
        out = self.mlp(aux)                     # (B, T, 2*C)
        gamma, beta = torch.split(out, C, dim=-1)  # 各自 (B, T, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, T, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        modulated = gamma * feat_map + beta           # 调制后的特征图，形状不变
        if self.out_dim == 3:
            return modulated
        elif self.out_dim == 1:
            return modulated.flatten(2)
        else:
            raise ValueError("Invalid out_dim")

if __name__ == '__main__':
     aux = torch.randn(24,36,6)
     tec = torch.randn(24,36,12,18,19)
     test = FilmFusion(6,12)
     b = test(tec,aux)
     print(b.size())
