import torch.nn as nn
import torch
class CoordGate(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y_coord = torch.linspace(-1, 1, steps=H, device=x.device, dtype=x.dtype).view(1,1,H,1).expand(B,1,H,W)
        x_coord = torch.linspace(-1, 1, steps=W, device=x.device, dtype=x.dtype).view(1,1,1,W).expand(B,1,H,W)
        coord_feat = torch.cat([x, y_coord, x_coord], dim=1)
        return x * self.gate_conv(coord_feat)


class CnnEncoder(nn.Module):
    """
    输入为（batch, seq_length, 71, 73）
    """
    def __init__(self, transmit_parameter):
        super().__init__()
        self.transmit_parameter = transmit_parameter
        # 每个卷积块后加入 CoordGate
        self.Tec_encoder = nn.Sequential(
            # 第一层：1×1卷积升维
            nn.Conv2d(in_channels=1, out_channels=transmit_parameter, padding=0, kernel_size=1, stride=1),
            nn.ReLU(),
            CoordGate(transmit_parameter),  # 注入坐标门控

            # 第二层：下采样 ×2
            nn.Conv2d(in_channels=transmit_parameter, out_channels=transmit_parameter*2, padding=1, kernel_size=3, stride=2),
            nn.ReLU(),
            CoordGate(transmit_parameter * 2),  # 注入坐标门控

            # 第三层：下采样 ×2
            nn.Conv2d(transmit_parameter * 2, transmit_parameter * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            CoordGate(transmit_parameter * 4),  # 注入坐标门控
        )
        self.fc_tec = nn.Linear(4 * 18 * 19 * transmit_parameter, 512)

    def forward(self, tec):
        batch_size = tec.size(0)
        seq_length = tec.size(1)
        tec = tec.reshape(batch_size * seq_length, 1, 71, 73)
        tec = self.Tec_encoder(tec)
        tec = tec.reshape(batch_size, seq_length, 4 * self.transmit_parameter, 18, 19)
        return tec