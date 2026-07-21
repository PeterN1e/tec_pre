from config import TrainConfig
cfg_train = TrainConfig()
import torch
import torch.nn as nn

# 同样导入或定义 CoordGate
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


class CnnDecoder(nn.Module):
    def __init__(self, transmit_parameter_de=3):
        super().__init__()
        self.tec_decoder = nn.Sequential(
            # 第一层反卷积：上采样 ×2
            nn.ConvTranspose2d(
                in_channels=transmit_parameter_de*4,
                out_channels=transmit_parameter_de*2,
                padding=1, kernel_size=3, stride=2, output_padding=(1,0)
            ),
            nn.ReLU(inplace=True),
            CoordGate(transmit_parameter_de * 2),  # 注入坐标门控

            # 第二层反卷积：上采样 ×2
            nn.ConvTranspose2d(
                in_channels=transmit_parameter_de*2,
                out_channels=transmit_parameter_de,
                kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
            CoordGate(transmit_parameter_de),  # 注入坐标门控

            # 输出层：1×1卷积映射为单通道TEC
            nn.Conv2d(in_channels=transmit_parameter_de, out_channels=1, kernel_size=1, stride=1, padding=0),
        )
        self.transmit_parameter_de = transmit_parameter_de

    def forward(self, x):
        x_pred = None
        if x.dim() == 4:  # (B, channels, h, w)
            x = self.tec_decoder(x)  # (B, 1, h, w)
            x = x.squeeze(1)
        elif x.dim() == 5:
            for i in range(x.size(1)):  # (B, pred, channels, h, w)
                x_cell = x[:, i, :]
                x_cell = self.tec_decoder(x_cell)  # 每个时间步单独过CoordGate
                if x_pred is not None:
                    x_pred = torch.cat([x_pred, x_cell], dim=1)
                else:
                    x_pred = x_cell
            x = x_pred.squeeze(1)
        else:
            print("解码器输入维度错误")
        return x


if __name__ == '__main__':
    test_a = CnnDecoder(transmit_parameter_de=3)
    test = torch.randn(cfg_train.batch_size, 36, 12, 18, 19)
    a = test_a(test)
    print(a.size())  # 输出尺寸不变，仍为 (batch_size, 36, 71, 73)