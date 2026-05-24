import torch.nn as nn

from config import ModelConfig
import torch

cfg_model = ModelConfig()
class ConvGRUCell(nn.Module):
    """ConvGRU单元 - 处理2D空间序列"""

    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        # 合并卷积：输入+隐藏 -> 3*hidden（重置门+更新门+候选）
        self.conv = nn.Conv2d(in_ch + hid_ch, 3 * hid_ch, kernel, padding=kernel // 2)

    def forward(self, x, h=None):
        # x: (B, C, H, W)
        B, _, H, W = x.shape
        h = h if h is not None else torch.zeros(B, self.conv.out_channels // 3, H, W, device=x.device)

        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)  # (B, in+hid, H, W)
        conv_out = self.conv(combined)  # (B, 3*hid, H, W)

        # 拆分三部分
        r, z, n = torch.split(conv_out, h.size(1), dim=1)

        # GRU公式
        r = torch.sigmoid(r)  # 重置门
        z = torch.sigmoid(z)  # 更新门
        n = torch.tanh(n)  # 候选状态

        h_new = (1 - z) * h + z * n
        return h_new


class ConvGRU(nn.Module):
    """
    简洁的ConvGRU预测模型
    输入: (B, T_in, C, H, W)  ->  (B, T_out, C, H, W)
    结构图解:
    ┌─────────────────────────────────────────────────────────┐
    │ 输入形状: (B, 10, 3, 18, 19)                              │
    ├─────────────────────────────────────────────────────────┤
    │ 1. 输入投影: 1x1卷积调整通道数                             │
    │    (B*10, 3, 18, 19) -> (B*10, 64, 18, 19)                │
    ├─────────────────────────────────────────────────────────┤
    │ 2. ConvGRU编码器: 处理时间序列                            │
    │    ┌─────┐ ┌─────┐ ... ┌─────┐                          │
    │    │GRU  │ │GRU  │     │GRU  │                          │
    │    │Cell │ │Cell │     │Cell │  (处理10个时间步)         │
    │    └─────┘ └─────┘     └─────┘                          │
    ├─────────────────────────────────────────────────────────┤
    │ 3. ConvGRU解码器: 多步预测                               │
    │    ┌─────┐ ┌─────┐ ... ┌─────┐                          │
    │    │GRU  │ │GRU  │     │GRU  │                          │
    │    │Cell │ │Cell │     │Cell │  (生成5个预测步)          │
    │    └─────┘ └─────┘     └─────┘                          │
    ├─────────────────────────────────────────────────────────┤
    │ 4. 输出投影: 1x1卷积恢复通道数                            │
    │    (B*5, 64, 18, 19) -> (B*5, 3, 18, 19)                 │
    ├─────────────────────────────────────────────────────────┤
    │ 输出形状: (B, 5, 3, 18, 19)                              │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, in_channels=12, hidden_channels=64, history_len=12, predict_len=1, gru_layers=2):
        super().__init__()
        self.history_len = history_len
        self.predict_len = predict_len
        # 输入输出调整
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, 1)
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, 1)
        # GRU层
        self.encoder = nn.ModuleList([
            ConvGRUCell(hidden_channels, hidden_channels) for _ in range(gru_layers)
        ])
        self.decoder = nn.ModuleList([
            ConvGRUCell(hidden_channels, hidden_channels) for _ in range(gru_layers)
        ])
        # 预测头
        self.predict = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        )
    def forward(self, x, teacher=None, force_ratio=0.5):
        """
        Args:
            x: (B, T_in, C, H, W)  输入序列
            teacher: (B, T_out, C, H, W) 真实值（用于teacher forcing）
            force_ratio: teacher forcing概率
        """
        B, T_in, C, H, W = x.shape

        # 1. 投影输入
        x = x.reshape(B * T_in, C, H, W)
        x = self.in_proj(x)  # (B*T_in, hid, H, W)
        x = x.reshape(B, T_in, -1, H, W)

        # 2. 编码器 - 处理历史序列
        h_states = [[None] * len(self.encoder) for _ in range(T_in)]

        for t in range(T_in):
            for l, gru in enumerate(self.encoder):
                h_prev = h_states[t - 1][l] if t > 0 else None
                h_curr = gru(x[:, t] if l == 0 else h_states[t][l - 1], h_prev)
                h_states[t][l] = h_curr

        # 3. 解码器 - 生成预测
        preds = []
        h_dec = h_states[-1]  # 初始状态 = 编码器最后时刻

        for t in range(self.predict_len):
            # 自回归输入
            if t == 0 or torch.rand(1) > force_ratio or teacher is None:
                dec_input = h_dec[-1]  # 用上一步的预测
            else:
                # Teacher forcing: 用真实值投影
                teacher_t = teacher[:, t - 1].reshape(B * C, H, W)
                dec_input = self.in_proj(teacher_t).reshape(B, -1, H, W)

            # 解码器GRU
            for l, gru in enumerate(self.decoder):
                h_prev = h_dec[l]
                h_curr = gru(dec_input if l == 0 else h_dec[l - 1], h_prev)
                h_dec[l] = h_curr

            # 生成预测
            pred = self.predict(h_dec[-1])  # (B, hid, H, W)
            pred = self.out_proj(pred)  # (B, C, H, W)
            preds.append(pred.unsqueeze(1))

        # 拼接所有预测
        return torch.cat(preds, dim=1)

if __name__ == '__main__':
    test = ConvGRU(12,12,history_len = 1)
    a = torch.randn(24,12,12,18,19)
    b= test(a)
    print(b.shape)
