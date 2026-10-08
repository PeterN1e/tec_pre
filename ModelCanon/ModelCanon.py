import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ModelCanon(nn.Module):
    """差分主导的 TEC 预测模型。

    输入 36 帧 TEC（前 3 天）+ 物理指数（f10.7/ssn/ae/ap/kp/dst）+ 周期 hour，
    预测第 3 天到第 4 天同一时刻的差值场 Δtec（12 帧），重构 day4 = day3 + Δtec。

    所有派生张量（差值 1/2、hour 的 sin/cos、指数空间广播）都在模型内部构造，
    数据管线只需提供 (tec_in, aux_in)，保证训练/推理框架通用。

    两条信息流：
      - 主流：tec + 指数 + hour，36 帧，过 temporal_encoder。
      - 差分流：diff1 = day2−day1、diff2 = day3−day2，12 帧（每 hour-of-day 一帧），
        过独立 diff_encoder。
    decoder 的 12 个查询对 [主流 ; 差分流] 拼接后的 memory 做 cross-attention，
    head 输出残差，再叠加可学习带符号的拼接式两日基线得到 Δtec。
    """

    def __init__(
        self,
        input_length,
        output_length,
        aux_dim,
        height,
        width,
        d_model,
        n_heads,
        e_layers,
        decoder_layers,
        d_ff,
        dropout,
        patch_size,
        diff_encoder_layers=2,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.height = height
        self.width = width
        self.d_model = d_model
        self.patch_size = patch_size
        self.aux_dim = aux_dim

        self.pad_h = (-height) % patch_size
        self.pad_w = (-width) % patch_size
        self.grid_h = (height + self.pad_h) // patch_size
        self.grid_w = (width + self.pad_w) // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # 主通道：tec(1) + 物理指数(aux_dim) + hour sin/cos(2)
        self.tec_in_channels = 1 + aux_dim + 2
        self.patch_embed_tec = nn.Conv2d(
            self.tec_in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        # 差分通道：diff1、diff2
        self.patch_embed_diff = nn.Conv2d(
            2, d_model, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, d_model) * 0.02)
        # 模态标签：0 = tec 流，1 = 差分流
        self.stream_embed = nn.Parameter(torch.randn(2, 1, 1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # hour 周期编码（固定，非学习）：假设每帧间隔 2h、每天 output_length 帧。
        t = torch.arange(input_length, dtype=torch.float32)
        hour = 2.0 * torch.remainder(t, float(output_length))
        ang = 2.0 * math.pi * hour / 24.0
        self.register_buffer("hour_sc", torch.stack([torch.sin(ang), torch.cos(ang)], dim=1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        diff_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.diff_encoder = nn.TransformerEncoder(diff_layer, num_layers=diff_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.pred_queries = nn.Parameter(torch.randn(output_length, d_model) * (d_model ** -0.5))

        self.head = nn.Linear(d_model, height * width)
        # 带符号、可学习的差分外推系数。相邻日差分是均值回归的（实测
        # slope(diff3~diff2) ≈ -0.4），所以初值取负。
        self.trend_alpha = nn.Parameter(torch.tensor(-0.4))

    def _split_baseline(self, diff1, diff2):
        """统一基线（12 帧差分场），全部用最近一日差分 diff2 = day3 − day2。

        实测 diff1（前二日差分）预测力接近 0，保留它反而引入噪声。
        """
        return self.trend_alpha * diff2

    def _tokenize(self, x, conv, stream_id):
        """(B, T, C, H, W) -> (B, T, num_patches, d_model)。"""
        B, T, C, _, _ = x.shape
        x = F.pad(x, (0, self.pad_w, 0, self.pad_h))
        hp = self.grid_h * self.patch_size
        wp = self.grid_w * self.patch_size
        x = x.reshape(B * T, C, hp, wp)
        x = conv(x)                                        # (B*T, d_model, grid_h, grid_w)
        x = x.reshape(B, T, self.d_model, self.num_patches)
        x = x.transpose(2, 3)                              # (B, T, num_patches, d_model)
        x = x + self.pos_embed + self.stream_embed[stream_id]
        return self.dropout(x)

    def _encode(self, x, encoder):
        """对每个 patch 独立做时序注意力：(B, T, N, D) -> (B, T, N, D)。"""
        B, T, N, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = encoder(x)
        return x.reshape(B, N, T, D).permute(0, 2, 1, 3)

    def _build_streams(self, tec, aux):
        B, T, H, W = tec.shape
        L = self.output_length

        idx = aux.unsqueeze(-1).unsqueeze(-1).expand(B, T, self.aux_dim, H, W)
        hour = self.hour_sc.view(1, T, 2, 1, 1).expand(B, T, 2, H, W)
        x_tec = torch.cat([tec.unsqueeze(2), idx, hour], dim=2)   # (B, T, 1+aux_dim+2, H, W)

        day1 = tec[:, -3 * L:-2 * L]
        day2 = tec[:, -2 * L:-L]
        day3 = tec[:, -L:]
        diff1 = day2 - day1
        diff2 = day3 - day2
        x_diff = torch.stack([diff1, diff2], dim=2)               # (B, L, 2, H, W)
        return x_tec, x_diff, diff1, diff2

    def forward_delta(self, tec, aux):
        B = tec.shape[0]
        x_tec, x_diff, diff1, diff2 = self._build_streams(tec, aux)

        tok_tec = self._tokenize(x_tec, self.patch_embed_tec, 0)
        tok_diff = self._tokenize(x_diff, self.patch_embed_diff, 1)

        tok_tec = self._encode(tok_tec, self.temporal_encoder)
        tok_diff = self._encode(tok_diff, self.diff_encoder)

        memory = torch.cat(
            [
                tok_tec.reshape(B, self.input_length * self.num_patches, self.d_model),
                tok_diff.reshape(B, self.output_length * self.num_patches, self.d_model),
            ],
            dim=1,
        )
        queries = self.pred_queries.unsqueeze(0).expand(B, -1, -1)
        dec = self.decoder(queries, memory)
        residual = self.head(dec).view(B, self.output_length, self.height, self.width)
        return residual + self._split_baseline(diff1, diff2)

    def forward(self, tec, aux):
        """Reconstruct day 4 as day 3 same-hour baseline + predicted delta."""
        delta = self.forward_delta(tec, aux)
        return tec[:, -self.output_length:, :, :] + delta


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length, output_length, aux_dim, height, width = 36, 12, 6, 71, 73

    model = ModelCanon(
        input_length=input_length,
        output_length=output_length,
        aux_dim=aux_dim,
        height=height,
        width=width,
        d_model=256,
        n_heads=8,
        e_layers=4,
        decoder_layers=2,
        d_ff=1024,
        dropout=0.05,
        patch_size=4,
    ).to(device)
    tec = torch.randn(2, input_length, height, width, device=device)
    aux = torch.randn(2, input_length, aux_dim, device=device)

    pred = model(tec, aux)
    delta = model.forward_delta(tec, aux)
    print("forward       ->", tuple(pred.shape))
    print("forward_delta ->", tuple(delta.shape))
    print("params        ->", f"{sum(p.numel() for p in model.parameters()):,}")

    loss = pred.mean()
    loss.backward()
    print("backward      -> OK")
