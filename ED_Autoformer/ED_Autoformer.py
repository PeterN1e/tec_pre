"""
ED_Autoformer 复现实现 (Zhou et al., 2025, Space Weather)
========================================================
论文原结构: U-Net 风格 CNN 编码器/解码器(空间) + Autoformer(时间)
  - 系列分解 (Series Decomposition): 把序列拆成季节分量 + 趋势分量
  - 自相关机制 (Auto-Correlation): FFT 发现周期依赖, 按时延聚合子序列
  - TEC 编码器: (B, T_in, 71, 73) 零填充为 73x73, 下采样到 8x8 空间特征
  - 时空融合:   8x8 展平 = 64 维, 与辅助指数拼接成 (B, T_in, 64+aux_dim)
  - Autoformer: 编解码器输出 (B, out_len, 64) 的 TEC 特征序列
  - TEC 解码器: 转置卷积上采样 + 跳跃连接, 还原为 (B, out_len, 71, 73)

本实现严格对齐仓库框架的模型接口:
    forward(tec, aux) -> (B, output_length, H, W)
其中 tec: (B, input_length, H, W), aux: (B, input_length, aux_dim)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EDAutoformerConfig, TrainConfig, DatasetConfig

cfg_model = EDAutoformerConfig()
cfg_train = TrainConfig()
cfg_dataset = DatasetConfig()


# ============================================================
# 1. 系列分解 (Series Decomposition)
# ============================================================
class MovingAvgBlock1D(nn.Module):
    """滑动平均块: 提取趋势分量, 保持序列长度不变"""

    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def mavg(self, x):
        # x: (B, L, D), 两端用首尾值复制补齐, 保证输出长度仍为 L
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def forward(self, x):
        return self.mavg(x)


class SeriesDecompBlock1D(MovingAvgBlock1D):
    """系列分解: 趋势 = 滑动平均, 季节 = 原序列 - 趋势"""

    def forward(self, x):
        moving_mean = self.mavg(x)
        res = x - moving_mean
        return res, moving_mean


# ============================================================
# 2. 自相关机制 (Auto-Correlation)
# ============================================================
class AutoCorrelation(nn.Module):
    """
    自相关机制, 替代标准 self-attention:
      1) 通过 FFT 计算序列与其滞后版本的互相关, 发现周期依赖
      2) 选择 top-k 时延, Softmax 归一化得到置信权重
      3) 按所选时延滚动 Value 并加权聚合 (time delay aggregation)
    """

    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def time_delay_agg(self, values, corr):
        """
        values: (B, H, D, L)  (H: 头数, D: 每头维度, L: 序列长度)
        corr:   (B, H, D, L)  各时延的自相关分数
        """
        B, H, D, L = values.shape
        top_k = int(self.factor * math.log(L))

        # 跨头/跨维求平均, 得到每个 batch 的时延得分曲线
        mean_corr = torch.mean(torch.mean(corr, dim=1), dim=1)          # (B, L)
        # 再跨 batch 平均, 选出全局 top-k 时延
        _, delay_idx = torch.topk(torch.mean(mean_corr, dim=0), top_k)  # (k,)
        delay_idx = delay_idx.tolist()

        # 对应时延处的置信权重并归一化
        weights = torch.stack([mean_corr[:, d] for d in delay_idx], dim=-1)  # (B, k)
        weights = torch.softmax(weights, dim=-1)

        # 时延聚合: 将 values 滚动 -delay 步后按权重累加
        delays_agg = torch.zeros_like(values)
        for i in range(top_k):
            pattern = torch.roll(values, shifts=-delay_idx[i], dims=-1)
            delays_agg = delays_agg + pattern * weights[:, i].view(B, 1, 1, 1)
        return delays_agg

    def forward(self, queries, keys, values, attn_mask=None):
        # queries/keys/values: (B, L, H, E)
        B, L, H, E = queries.shape
        S = values.shape[1]
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :])
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # 周期依赖发现: 互相关 = IFFT(FFT(Q) * conj(FFT(K)))
        q = queries.permute(0, 2, 3, 1).contiguous()   # (B, H, E, L)
        k = keys.permute(0, 2, 3, 1).contiguous()
        q_fft = torch.fft.rfft(q, dim=-1)
        k_fft = torch.fft.rfft(k, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=-1)  # (B, H, E, L)

        v = values.permute(0, 2, 3, 1).contiguous()
        out = self.time_delay_agg(v, corr)              # (B, H, E, L)
        out = out.permute(0, 3, 1, 2).contiguous()      # (B, L, H, E)

        if self.output_attention:
            return out, corr.permute(0, 3, 1, 2)
        return out, None


class AutoCorrelationLayer(nn.Module):
    """带 Q/K/V 投影的自相关层, 与 Multi-Head Attention 结构对应"""

    def __init__(self, correlation, d_model, n_heads):
        super().__init__()
        d_keys = d_model // n_heads
        self.n_heads = n_heads
        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ============================================================
# 3. 输入嵌入 (值嵌入; 论文中的时间戳嵌入可由 aux 扩展)
# ============================================================
class TokenEmbedding(nn.Module):
    """对每个时间步做 1D 卷积嵌入, 聚合相邻时间步信息"""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        # x: (B, L, C) -> (B, d_model, L) -> (B, L, d_model)
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """
    数据嵌入: 值嵌入 + dropout。
    论文 (DataEmbedding_wo_pos) 不采用位置编码, 周期性由自相关机制捕捉;
    时间上下文 (年/年积日/小时) 可拼接进 aux 后送入, 增强时间表示。
    """

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class MyLayernorm(nn.Module):
    """Autoformer 特制 LayerNorm: 归一化后再减去时间维均值"""

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


# ============================================================
# 4. Autoformer 编码器 / 解码器
# ============================================================
class EncoderLayer(nn.Module):
    """渐进式分解编码层: 自相关注意力 -> 分解 -> FFN -> 分解"""

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecompBlock1D(moving_avg)
        self.decomp2 = SeriesDecompBlock1D(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)          # 丢弃趋势, 保留季节分量

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)    # FFN 后再分解一次
        return res, attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, e_layers, factor, dropout, activation, moving_avg):
        super().__init__()
        attn_layers = []
        for _ in range(e_layers):
            ac = AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False)
            acl = AutoCorrelationLayer(ac, d_model, cfg_model.n_heads)
            attn_layers.append(
                EncoderLayer(
                    acl,
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = MyLayernorm(d_model)

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    """解码层: 自相关自注意力 + 交叉注意力 + FFN, 累积趋势分量"""

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = SeriesDecompBlock1D(moving_avg)
        self.decomp2 = SeriesDecompBlock1D(moving_avg)
        self.decomp3 = SeriesDecompBlock1D(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        sa_value, _ = self.self_attention(x, x, x, attn_mask=x_mask)
        x = x + self.dropout(sa_value)
        x, trend1 = self.decomp1(x)

        ca_value, _ = self.cross_attention(x, cross, cross, attn_mask=cross_mask)
        x = x + self.dropout(ca_value)
        x, trend2 = self.decomp2(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_layers, factor, dropout, activation, moving_avg, c_out):
        super().__init__()
        decoder_layers = []
        for _ in range(d_layers):
            ac_self = AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False)
            ac_cross = AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False)
            decoder_layers.append(
                DecoderLayer(
                    AutoCorrelationLayer(ac_self, d_model, cfg_model.n_heads),
                    AutoCorrelationLayer(ac_cross, d_model, cfg_model.n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.layers = nn.ModuleList(decoder_layers)
        self.norm = MyLayernorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x, trend


# ============================================================
# 5. U-Net 风格 TEC 编码器 / 解码器
# ============================================================
def _conv_out_size(in_size, kernel=3, stride=2, padding=0):
    return (in_size + 2 * padding - kernel) // stride + 1


class UnetDownModule(nn.Module):
    """U-Net 下采样块: 可选步长 2 卷积下采样 + 2 x (卷积 + BN + ReLU)"""

    def __init__(self, in_channels, out_channels, downsample=True, padding=False):
        super().__init__()
        if downsample:
            self.down = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1 if padding else 0
            )
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.down = None
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1 if padding else 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.down is not None:
            x = self.down(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UnetEncoder(nn.Module):
    """TEC 空间编码器: 输入 (B, C_in, H, W), 输出 (B, C_last, enc_h, enc_w)"""

    def __init__(self, num_channels, channels=(64, 128, 256, 512)):
        super().__init__()
        layer = len(channels)
        paddings = [False] * layer  # 与论文一致: 首层与下采样层均不补零
        self.module1 = UnetDownModule(num_channels, channels[0], downsample=False, padding=paddings[0])
        down_modules = [
            UnetDownModule(channels[l - 1], channels[l], downsample=True, padding=paddings[l])
            for l in range(1, layer)
        ]
        self.down_modules = nn.ModuleList(down_modules)
        self.enc_channels = channels

    def forward(self, x, output_inner=False):
        x = self.module1(x)
        feats = []
        for mod in self.down_modules:
            feats.append(x)
            x = mod(x)
        if output_inner:
            return x, feats
        return x, None


class UnetUpsampleBlock(nn.Module):
    """U-Net 上采样块: 转置卷积 x2 -> 拼接跳跃连接 -> 2 x (卷积 + BN + ReLU)"""

    def __init__(self, in_channels, out_channels, skip_in=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1, output_padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels + skip_in, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection=None):
        x = self.relu(self.bn1(self.up(x)))
        if skip_connection is not None:
            xshape = x.shape[-2:]
            sshape = skip_connection.shape[-2:]
            if xshape != sshape:
                if sshape[0] < xshape[0] or sshape[1] < xshape[1]:
                    # 跳跃连接比上采样结果小 -> 零填充
                    sc = torch.zeros(
                        x.shape[0], skip_connection.shape[1], *xshape, device=x.device
                    )
                    sc[:, :, :sshape[0], :sshape[1]] = skip_connection
                else:
                    # 跳跃连接比上采样结果大 -> 裁剪
                    sc = skip_connection[:, :, :xshape[0], :xshape[1]]
            else:
                sc = skip_connection
            x = torch.cat([x, sc], dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UnetDecoder(nn.Module):
    """TEC 空间解码器: 从 (B, C, enc_h, enc_w) 上采样还原"""

    def __init__(self, enc_channels, use_skip=True):
        super().__init__()
        skip_channels = enc_channels[:-1][::-1]            # 与编码器中间特征对应
        in_channels = enc_channels[1:][::-1]
        out_channels = (*in_channels[1:], enc_channels[0])
        self.blocks = nn.ModuleList(
            [
                UnetUpsampleBlock(inc, outc, sc)
                for inc, outc, sc in zip(in_channels, out_channels, skip_channels)
            ]
        )
        self.use_skip = use_skip

    def forward(self, x, inner_feats=None):
        for idx, block in enumerate(self.blocks):
            sc = inner_feats[-idx - 1] if (self.use_skip and inner_feats is not None) else None
            x = block(x, skip_connection=sc)
        return x


# ============================================================
# 6. ED_Autoformer 主模型
# ============================================================
class EDAutoformer(nn.Module):
    """
    ED_Autoformer: Encoder-Decoder Autoformer
    输入: tec (B, T_in, H, W), aux (B, T_in, aux_dim)
    输出: (B, T_out, H, W) 的 TEC 预测
    """

    def __init__(
        self,
        input_length=cfg_train.input_length,
        output_length=cfg_train.output_length,
        aux_dim=cfg_dataset.aux_dim,
        d_model=cfg_model.d_model,
        n_heads=cfg_model.n_heads,
        d_ff=cfg_model.d_ff,
        e_layers=cfg_model.e_layers,
        d_layers=cfg_model.d_layers,
        moving_avg=cfg_model.moving_avg,
        factor=cfg_model.factor,
        dropout=cfg_model.dropout,
        activation=cfg_model.activation,
        label_len=None,
        encode_channels=cfg_model.encode_channels,
    ):
        super().__init__()
        self.seq_len = input_length
        self.pred_len = output_length
        self.label_len = label_len if label_len is not None else input_length // 2
        self.aux_dim = aux_dim
        self.enc_channels = tuple(encode_channels)

        # 零填充 73x73 后, 经首层卷积 + 3 次步长 2 下采样得到编码特征尺寸
        enc_h = _conv_out_size(73, kernel=3, stride=1, padding=0)
        for _ in range(len(self.enc_channels) - 1):
            enc_h = _conv_out_size(enc_h, kernel=3, stride=2, padding=0)
        self.enc_h = self.enc_w = enc_h
        self.tec_c_out = self.enc_h * self.enc_w            # 展平后的空间特征维数 (8x8=64)

        # ---------- TEC 编码器 ----------
        self.visual_encoder = UnetEncoder(input_length, self.enc_channels)
        # 1x1 卷积把通道数对齐到时间序列长度, 使展平特征与时间轴对齐
        self.tec_encode_conv = nn.Conv2d(self.enc_channels[-1], input_length, 1)

        # ---------- Autoformer 时间模块 ----------
        d_in = self.tec_c_out + aux_dim                    # 64 + 6 = 70
        self.enc_embedding = DataEmbedding(d_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(d_in, d_model, dropout)
        self.decomp = SeriesDecompBlock1D(moving_avg)
        self.encoder = Encoder(d_model, d_ff, e_layers, factor, dropout, activation, moving_avg)
        self.decoder = Decoder(
            d_model, d_ff, d_layers, factor, dropout, activation, moving_avg, c_out=self.tec_c_out
        )

        # ---------- TEC 解码器 ----------
        self.tec_decode_conv = nn.Conv2d(output_length, self.enc_channels[-1], 1)
        self.visual_decoder = UnetDecoder(self.enc_channels, use_skip=True)
        self.tec_output_conv = nn.Conv2d(self.enc_channels[0], output_length, 1)

    def forward(self, tec, aux):
        # tec: (B, T_in, H, W), aux: (B, T_in, aux_dim)
        B, T_in, H, W = tec.shape
        device = tec.device

        # ---------- 空间编码: 零填充 73x73, 下采样到 enc_h x enc_w ----------
        pad_h = max(0, 73 - H)
        pad_w = max(0, 73 - W)
        x = F.pad(tec, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else tec

        enc_out, inner_feats = self.visual_encoder(x, output_inner=True)   # (B, 512, 8, 8)
        assert enc_out.shape[-2:] == (self.enc_h, self.enc_w), (
            f"编码器输出尺寸 {tuple(enc_out.shape[-2:])} 与期望 ({self.enc_h}, {self.enc_w}) 不符"
        )

        # 1x1 卷积对齐到 T_in 个时间步, 展平空间维
        enc_out = F.relu(self.tec_encode_conv(enc_out))                    # (B, T_in, 8, 8)
        enc_out = enc_out.reshape(B, T_in, -1)                             # (B, T_in, 64)

        # ---------- 时空融合 ----------
        x_series = torch.cat([enc_out, aux], dim=-1)                       # (B, T_in, 70)

        # ---------- 系列分解, 构造解码器输入 ----------
        seasonal_init, trend_init = self.decomp(x_series)
        dec_seasonal = torch.cat(
            [
                seasonal_init[:, -self.label_len:, :],
                torch.zeros(B, self.pred_len, seasonal_init.shape[-1], device=device),
            ],
            dim=1,
        )
        mean_series = torch.mean(x_series, dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        dec_trend = torch.cat([trend_init[:, -self.label_len:, :], mean_series], dim=1)

        # ---------- Autoformer 编码 - 解码 ----------
        enc_feat, _ = self.encoder(self.enc_embedding(x_series))
        dec_embed = self.dec_embedding(dec_seasonal)
        dec_seasonal_out, dec_trend_out = self.decoder(dec_embed, enc_feat, trend=dec_trend)
        dec_out = dec_seasonal_out + dec_trend_out                         # (B, L_dec, 64)
        dec_out = dec_out[:, -self.pred_len:, :]                           # (B, T_out, 64)

        # ---------- 空间解码: 还原 TEC 图 ----------
        tec_feat = dec_out.reshape(B, self.pred_len, self.enc_h, self.enc_w)
        tec_feat = self.tec_decode_conv(tec_feat)                          # (B, 512, 8, 8)
        tec_map = self.visual_decoder(tec_feat, inner_feats=inner_feats)   # (B, 64, 64, 64)
        tec_map = self.tec_output_conv(tec_map)                            # (B, T_out, 64, 64)
        if tec_map.shape[-2:] != (H, W):
            tec_map = F.interpolate(tec_map, size=(H, W), mode="bilinear", align_corners=False)
        return tec_map                                                     # (B, T_out, H, W)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    model = EDAutoformer().to(cfg_train.device)
    tec_test = torch.randn(2, cfg_train.input_length, 71, 73, device=cfg_train.device)
    aux_test = torch.randn(2, cfg_train.input_length, cfg_dataset.aux_dim, device=cfg_train.device)

    with torch.no_grad():
        pred = model(tec_test, aux_test)

    print("预测输出形状:", tuple(pred.shape))
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
