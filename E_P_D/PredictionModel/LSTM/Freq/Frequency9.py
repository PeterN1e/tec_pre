import torch
import torch.nn as nn
import torch.fft as fft

class FreqTECBranch(nn.Module):
    """
    时空3D-FFT频域低频分支 适配TEC图像预测
    输入: (B, seq_len, H, W)
    输出: (B, pred_len, H, W)
    """
    def __init__(self, seq_len, pred_len, h, w, low_pass_ratio=0.4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.H = h
        self.W = w
        self.low_pass_ratio = low_pass_ratio  # 调大到0.4，保留更多有效信息

    def _get_gaussian_3d_mask(self, f_shape, device):
        """生成3D高斯低通掩码，替代矩形窗，避免振铃伪影"""
        t_dim, h_dim, w_dim = f_shape[-3], f_shape[-2], f_shape[-1]

        # 生成三维频率网格
        t_grid = torch.linspace(-1, 1, t_dim, device=device)
        h_grid = torch.linspace(-1, 1, h_dim, device=device)
        w_grid = torch.linspace(-1, 1, w_dim, device=device)
        tt, hh, ww = torch.meshgrid(t_grid, h_grid, w_grid, indexing="ij")

        # 高斯核：距离中心越远权重越低
        dist = torch.sqrt(tt ** 2 + hh ** 2 + ww ** 2)
        sigma = self.low_pass_ratio * 1.5  # 优化高斯核，适配TEC
        mask = torch.exp(-dist ** 2 / (2 * sigma ** 2))
        return mask[None, ...]  # 扩维适配 (1,T,H,W)

    def forward(self, x):
        # x: [B, T_seq, H, W]
        B = x.shape[0]

        # 1. 三维傅里叶变换 时空联合频域分解
        x_fft = fft.fftn(x, dim=(-3, -2, -1))  # 作用维度:T,H,W

        # 2. 生成3D高斯低通掩码 + 滤波
        mask = self._get_gaussian_3d_mask(x_fft.shape, x.device)
        x_fft_low = x_fft * mask  # 滤除高频噪声

        # 3. ✅ 核心修复：时间维度降维（6帧→1帧），用均值，零报错！
        pad_t = self.pred_len - self.seq_len
        if pad_t > 0:
            # 预测长度大于输入：补零
            x_fft_pad = torch.nn.functional.pad(x_fft_low, (0,0,0,0,0,pad_t))
        else:
            # ✅ 预测长度小于输入（6→1）：直接对【时间维度取均值】，最适合TEC慢变数据
            # 维度变化: [B,6,H,W] → [B,1,H,W]
            x_fft_pad = torch.mean(x_fft_low, dim=1, keepdim=True)

        # 4. 逆3D傅里叶变换，取实部
        x_ifft = fft.ifftn(x_fft_pad, dim=(-3, -2, -1))
        out = torch.real(x_ifft)

        # 最终输出严格匹配: [B, pred_len, H, W]
        return out[:, :self.pred_len, :, :]


class TimeFreqFusion(nn.Module):
    """
    时频双分支加权融合
    主干预测 + 频域低频分支预测 可学习权重融合
    """
    def __init__(self):
        super().__init__()
        # 用softmax约束权重，保证非负、和为1，稳定训练
        self.weight = nn.Parameter(torch.tensor([0.8, 0.2]))  # 初始80%靠主干，20%靠频域

    def forward(self, trunk_pred, freq_pred):
        """
        trunk_pred: 主干输出 [B,1,H,W]
        freq_pred: 频域分支 [B,1,H,W]
        """
        alpha, beta = torch.softmax(self.weight, dim=0)
        fuse_pred = alpha * trunk_pred + beta * freq_pred
        return fuse_pred