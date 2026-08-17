import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class STLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # 时间记忆C的门控卷积
        self.conv_x = nn.Conv2d(in_channels, hidden_channels * 4, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_channels, hidden_channels * 4, kernel_size, padding=padding)

        # 空间记忆M的门控卷积（垂直层间传递）
        self.conv_xm = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.conv_cm = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.conv_mm = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size, padding=padding)

        # 输出门与特征融合
        self.conv_o = nn.Conv2d(in_channels + hidden_channels * 2, hidden_channels, kernel_size, padding=padding)
        self.conv_fuse = nn.Conv2d(hidden_channels * 2, hidden_channels, 1)

    def forward(self, x, prev_h, prev_c, prev_m):
        """
        Args:
            x: 当前输入 (B, in_channels, H, W)
            prev_h: 上一时刻隐藏态 (B, hidden_channels, H, W)
            prev_c: 上一时刻时间记忆 (B, hidden_channels, H, W)
            prev_m: 上一层当前时刻空间记忆 (B, hidden_channels, H, W)
        Returns:
            h, c, m: 当前时刻隐藏态、时间记忆、空间记忆
        """
        # 时间记忆更新（水平时序传递）
        gates = self.conv_x(x) + self.conv_h(prev_h) + self.conv_m(prev_m)
        i, f, c_tilde, o = torch.split(gates, self.hidden_channels, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * prev_c + i * torch.tanh(c_tilde)

        # 空间记忆更新（垂直层间传递，从下往上）
        gates_m = self.conv_xm(x) + self.conv_cm(c) + self.conv_mm(prev_m)
        i_m, f_m = torch.split(gates_m, self.hidden_channels, dim=1)
        i_m, f_m = torch.sigmoid(i_m), torch.sigmoid(f_m)
        m_tilde = torch.tanh(self.conv_xm(x) + self.conv_cm(c))
        m = f_m * prev_m + i_m * m_tilde

        # 输出隐藏态
        o = torch.sigmoid(self.conv_o(torch.cat([x, c, m], dim=1)))
        h = o * torch.tanh(self.conv_fuse(torch.cat([c, m], dim=1)))
        return h, c, m
class HaloAttention(nn.Module):
    def __init__(self, dim, block_size=8, halo_size=2, num_heads=4):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        bs, hs = self.block_size, self.halo_size

        # 填充：保证能被block整除，同时四周加halo边缘
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        x_pad = F.pad(x, (hs, hs + pad_w, hs, hs + pad_h))
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

        # 生成QKV并拆分为多头
        qkv = self.qkv_conv(x_pad)
        q, k, v = torch.split(qkv, self.dim, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H_pad, W_pad)
        k = k.view(B, self.num_heads, self.head_dim, H_pad, W_pad)
        v = v.view(B, self.num_heads, self.head_dim, H_pad, W_pad)

        # Query只取中间有效块（去掉halo区域）
        q_valid = q[:, :, :, hs:-hs, hs:-hs]
        q_blocks = F.unfold(q_valid, kernel_size=bs, stride=bs)
        num_blocks = q_blocks.shape[-1]
        q_blocks = q_blocks.view(B, self.num_heads, self.head_dim, bs*bs, num_blocks)

        # Key/Value取带halo的完整块
        kv_block_size = bs + 2 * hs
        k_blocks = F.unfold(k, kernel_size=kv_block_size, stride=bs)
        v_blocks = F.unfold(v, kernel_size=kv_block_size, stride=bs)
        k_blocks = k_blocks.view(B, self.num_heads, self.head_dim, kv_block_size**2, num_blocks)
        v_blocks = v_blocks.view(B, self.num_heads, self.head_dim, kv_block_size**2, num_blocks)

        # 局部自注意力计算
        attn = torch.einsum('bhdqn,bhdkn->bhqkn', q_blocks, k_blocks) * self.scale
        attn = F.softmax(attn, dim=-2)
        out = torch.einsum('bhqkn,bhdkn->bhdqn', attn, v_blocks)
        out = out.contiguous().view(B, self.dim, bs*bs, num_blocks)

        # 折叠回原图并裁剪回原始尺寸
        out = F.fold(out, output_size=(H_pad - 2*hs, W_pad - 2*hs), kernel_size=bs, stride=bs)
        out = out[:, :, :H, :W]
        return self.out_conv(out)