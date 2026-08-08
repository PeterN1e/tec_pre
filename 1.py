"""
GA-Predrnn: Enhanced Spatiotemporal Network for Global Ionospheric TEC Prediction
论文来源: IGARSS 2025, Tsinghua University
代码实现基于论文描述的架构

核心组件:
1. ST-LSTM Cell - 时空长短期记忆单元
2. Predrnn - 时空循环预测网络
3. Halo Attention - 局部自注意力机制
4. ST-Attention Module - 时空注意力模块
5. Discriminator with D3Blocks - 判别器
6. Cross-Training Strategy - 交叉训练策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


# ============================================================
# 1. ST-LSTM Cell (时空长短期记忆单元)
# ============================================================
# Predrnn的核心创新在于引入了spatiotemporal memory (M),
# 与传统的temporal memory (C) 不同, M不仅在时间维度上传递,
# 还在垂直方向(层间)传递, 实现跨层的时空信息流动.
#
# 论文中: C和J为记忆存储单元, s为时间记忆的水平传递函数,
# J控制时空记忆的垂直传递, H为隐藏状态.
# ============================================================

class STLSTMCell(nn.Module):
    """
    时空长短期记忆单元 (ST-LSTM Cell)

    与标准LSTM的区别:
    - 标准LSTM只有temporal memory C
    - ST-LSTM额外引入spatiotemporal memory M
    - M在层间垂直传递, 捕获不同层级的时空特征

    参数:
        in_channels: 输入通道数
        hidden_channels: 隐藏状态通道数
        kernel_size: 卷积核大小 (默认3, 使用padding保持尺寸)
        num_layers: ST-LSTM堆叠层数 (用于确定M的传递方向)
    """

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # 输入到状态的卷积 (input-to-state)
        # 将输入x_t映射到4个门控信号的空间
        self.conv_x = nn.Conv2d(
            in_channels, hidden_channels * 4,
            kernel_size, padding=padding, bias=True
        )

        # 状态到状态的卷积 (state-to-state)
        # 将上一时刻的隐藏状态H_{t-1}映射到4个门控信号的空间
        self.conv_h = nn.Conv2d(
            hidden_channels, hidden_channels * 4,
            kernel_size, padding=padding, bias=True
        )

        # 时空记忆M的卷积
        # M在垂直方向(层间)传递, 这是Predrnn的关键创新
        self.conv_m = nn.Conv2d(
            hidden_channels, hidden_channels * 4,
            kernel_size, padding=padding, bias=True
        )

        # 输出门的额外卷积, 用于结合时空记忆M来生成最终隐藏状态
        self.conv_o = nn.Conv2d(
            hidden_channels * 2, hidden_channels,
            kernel_size, padding=padding, bias=True
        )

        # Layer normalization for stability
        self.norm = nn.LayerNorm([hidden_channels])

    def forward(self, x_t, h_prev, c_prev, m_prev):
        """
        前向传播

        参数:
            x_t: 当前时刻输入 [B, C_in, H, W]
            h_prev: 上一时刻隐藏状态 [B, C_hidden, H, W]
            c_prev: 上一时刻temporal memory [B, C_hidden, H, W]
            m_prev: 上一时刻(上一层)的spatiotemporal memory [B, C_hidden, H, W]

        返回:
            h_t: 当前时刻隐藏状态
            c_t: 更新后的temporal memory
            m_t: 更新后的spatiotemporal memory (用于传递给下一层)
        """
        # ---- Step 1: 计算输入门控信号 ----
        # 将输入x和上一时刻隐藏状态h分别卷积后相加
        gates_x = self.conv_x(x_t)        # [B, 4*C, H, W]
        gates_h = self.conv_h(h_prev)     # [B, 4*C, H, W]
        gates_m = self.conv_m(m_prev)     # [B, 4*C, H, W]

        # 三路信号融合
        gates = gates_x + gates_h + gates_m

        # ---- Step 2: 分割为四个门 ----
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)   # 输入门 (input gate)
        f = torch.sigmoid(f)   # 遗忘门 (forget gate)
        g = torch.tanh(g)      # 输入调制 (input modulation)
        o = torch.sigmoid(o)   # 输出门 (output gate)

        # ---- Step 3: 更新temporal memory C ----
        # C_t = f ⊙ C_{t-1} + i ⊙ g
        # 这是标准LSTM的memory更新, 在时间维度水平传递
        c_t = f * c_prev + i * g

        # ---- Step 4: 更新spatiotemporal memory M ----
        # M的关键: 它不仅在时间维度更新, 还在层间垂直传递
        # 这里使用与C类似的更新方式, 但M会被传递给下一层
        m_t = f * m_prev + i * g  # M也经历类似的门控更新

        # ---- Step 5: 生成隐藏状态H ----
        # 结合temporal memory C和spatiotemporal memory M
        h_t = o * torch.tanh(self.conv_o(torch.cat([c_t, m_t], dim=1)))

        return h_t, c_t, m_t


# ============================================================
# 2. Predrnn Model (时空循环预测网络)
# ============================================================
# Predrnn通过堆叠多层ST-LSTM构建encoder-decoder结构.
# 关键设计: spatiotemporal memory M在层间垂直流动
# (论文图1中的J方向), 而temporal memory C在时间维度
# 水平流动 (论文图1中的s方向).
# ============================================================

class Predrnn(nn.Module):
    """
    Predrnn: 时空循环预测网络

    结构: 堆叠N层ST-LSTM
    - 每层接收上一层的输出作为输入
    - temporal memory C在每层内部沿时间传递
    - spatiotemporal memory M跨层垂直传递

    论文中指出: Predrnn不能在时间维度上充分捕获时间特征,
    因此需要引入时空注意力模块来增强.
    """

    def __init__(self, input_channels, hidden_channels, num_layers,
                 output_channels=None, seq_len=24, pred_len=12):
        """
        参数:
            input_channels: 输入通道数 (TEC + 辅助数据)
            hidden_channels: 每层ST-LSTM的隐藏通道数
            num_layers: ST-LSTM堆叠层数 (论文图1中的n)
            output_channels: 输出通道数 (默认等于input_channels)
            seq_len: 输入序列长度 (论文中J=24, 即前2天)
            pred_len: 预测序列长度 (论文中K=12, 即第3天)
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_channels = output_channels or input_channels

        # 为每一层创建ST-LSTM Cell
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(STLSTMCell(in_ch, hidden_channels))

        # 输出投影层: 将隐藏状态映射回输入通道空间
        self.output_conv = nn.Conv2d(
            hidden_channels, self.output_channels,
            kernel_size=1, bias=True
        )

    def forward(self, input_seq, return_hidden=False):
        """
        参数:
            input_seq: [B, T, C, H, W] 输入序列
            return_hidden: 是否返回所有隐藏状态 (用于注意力模块)

        返回:
            predictions: [B, pred_len, C_out, H, W] 预测序列
            hidden_states: (可选) 所有时间步的隐藏状态列表
        """
        B, T, C, H, W = input_seq.shape

        # 初始化所有层的隐藏状态、temporal memory、spatiotemporal memory
        h_list = [torch.zeros(B, self.hidden_channels, H, W,
                              device=input_seq.device)
                  for _ in range(self.num_layers)]
        c_list = [torch.zeros(B, self.hidden_channels, H, W,
                              device=input_seq.device)
                  for _ in range(self.num_layers)]
        m_list = [torch.zeros(B, self.hidden_channels, H, W,
                              device=input_seq.device)
                  for _ in range(self.num_layers)]

        predictions = []
        all_hidden_states = []  # 收集最后层的隐藏状态

        # ---- 编码阶段 + 解码阶段 (统一处理) ----
        total_steps = self.seq_len + self.pred_len

        for t in range(total_steps):
            if t < self.seq_len:
                # 编码阶段: 使用真实输入
                x_t = input_seq[:, t]  # [B, C, H, W]
            else:
                # 解码阶段: 使用上一步的预测作为输入 (自回归)
                x_t = pred  # 使用上一步的预测输出

            # 逐层前向传播
            # 关键: M的垂直流动 - 上一层的m_t作为下一层的m_prev
            current_input = x_t
            for layer_idx in range(self.num_layers):
                h_list[layer_idx], c_list[layer_idx], m_list[layer_idx] = \
                    self.cells[layer_idx](
                        current_input,
                        h_list[layer_idx],  # 该层上一时刻的H
                        c_list[layer_idx],  # 该层上一时刻的C
                        m_list[layer_idx]   # 该层上一时刻的M (或来自上一层)
                    )
                current_input = h_list[layer_idx]

            # 最后一层的隐藏状态作为输出
            last_hidden = h_list[-1]

            # 收集解码阶段的隐藏状态 (用于注意力模块)
            if return_hidden:
                all_hidden_states.append(last_hidden)

            # 只在解码阶段生成预测
            if t >= self.seq_len:
                pred = self.output_conv(last_hidden)  # [B, C_out, H, W]
                predictions.append(pred)

        # 堆叠预测结果
        predictions = torch.stack(predictions, dim=1)  # [B, pred_len, C_out, H, W]

        if return_hidden:
            all_hidden_states = torch.stack(all_hidden_states, dim=1)
            return predictions, all_hidden_states
        return predictions


# ============================================================
# 3. Halo Attention (局部自注意力机制)
# ============================================================
# 来自文献[14]: Vaswani et al., "Scaling Local Self-Attention
# for Parameter Efficient Visual Backbones", CVPR 2021
#
# 核心思想:
# 1. 将图像划分为多个block
# 2. 每个block向外扩展"光晕"(Halo)区域, 扩大感受野
# 3. 在扩展后的block内进行局部自注意力计算
#
# 论文指出: 在TEC预测中, 远处位置的TEC对目标区域影响很小,
# 因此只需关注邻近区域, Halo Attention正好满足这一需求.
# ============================================================

class HaloAttention(nn.Module):
    """
    Halo Attention 机制

    步骤:
    1. 将输入特征图划分为 block_size × block_size 的块
    2. 每个块向外扩展 halo_size 个像素 (使用邻近块的信息填充)
    3. 在扩展后的块内计算多头自注意力

    参数:
        dim: 输入通道数
        block_size: 每个块的空间大小
        halo_size: 光晕扩展大小 (每个方向扩展的像素数)
        num_heads: 注意力头数
        dim_head: 每个头的维度
    """

    def __init__(self, dim, block_size=8, halo_size=3,
                 num_heads=4, dim_head=32):
        super().__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        # QKV投影
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # 位置编码 (相对位置偏置)
        self.rel_bias = nn.Parameter(
            torch.zeros(num_heads,
                       (block_size + 2 * halo_size) ** 2,
                       (block_size + 2 * halo_size) ** 2)
        )

    def forward(self, x):
        """
        参数:
            x: [B, C, H, W] 输入特征图

        返回:
            out: [B, C, H, W] 注意力增强后的特征图
        """
        B, C, H, W = x.shape
        bs = self.block_size
        hs = self.halo_size
        padded_bs = bs + 2 * hs  # 扩展后的块大小

        # ---- Step 1: Padding (确保H, W可被block_size整除) ----
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))  # [B, C, H', W']
        _, _, H_p, W_p = x_padded.shape

        # ---- Step 2: Halo Padding (光晕填充) ----
        # 用邻近区域的信息填充每个块的外围, 扩大感受野
        # 这是Halo Attention的核心: 不是全局注意力, 而是带上下文的局部注意力
        x_halo = F.pad(x_padded, (hs, hs, hs, hs))  # [B, C, H'+2h, W'+2h]

        # ---- Step 3: 提取所有扩展后的块 ----
        num_blocks_h = H_p // bs
        num_blocks_w = W_p // bs

        # 使用unfold操作提取所有块
        # 每个块的大小为 padded_bs × padded_bs
        blocks = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # 计算在padded特征图中的起止坐标
                h_start = i * bs
                w_start = j * bs
                # 在halo-padded特征图中, 坐标偏移了hs
                block = x_halo[:, :,
                              h_start:h_start + padded_bs,
                              w_start:w_start + padded_bs]
                blocks.append(block)

        # [B, num_blocks, C, padded_bs, padded_bs]
        blocks = torch.stack(blocks, dim=1)
        num_blocks = num_blocks_h * num_blocks_w

        # ---- Step 4: 计算局部自注意力 ----
        # Reshape: [B * num_blocks, C, padded_bs^2]
        blocks_flat = blocks.reshape(B * num_blocks, C, -1).permute(0, 2, 1)
        # blocks_flat: [B*num_blocks, num_tokens, C]

        # QKV投影
        q = self.to_q(blocks_flat)  # [B*N, T, inner_dim]
        kv = self.to_kv(blocks_flat)
        k, v = kv.chunk(2, dim=-1)

        # 多头注意力
        T = blocks_flat.shape[1]
        q = q.reshape(B * num_blocks, T, self.num_heads, self.dim_head)
        k = k.reshape(B * num_blocks, T, self.num_heads, self.dim_head)
        v = v.reshape(B * num_blocks, T, self.num_heads, self.dim_head)

        # [B*N, heads, T, dim_head]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self.rel_bias.unsqueeze(0)  # 加入相对位置偏置
        attn = F.softmax(attn, dim=-1)

        # 加权求和
        out = torch.matmul(attn, v)  # [B*N, heads, T, dim_head]
        out = out.permute(0, 2, 1, 3).reshape(B * num_blocks, T, -1)
        out = self.to_out(out)  # [B*N, T, C]

        # ---- Step 5: 将块重新组合为特征图 ----
        out = out.permute(0, 2, 1)  # [B*N, C, T]
        out = out.reshape(B, num_blocks, C, padded_bs, padded_bs)

        # 只保留中心区域 (去掉halo部分)
        out = out[:, :, :, hs:hs+bs, hs:hs+bs]

        # 重新排列为特征图
        result = torch.zeros(B, C, H_p, W_p, device=x.device)
        idx = 0
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                result[:, :, i*bs:(i+1)*bs, j*bs:(j+1)*bs] = out[:, idx]
                idx += 1

        # 去掉padding
        result = result[:, :, :H, :W]

        return result


# ============================================================
# 4. Spatiotemporal Attention Module (时空注意力模块)
# ============================================================
# 论文的核心创新之一.
#
# Halo Attention只在空间域做自注意力, 但TEC预测基于时空框架.
# 论文的做法:
# 1. 在Predrnn的每一层(或最后层)后应用Halo Attention进行空间注意力
# 2. 将上一层(或上一时间步)的隐藏状态与当前张量拼接
# 3. 这样历史时序信息就融入了当前状态
#
# 论文原文: "the hidden state from the last layer is concatenated
# with the current tensor, integrating historical time-series
# information with the current state."
#
# 效果: 模型能同时考虑过去隐藏状态和当前预测,
# 增强时间流动(temporal flow), 更好学习时间步间的关系.
# ============================================================

class SpatiotemporalAttention(nn.Module):
    """
    时空注意力模块

    结合Halo Attention(空间注意力)与隐藏状态拼接(时间增强):
    1. 对输入特征应用Halo Attention进行空间特征提取
    2. 将历史隐藏状态与当前特征拼接, 通过1×1卷积融合
    3. 输出增强后的时空特征

    参数:
        dim: 特征通道数
        block_size: Halo Attention的块大小
        halo_size: Halo光晕大小
        num_heads: 注意力头数
    """

    def __init__(self, dim, block_size=8, halo_size=3, num_heads=4):
        super().__init__()

        # Halo Attention: 空间域的局部自注意力
        self.halo_attn = HaloAttention(
            dim=dim, block_size=block_size,
            halo_size=halo_size, num_heads=num_heads
        )

        # 时间增强: 将拼接后的特征(当前+历史隐藏状态)融合
        # 输入通道为 2*dim (因为拼接了两份)
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        )

        # Layer normalization
        self.norm1 = nn.GroupNorm(min(32, dim), dim)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)

        # 残差缩放参数 (可学习)
        self.gamma_spatial = nn.Parameter(torch.zeros(1))
        self.gamma_temporal = nn.Parameter(torch.zeros(1))

    def forward(self, x, h_prev=None):
        """
        参数:
            x: 当前特征张量 [B, C, H, W]
            h_prev: 上一时间步(或上一层)的隐藏状态 [B, C, H, W]
                    如果为None, 则不做时间增强

        返回:
            out: 时空增强后的特征 [B, C, H, W]
        """
        # ---- Step 1: 空间注意力 (Halo Attention) ----
        residual = x
        x_norm = self.norm1(x)
        spatial_out = self.halo_attn(x_norm)
        x = residual + self.gamma_spatial * spatial_out  # 残差连接

        # ---- Step 2: 时间增强 (隐藏状态拼接) ----
        if h_prev is not None:
            residual = x
            x_norm = self.norm2(x)
            # 核心操作: 将历史隐藏状态与当前张量拼接
            # 论文原文: "the hidden state from the last layer
            # is concatenated with the current tensor"
            concatenated = torch.cat([x_norm, h_prev], dim=1)  # [B, 2C, H, W]
            temporal_out = self.temporal_fusion(concatenated)   # [B, C, H, W]
            x = residual + self.gamma_temporal * temporal_out   # 残差连接

        return x


# ============================================================
# 5. Predictor (预测器)
# ============================================================
# 整合Predrnn + 时空注意力模块的完整预测器.
# 采用编码器-解码器架构:
# - 编码器: 多层ST-LSTM处理历史序列
# - 注意力: 在编码器和解码器之间/内部应用时空注意力
# - 解码器: 多层ST-LSTM自回归生成预测序列
# ============================================================

class Predictor(nn.Module):
    """
    GA-Predrnn的预测器

    论文中描述: "The predictor primarily employs an encoder-decoder
    architecture, which consists of the Predrnn model and a
    spatiotemporal attention module."

    结构:
    1. Predrnn编码器: 处理历史TEC序列
    2. 时空注意力模块: 增强时空特征提取
    3. Predrnn解码器: 自回归生成预测序列
    """

    def __init__(self, input_channels, hidden_channels, num_layers,
                 pred_len=12, block_size=8, halo_size=3, num_heads=4):
        """
        参数:
            input_channels: 输入通道数
            hidden_channels: ST-LSTM隐藏通道数
            num_layers: ST-LSTM层数
            pred_len: 预测时间步数 (论文中K=12)
            block_size: Halo Attention块大小
            halo_size: Halo光晕大小
            num_heads: 注意力头数
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.pred_len = pred_len

        # 编码器: Predrnn (输入→隐藏)
        self.encoder = Predrnn(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            output_channels=hidden_channels,
            pred_len=0  # 编码器不预测, 只提取特征
        )

        # 时空注意力模块
        self.st_attention = SpatiotemporalAttention(
            dim=hidden_channels,
            block_size=block_size,
            halo_size=halo_size,
            num_heads=num_heads
        )

        # 解码器: Predrnn (隐藏→隐藏)
        self.decoder = Predrnn(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            output_channels=hidden_channels,
            seq_len=0,  # 解码器无编码输入
            pred_len=pred_len
        )

        # 输出投影: 隐藏空间→TEC预测空间
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, input_channels, 1)
        )

    def forward(self, x):
        """
        参数:
            x: [B, T_in, C, H, W] 历史TEC序列

        返回:
            pred: [B, pred_len, C_out, H, W] TEC预测序列
        """
        # ---- 编码阶段 ----
        # 用Predrnn处理历史序列, 提取时空特征
        # 同时获取最后层的隐藏状态 (用于注意力模块)
        B, T_in, C, H, W = x.shape

        # 编码: 获取编码器最后时刻的隐藏特征
        enc_out, enc_hidden = self.encoder(x, return_hidden=True)
        # enc_hidden: [B, T_in, hidden, H, W] - 编码器各时间步最后层的隐藏状态

        # 取最后一个编码时间步的隐藏状态作为上下文
        context = enc_hidden[:, -1]  # [B, hidden, H, W]

        # ---- 时空注意力增强 ----
        # 论文: 将隐藏状态与当前张量拼接, 增强时间流
        # 这里用最后一个编码状态作为h_prev, context作为空间特征
        enhanced_context = self.st_attention(context, h_prev=context)
        # enhanced_context: [B, hidden, H, W]

        # ---- 解码阶段 ----
        # 将增强后的上下文作为解码器的初始输入
        # 构建输入序列: 重复enhanced_context作为每步输入
        dec_input = enhanced_context.unsqueeze(1).repeat(1, self.pred_len, 1, 1, 1)
        # dec_input: [B, pred_len, hidden, H, W]

        pred = self.decoder(dec_input)  # [B, pred_len, hidden, H, W]

        # 输出投影
        pred = self.output_proj(pred.reshape(-1, self.hidden_channels, H, W))
        pred = pred.reshape(B, self.pred_len, -1, H, W)

        return pred


# ============================================================
# 6. D3Block (判别器核心模块)
# ============================================================
# 论文图3描述: 判别器由 space-to-depth block, 两个D3Block,
# 四个D3Block, 一个全连接层组成.
# D3Block使用谱归一化(spectral normalization)来规范化卷积.
#
# 谱归一化的作用: 约束判别器的Lipschitz常数, 稳定GAN训练.
# ============================================================

class D3Block(nn.Module):
    """
    D3Block: 判别器中的基本卷积块

    包含:
    - 谱归一化的卷积层 (spectral normalization)
    - LeakyReLU激活
    - 可选的下采样

    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长 (stride=2时进行下采样)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride,
            padding=1, bias=True
        ))
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out


# ============================================================
# 7. Discriminator (判别器)
# ============================================================
# 论文描述: "The discriminator is designed to distinguish between
# predictions generated by the predictor and the real observations
# of global ionospheric TEC."
#
# 结构 (论文图3):
# - Space-to-Depth Block: 将空间信息重排到通道维度
# - 2个D3Block: 初步特征提取
# - 4个D3Block: 深层特征提取
# - 全连接层: 输出真/假分数
# ============================================================

class SpaceToDepth(nn.Module):
    """
    Space-to-Depth变换

    将空间维度的信息重排到通道维度:
    输入: [B, C, H, W]
    输出: [B, C * block_size^2, H/block_size, W/block_size]

    这种变换保留了所有信息, 同时减小空间尺寸.
    """

    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        B, C, H, W = x.shape
        bs = self.block_size

        # 检查是否可以整除
        assert H % bs == 0 and W % bs == 0, \
            f"H={H}, W={W} must be divisible by block_size={bs}"

        # 重排空间像素到通道维度
        x = x.reshape(B, C, H // bs, bs, W // bs, bs)
        x = x.permute(0, 1, 3, 5, 2, 4)  # [B, C, bs, bs, H//bs, W//bs]
        x = x.reshape(B, C * bs * bs, H // bs, W // bs)

        return x


class Discriminator(nn.Module):
    """
    判别器

    论文中: 给定预测序列, 可以生成12对图像.
    输入IGS TEC图时输出高分, 输入预测图时输出低分数.

    参数:
        input_channels: 输入通道数
        hidden_channels: 隐藏通道数
    """

    def __init__(self, input_channels=1, hidden_channels=64):
        super().__init__()

        # Space-to-Depth: 减小空间尺寸, 增加通道数
        self.space_to_depth = SpaceToDepth(block_size=2)
        s2d_channels = input_channels * 4  # 2*2=4

        # 前两个D3Block (论文: "two D3Blocks")
        self.initial_blocks = nn.Sequential(
            D3Block(s2d_channels, hidden_channels, stride=1),
            D3Block(hidden_channels, hidden_channels * 2, stride=2),
        )

        # 后四个D3Block (论文: "four D3Blocks")
        self.deep_blocks = nn.Sequential(
            D3Block(hidden_channels * 2, hidden_channels * 4, stride=2),
            D3Block(hidden_channels * 4, hidden_channels * 4, stride=1),
            D3Block(hidden_channels * 4, hidden_channels * 8, stride=2),
            D3Block(hidden_channels * 8, hidden_channels * 8, stride=1),
        )

        # 全局平均池化 + 全连接层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = spectral_norm(nn.Linear(hidden_channels * 8, 1))

    def forward(self, x):
        """
        参数:
            x: [B, C, H, W] TEC图 (真实或预测)

        返回:
            score: [B, 1] 真实度分数
        """
        # Space-to-Depth变换
        out = self.space_to_depth(x)  # [B, C*4, H/2, W/2]

        # D3Blocks特征提取
        out = self.initial_blocks(out)
        out = self.deep_blocks(out)

        # 全局池化 + 全连接
        out = self.global_pool(out)  # [B, hidden*8, 1, 1]
        out = out.flatten(1)          # [B, hidden*8]
        score = self.fc(out)           # [B, 1]

        return score


# ============================================================
# 8. GA-Predrnn (完整模型)
# ============================================================
# 整合Predictor和Discriminator, 实现交叉训练策略.
#
# 训练流程 (论文图4):
# 1. TEC数据输入Predictor生成预测图
# 2. 预测图和真实图一起输入Discriminator
# 3. 双传播机制:
#    - 第一次反向传播: 更新Discriminator
#    - 第二次反向传播: 更新Predictor
# ============================================================

class GAPredrnn(nn.Module):
    """
    GA-Predrnn: 增强型时空预测网络

    论文两个核心创新:
    1. 时空注意力模块 (Spatiotemporal Attention Module)
    2. 基于判别器的交叉训练策略 (Cross-Training Strategy)

    参数:
        input_channels: 输入通道数
        hidden_channels: 隐藏通道数
        num_layers: ST-LSTM层数
        seq_len: 输入序列长度 (默认24)
        pred_len: 预测序列长度 (默认12)
    """

    def __init__(self, input_channels=4, hidden_channels=64,
                 num_layers=4, seq_len=24, pred_len=12,
                 block_size=8, halo_size=3, num_heads=4):
        super().__init__()

        # 预测器: Predrnn + 时空注意力
        self.predictor = Predictor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            pred_len=pred_len,
            block_size=block_size,
            halo_size=halo_size,
            num_heads=num_heads
        )

        # 判别器: D3Blocks + 谱归一化
        self.discriminator = Discriminator(
            input_channels=input_channels,
            hidden_channels=32  # 判别器通道数通常小于生成器
        )

        self.pred_len = pred_len

    def forward(self, x, return_disc_scores=False):
        """
        前向传播

        参数:
            x: [B, T_in, C, H, W] 历史序列
            return_disc_scores: 是否返回判别器分数

        返回:
            predictions: [B, pred_len, C, H, W] 预测序列
            disc_scores_pred: (可选) 判别器对预测图的分数
        """
        # ---- 预测器生成预测 ----
        predictions = self.predictor(x)  # [B, pred_len, C, H, W]

        if return_disc_scores:
            # 对每个预测时间步计算判别器分数
            disc_scores_list = []
            for t in range(self.pred_len):
                score = self.discriminator(predictions[:, t])
                disc_scores_list.append(score)
            disc_scores_pred = torch.stack(disc_scores_list, dim=1)
            return predictions, disc_scores_pred

        return predictions


# ============================================================
# 9. Loss Functions (损失函数)
# ============================================================
# 论文公式(2): 判别器损失
# L_D = relu(1 - D(x)) + relu(1 + D(p))
#
# 论文公式(3): 预测器损失
# L_P = α × L_IGS + β × L_AUX - mean(D(p))
# 其中 α >> β, 以突出TEC预测的重要性.
# ============================================================

class DiscriminatorLoss(nn.Module):
    """
    判别器损失 (论文公式2)

    L_D = relu(1 - D(x)) + relu(1 + D(p))

    - D(x): 判别器对真实图的评分 (应该高分, 接近1)
    - D(p): 判别器对预测图的评分 (应该低分, 接近-1)

    注: 论文指出原始GAN应使用极大极小损失, 但为训练方便进行了简化.
    这里使用ReLU的形式, 类似于Hinge Loss的变体.
    """

    def forward(self, real_scores, fake_scores):
        """
        参数:
            real_scores: [B, T] 判别器对真实TEC图的评分
            fake_scores: [B, T] 判别器对预测TEC图的评分

        返回:
            loss: 标量判别器损失
        """
        # 真实图应该得到高分 (接近1)
        loss_real = F.relu(1 - real_scores).mean()
        # 预测图应该得到低分 (接近-1)
        loss_fake = F.relu(1 + fake_scores).mean()

        loss = loss_real + loss_fake
        return loss


class PredictorLoss(nn.Module):
    """
    预测器损失 (论文公式3)

    L_P = α × L_IGS + β × L_AUX - mean(D(p))

    - L_IGS: TEC数据的L1损失 (主要损失)
    - L_AUX: 辅助数据(F10.7, DST, AP)的L1损失 (辅助损失)
    - mean(D(p)): 判别器对预测图的平均评分 (对抗损失)
    - α >> β: 根据实验结果, α应远大于β

    论文原文: "According to the results of many experiments,
    α should be much larger than β to highlight the prediction
    of global ionospheric TEC"
    """

    def __init__(self, alpha=10.0, beta=1.0):
        """
        参数:
            alpha: TEC损失权重 (较大值, 论文建议远大于beta)
            beta: 辅助数据损失权重 (较小值)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_tec, true_tec, pred_aux=None,
                true_aux=None, disc_scores=None):
        """
        参数:
            pred_tec: [B, T, C_tec, H, W] 预测TEC
            true_tec: [B, T, C_tec, H, W] 真实TEC
            pred_aux: [B, T, C_aux, H, W] 预测辅助数据 (可选)
            true_aux: [B, T, C_aux, H, W] 真实辅助数据 (可选)
            disc_scores: [B, T] 判别器对预测的评分

        返回:
            total_loss: 总预测器损失
            loss_dict: 各项损失的字典 (用于监控)
        """
        loss_dict = {}

        # L_IGS: TEC数据的L1损失
        l_igs = self.l1_loss(pred_tec, true_tec)
        loss_dict['L_IGS'] = l_igs.item()

        # L_AUX: 辅助数据的L1损失
        l_aux = torch.tensor(0.0, device=pred_tec.device)
        if pred_aux is not None and true_aux is not None:
            l_aux = self.l1_loss(pred_aux, true_aux)
        loss_dict['L_AUX'] = l_aux.item()

        # 组合损失
        total_loss = self.alpha * l_igs + self.beta * l_aux

        # 对抗损失: -mean(D(p))
        if disc_scores is not None:
            adv_loss = -disc_scores.mean()
            total_loss = total_loss + adv_loss
            loss_dict['adv_loss'] = adv_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ============================================================
# 10. Cross-Training Strategy (交叉训练策略)
# ============================================================
# 论文核心: 双传播机制 (Dual Propagation Mechanism)
#
# 在一次完整参数更新中执行两次反向传播:
# 1. 第一次: 更新判别器 (提供预测图的大致轮廓)
# 2. 第二次: 更新预测器 (细化预测细节)
#
# 训练阶段策略:
# - 初期: 判别器损失主导, 建立稳健基线
# - 后期: 预测器损失主导, 精修细节
# ============================================================

class CrossTrainer:
    """
    交叉训练策略

    论文描述: "To implement a comprehensive one-step parameter update,
    we perform two back-propagation passes: one for the discriminator
    and the other for the predictor."

    训练流程:
    1. 前向传播: Predictor生成预测
    2. 判别器评估: 分别对真实图和预测图评分
    3. 反向传播1: 计算判别器损失, 更新判别器
    4. 反向传播2: 计算预测器损失(含对抗项), 更新预测器
    """

    def __init__(self, model, lr_g=1e-4, lr_d=1e-4,
                 alpha=10.0, beta=1.0, disc_weight_schedule=True):
        """
        参数:
            model: GA-Predrnn模型
            lr_g: 预测器学习率
            lr_d: 判别器学习率
            alpha: TEC损失权重
            beta: 辅助数据损失权重
            disc_weight_schedule: 是否使用训练阶段调度
                (初期判别器主导, 后期预测器主导)
        """
        self.model = model
        self.disc_weight_schedule = disc_weight_schedule

        # 分离优化器
        self.optimizer_G = torch.optim.Adam(
            model.predictor.parameters(), lr=lr_g, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            model.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
        )

        # 损失函数
        self.disc_loss_fn = DiscriminatorLoss()
        self.pred_loss_fn = PredictorLoss(alpha=alpha, beta=beta)

    def train_step(self, input_seq, true_tec, epoch=0, total_epochs=100):
        """
        单步训练 (包含两次反向传播)

        参数:
            input_seq: [B, T_in, C, H, W] 输入序列
            true_tec: [B, T_pred, C, H, W] 真实TEC序列
            epoch: 当前epoch
            total_epochs: 总epoch数

        返回:
            loss_info: 损失信息字典
        """
        B = input_seq.shape[0]

        # ==========================================
        # Step 1: 前向传播 - Predictor生成预测
        # ==========================================
        predictions = self.model.predictor(input_seq)
        # predictions: [B, pred_len, C, H, W]

        # ==========================================
        # Step 2: 判别器评估
        # ==========================================
        # 对每个时间步分别计算判别器分数
        real_scores_list = []
        fake_scores_list = []

        for t in range(predictions.shape[1]):
            real_score = self.model.discriminator(true_tec[:, t])
            fake_score = self.model.discriminator(predictions[:, t].detach())
            real_scores_list.append(real_score)
            fake_scores_list.append(fake_score)

        real_scores = torch.stack(real_scores_list, dim=1).squeeze(-1)
        fake_scores_d = torch.stack(fake_scores_list, dim=1).squeeze(-1)

        # ==========================================
        # Step 3: 反向传播1 - 更新判别器
        # ==========================================
        # 论文: "The discriminator loss serves to provide the predicted
        # map with an initial outline resembling the final result"
        d_loss = self.disc_loss_fn(real_scores, fake_scores_d)

        self.optimizer_D.zero_grad()
        d_loss.backward()
        self.optimizer_D.step()

        # ==========================================
        # Step 4: 反向传播2 - 更新预测器
        # ==========================================
        # 论文: "the predictor loss focuses on fine-tuning specific details"

        # 重新计算判别器对预测的分数 (这次不detach, 要反向传播到Predictor)
        fake_scores_g_list = []
        for t in range(predictions.shape[1]):
            fake_score = self.model.discriminator(predictions[:, t])
            fake_scores_g_list.append(fake_score)
        fake_scores_g = torch.stack(fake_scores_g_list, dim=1).squeeze(-1)

        # 计算预测器损失
        g_loss, loss_dict = self.pred_loss_fn(
            pred_tec=predictions,
            true_tec=true_tec,
            disc_scores=fake_scores_g
        )

        # 训练阶段调度: 初期判别器主导, 后期预测器主导
        if self.disc_weight_schedule:
            progress = epoch / total_epochs
            # 初期: 对抗损失权重较大 (建立基线)
            # 后期: L1损失权重较大 (精修细节)
            # 这里的调度已经在loss函数的权重中体现
            # 可以进一步调整: 如初期不加对抗损失, 后期再加入
            if progress < 0.3:
                # 初期: 仅用L1损失建立基线
                g_loss, loss_dict = self.pred_loss_fn(
                    pred_tec=predictions,
                    true_tec=true_tec,
                    disc_scores=None  # 不使用对抗损失
                )

        self.optimizer_G.zero_grad()
        g_loss.backward()
        self.optimizer_G.step()

        loss_info = {
            'd_loss': d_loss.item(),
            'g_loss': loss_dict['total'],
            'l_igs': loss_dict['L_IGS'],
            'l_aux': loss_dict.get('L_AUX', 0),
            'adv_loss': loss_dict.get('adv_loss', 0),
        }

        return loss_info


# ============================================================
# 11. 使用示例
# ============================================================

def demo_usage():
    """
    使用示例: 展示GA-Predrnn的完整训练流程

    数据配置 (论文描述):
    - IGS全球TEC图: 时间分辨率2小时, 纬度5°, 经度2.5°
    - 网格: 73 × 71 = 5183个网格点
    - 辅助数据: F10.7, DST, AP (来自OMNI数据集)
    - 沿通道维度拼接: 73 × 71 × (1 + 3) 或 73 × 71 × 1
    - 输入: 前2天 (24个时间步)
    - 输出: 第3天 (12个时间步)
    """
    # ---- 模型配置 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据参数 (可根据实际数据调整)
    H, W = 73, 71          # 全球TEC图的空间分辨率
    input_channels = 4      # TEC + F10.7 + DST + AP (沿通道拼接)
    seq_len = 24            # 输入: 前2天, 2小时分辨率
    pred_len = 12           # 输出: 第3天, 2小时分辨率

    # 模型超参数
    hidden_channels = 64    # ST-LSTM隐藏通道数
    num_layers = 4          # ST-LSTM堆叠层数
    block_size = 8          # Halo Attention块大小
    halo_size = 3           # Halo光晕大小
    num_heads = 4           # 注意力头数

    # ---- 创建模型 ----
    model = GAPredrnn(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        seq_len=seq_len,
        pred_len=pred_len,
        block_size=block_size,
        halo_size=halo_size,
        num_heads=num_heads
    ).to(device)

    # ---- 创建训练器 ----
    trainer = CrossTrainer(
        model=model,
        lr_g=1e-4,           # 预测器学习率
        lr_d=1e-4,           # 判别器学习率
        alpha=10.0,          # TEC损失权重 (论文: α >> β)
        beta=1.0,            # 辅助数据损失权重
        disc_weight_schedule=True  # 训练阶段调度
    )

    # ---- 模拟数据 (实际使用时替换为真实数据) ----
    # 输入: [B, 24, 4, 73, 71] = 前2天 × 4通道 × 73×71网格
    dummy_input = torch.randn(2, seq_len, input_channels, H, W).to(device)
    # 真实TEC: [B, 12, 1, 73, 71] = 第3天 × 1通道(纯TEC) × 73×71网格
    dummy_true = torch.randn(2, pred_len, input_channels, H, W).to(device)

    # ---- 训练一步 ----
    model.train()
    loss_info = trainer.train_step(dummy_input, dummy_true, epoch=0, total_epochs=100)

    print("=" * 60)
    print("GA-Predrnn 训练一步结果:")
    print("=" * 60)
    print(f"  判别器损失 (D loss): {loss_info['d_loss']:.4f}")
    print(f"  预测器损失 (G loss): {loss_info['g_loss']:.4f}")
    print(f"  TEC L1损失 (L_IGS):  {loss_info['l_igs']:.4f}")
    print(f"  辅助数据损失 (L_AUX): {loss_info['l_aux']:.4f}")
    print(f"  对抗损失 (adv):      {loss_info['adv_loss']:.4f}")

    # ---- 模型参数统计 ----
    total_params = sum(p.numel() for p in model.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())
    discriminator_params = sum(p.numel() for p in model.discriminator.parameters())

    print(f"\n模型参数统计:")
    print(f"  总参数量:       {total_params:,.0f}")
    print(f"  预测器参数量:   {predictor_params:,.0f}")
    print(f"  判别器参数量:   {discriminator_params:,.0f}")

    # ---- 推理示例 ----
    model.eval()
    with torch.no_grad():
        predictions = model(dummy_input)
        print(f"\n推理输出形状: {predictions.shape}")
        print(f"  期望: [2, {pred_len}, {input_channels}, {H}, {W}]")


if __name__ == '__main__':
    demo_usage()