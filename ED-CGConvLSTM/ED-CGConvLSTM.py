import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordGate(nn.Module):
    """空间感知卷积模块，保持空间分辨率不变"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2   # 保持输出尺寸与输入相同
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.coord_encoder = nn.Conv2d(2, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C_in, H, W)
        B, _, H, W = x.shape
        F_map = self.conv(x)  # (B, C_out, H, W) 由于padding设置，尺寸不变

        # 生成归一化坐标图 (0~1)
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        )
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        G = self.coord_encoder(coord)              # (B, C_out, H, W)
        G = torch.sigmoid(G)
        return F_map * G


class CGConvLSTMCell(nn.Module):
    """CGConvLSTM单元，所有卷积替换为CoordGate"""
    def __init__(self, input_dim, hidden_dim, kernel_size, padding=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        if padding is None:
            padding = kernel_size // 2
        # 输入门
        self.W_xi = CoordGate(input_dim, hidden_dim, kernel_size, padding=padding)
        self.W_hi = CoordGate(hidden_dim, hidden_dim, kernel_size, padding=padding)
        # 遗忘门
        self.W_xf = CoordGate(input_dim, hidden_dim, kernel_size, padding=padding)
        self.W_hf = CoordGate(hidden_dim, hidden_dim, kernel_size, padding=padding)
        # 候选记忆
        self.W_xc = CoordGate(input_dim, hidden_dim, kernel_size, padding=padding)
        self.W_hc = CoordGate(hidden_dim, hidden_dim, kernel_size, padding=padding)
        # 输出门
        self.W_xo = CoordGate(input_dim, hidden_dim, kernel_size, padding=padding)
        self.W_ho = CoordGate(hidden_dim, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        # x, h_prev, c_prev: (B, C, H, W)
        i = torch.sigmoid(self.W_xi(x) + self.W_hi(h_prev))
        f = torch.sigmoid(self.W_xf(x) + self.W_hf(h_prev))
        c_tilde = torch.tanh(self.W_xc(x) + self.W_hc(h_prev))
        c = f * c_prev + i * c_tilde
        o = torch.sigmoid(self.W_xo(x) + self.W_ho(h_prev))
        h = o * torch.tanh(c)
        return h, c


class CGConvLSTM(nn.Module):
    """多层CGConvLSTM，处理完整序列"""
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, padding=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        if padding is None:
            padding = kernel_size // 2
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(CGConvLSTMCell(in_dim, hidden_dim, kernel_size, padding))

    def forward(self, x, states=None):
        # x: (B, T, C_in, H, W)
        B, T, C, H, W = x.shape
        if states is None:
            h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = states

        outputs = []
        for t in range(T):
            x_t = x[:, t, ...]          # (B, C, H, W)
            layer_outputs = []          # 当前时间步各层输出，用于层间传递
            for l in range(self.num_layers):
                inp = x_t if l == 0 else layer_outputs[-1]   # 同一时间步上一层的输出
                h[l], c[l] = self.cells[l](inp, h[l], c[l])
                layer_outputs.append(h[l])
            outputs.append(h[-1].unsqueeze(1))   # 最后一层输出作为该时间步最终输出

        outputs = torch.cat(outputs, dim=1)      # (B, T, hidden_dim, H, W)
        return outputs, (h, c)


class ED_CGConvLSTM(nn.Module):
    """编码器-解码器CGConvLSTM预测模型"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = CGConvLSTM(input_dim, hidden_dim, num_layers, kernel_size)
        self.decoder = CGConvLSTM(hidden_dim, hidden_dim, num_layers, kernel_size)
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.map_to_hidden = nn.Conv2d(output_dim, hidden_dim, kernel_size=1)

    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        # x: (B, T, C_in, H, W)
        B, T, C, H, W = x.shape
        # 编码
        _, (h_enc, c_enc) = self.encoder(x)

        # 解码器初始状态：取编码器各层的最终状态
        h_dec = [h_enc[l] for l in range(self.num_layers)]
        c_dec = [c_enc[l] for l in range(self.num_layers)]

        # 初始输入（全零，尺寸为隐藏维度）
        dec_input = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        outputs = []

        for t in range(T):
            # 教师强制：若提供target且随机概率小于阈值，则使用目标值
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # target[:, t, ...] 形状 (B, output_dim, H, W)
                dec_input = self.map_to_hidden(target[:, t, ...])

            # 单步解码（输入需增加时间维）
            dec_input_seq = dec_input.unsqueeze(1)   # (B, 1, hidden_dim, H, W)
            _, (h_dec, c_dec) = self.decoder(dec_input_seq, (h_dec, c_dec))
            h_out = h_dec[-1]                        # 最后一层隐藏状态
            pred = self.conv_out(h_out)              # (B, output_dim, H, W)
            outputs.append(pred.unsqueeze(1))

            # 更新下一时刻的输入：将当前预测映射回隐空间
            dec_input = self.map_to_hidden(pred)

        outputs = torch.cat(outputs, dim=1)          # (B, T, output_dim, H, W)
        return outputs


# ========== 使用示例 ==========
if __name__ == "__main__":
    batch_size = 4
    seq_len = 12
    input_dim = 1
    hidden_dim = 60      # 论文贝叶斯优化结果
    output_dim = 1
    num_layers = 4
    kernel_size = 3
    H, W = 71, 73

    model = ED_CGConvLSTM(input_dim, hidden_dim, output_dim, num_layers, kernel_size, seq_len)
    x = torch.randn(batch_size, seq_len, input_dim, H, W)
    pred = model(x)      # 测试前向传播
    print("预测输出形状:", pred.shape)   # 应为 (4, 12, 1, 71, 73)