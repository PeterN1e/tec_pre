import torch
import torch.nn as nn
from config import EDCGConvLSTMConfig, TrainConfig

cfg_model = EDCGConvLSTMConfig()
cfg_train = TrainConfig()


# ========== 坐标网格工具函数（全局只算一次）==========
def _build_coord_grid(H, W, device):
    """生成归一化坐标网格 (2, H, W)，由外部按需调用，避免每个 CoordGate 重复创建"""
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    return torch.stack([xx, yy], dim=0)   # (2, H, W)


# ========== CoordGate ==========
class CoordGate(nn.Module):
    """空间感知卷积模块，保持空间分辨率不变"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.coord_encoder = nn.Conv2d(2, out_channels, kernel_size=1)

    def forward(self, x, coord_grid):
        """
        x:          (B, C_in, H, W)
        coord_grid: (2, H, W) —— 由外部统一提供，全局只算一次
        """
        B = x.shape[0]
        F_map = self.conv(x)
        G = self.coord_encoder(coord_grid.unsqueeze(0).expand(B, -1, -1, -1))
        G = torch.sigmoid(G)
        return F_map * G


# ========== CGConvLSTMCell ==========
class CGConvLSTMCell(nn.Module):
    """CGConvLSTM 单元，所有卷积替换为 CoordGate"""
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

    def forward(self, x, h_prev, c_prev, coord_grid):
        i  = torch.sigmoid(self.W_xi(x, coord_grid) + self.W_hi(h_prev, coord_grid))
        f  = torch.sigmoid(self.W_xf(x, coord_grid) + self.W_hf(h_prev, coord_grid))
        c_ = torch.tanh(   self.W_xc(x, coord_grid) + self.W_hc(h_prev, coord_grid))
        c  = f * c_prev + i * c_
        o  = torch.sigmoid(self.W_xo(x, coord_grid) + self.W_ho(h_prev, coord_grid))
        h  = o * torch.tanh(c)
        return h, c


# ========== CGConvLSTM（多层，处理完整序列）==========
class CGConvLSTM(nn.Module):
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

        # ★ 只计算一次坐标网格
        coord_grid = _build_coord_grid(H, W, x.device)

        if states is None:
            h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = states

        outputs = []
        for t in range(T):
            x_t = x[:, t, ...]
            layer_outputs = []
            for l in range(self.num_layers):
                inp = x_t if l == 0 else layer_outputs[-1]
                h[l], c[l] = self.cells[l](inp, h[l], c[l], coord_grid)
                layer_outputs.append(h[l])
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c)


# ========== EDCGConvLSTM ==========
class EDCGConvLSTM(nn.Module):
    """编码器-解码器 CGConvLSTM"""
    def __init__(self, input_dim=cfg_model.input_dim,
                 hidden_dim=cfg_model.hidden_dim,
                 output_dim=cfg_model.output_dim,
                 num_layers=cfg_model.num_layers,
                 kernel_size=cfg_model.kernel_size,       # ★ 修复：原来是 cfg_model.num_layers
                 input_length=cfg_train.input_length,
                 output_length=cfg_train.output_length):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.encoder = CGConvLSTM(input_dim, hidden_dim, num_layers, kernel_size)
        self.decoder = CGConvLSTM(hidden_dim, hidden_dim, num_layers, kernel_size)
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.map_to_hidden = nn.Conv2d(output_dim, hidden_dim, kernel_size=1)
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        # x: (B, T_in, C_in, H, W)
        B, T_in, C, H, W = x.shape

        # 编码
        _, (h_enc, c_enc) = self.encoder(x)

        # 解码器初始状态
        h_dec = [h_enc[l].clone() for l in range(self.num_layers)]
        c_dec = [c_enc[l].clone() for l in range(self.num_layers)]

        # 初始解码输入
        dec_input = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        outputs = []

        for t in range(self.output_length):
            # 教师强制
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = self.map_to_hidden(target[:, t, ...])

            # 单步解码
            dec_input_seq = dec_input.unsqueeze(1)
            _, (h_dec, c_dec) = self.decoder(dec_input_seq, (h_dec, c_dec))
            h_out = h_dec[-1]
            pred = self.conv_out(h_out)
            outputs.append(pred.unsqueeze(1))

            # 自回归
            dec_input = self.map_to_hidden(pred)

        outputs = torch.cat(outputs, dim=1)
        return outputs


# ========== 测试 ==========
if __name__ == "__main__":
    batch_size = 48
    seq_len = 36
    input_dim = 1
    hidden_dim = 60
    output_dim = 1
    num_layers = 4
    kernel_size = 3
    H, W = 71, 73

    model = EDCGConvLSTM().to(cfg_train.device)
    x = torch.randn(batch_size, seq_len, input_dim, H, W,device=cfg_train.device)

    # ★ 推理测试时关闭 autograd，内存从 ~20GB 降到几百 MB
    with torch.no_grad():
        pred = model(x)

    print("预测输出形状:", pred.shape)   # (48, 12, 1, 71, 73)