import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
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


_coord_grid_cache = {}


def _get_coord_grid(H, W, device):
    """按 (H, W, device) 缓存坐标网格，避免每个 batch 重复创建"""
    key = (H, W, device)
    grid = _coord_grid_cache.get(key)
    if grid is None:
        grid = _build_coord_grid(H, W, device)
        _coord_grid_cache[key] = grid
    return grid


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
    """CGConvLSTM 单元，i/f/c/o 四门合并为输入/隐状态两个大卷积"""
    def __init__(self, input_dim, hidden_dim, kernel_size, padding=None, use_checkpoint=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        if padding is None:
            padding = kernel_size // 2
        # 输入侧和隐状态侧各做一次卷积，输出通道为 4*hidden，按门切分
        self.W_x = CoordGate(input_dim, hidden_dim * 4, kernel_size, padding=padding)
        self.W_h = CoordGate(hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def _step(self, x, h_prev, c_prev, coord_grid):
        gx = self.W_x(x, coord_grid)
        gh = self.W_h(h_prev, coord_grid)
        xi, xf, xc, xo = gx.chunk(4, dim=1)
        hi, hf, hc, ho = gh.chunk(4, dim=1)
        i  = torch.sigmoid(xi + hi)
        f  = torch.sigmoid(xf + hf)
        c_ = torch.tanh(xc + hc)
        c  = f * c_prev + i * c_
        o  = torch.sigmoid(xo + ho)
        h  = o * torch.tanh(c)
        return h, c

    def forward(self, x, h_prev, c_prev, coord_grid):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(
                self._step, x, h_prev, c_prev, coord_grid, use_reentrant=False
            )
        return self._step(x, h_prev, c_prev, coord_grid)


# ========== CGConvLSTM（多层，处理完整序列）==========
class CGConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size, padding=None,
                 use_checkpoint=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
        if padding is None:
            padding = kernel_size // 2
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(
                CGConvLSTMCell(in_dim, hidden_dim, kernel_size, padding, use_checkpoint)
            )

    def forward(self, x, states=None, return_outputs=True):
        # x: (B, T, C_in, H, W)
        B, T, C, H, W = x.shape

        coord_grid = _get_coord_grid(H, W, x.device)

        if states is None:
            h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
            c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) for _ in range(self.num_layers)]
        else:
            h, c = states

        outputs = [] if return_outputs else None
        for t in range(T):
            x_t = x[:, t, ...]
            layer_outputs = []
            for l in range(self.num_layers):
                inp = x_t if l == 0 else layer_outputs[-1]
                h[l], c[l] = self.cells[l](inp, h[l], c[l], coord_grid)
                layer_outputs.append(h[l])
            if return_outputs:
                outputs.append(h[-1].unsqueeze(1))

        if return_outputs:
            outputs = torch.cat(outputs, dim=1)
            return outputs, (h, c)
        return None, (h, c)


# ========== EDCGConvLSTM ==========
class EDCGConvLSTM(nn.Module):
    """编码器-解码器 CGConvLSTM"""
    def __init__(self, input_dim=cfg_model.input_dim,
                 hidden_dim=cfg_model.hidden_dim,
                 output_dim=cfg_model.output_dim,
                 num_layers=cfg_model.num_layers,
                 kernel_size=cfg_model.kernel_size,       # ★ 修复：原来是 cfg_model.num_layers
                 input_length=cfg_train.input_length,
                 output_length=cfg_train.output_length,
                 use_checkpoint=None,
                 use_torch_compile=None):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_checkpoint = cfg_model.use_checkpoint if use_checkpoint is None else use_checkpoint
        self.use_torch_compile = cfg_model.use_torch_compile if use_torch_compile is None else use_torch_compile
        self._compiled = None

        self.encoder = CGConvLSTM(input_dim, hidden_dim, num_layers, kernel_size,
                                  use_checkpoint=self.use_checkpoint)
        self.decoder = CGConvLSTM(hidden_dim, hidden_dim, num_layers, kernel_size,
                                  use_checkpoint=self.use_checkpoint)
        self.conv_out = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.map_to_hidden = nn.Conv2d(output_dim, hidden_dim, kernel_size=1)

    def _forward_impl(self, x, target=None, teacher_forcing_ratio=0.0):
        # x: (B, T_in, C_in, H, W)
        B, T_in, C, H, W = x.shape

        # 编码（训练时不需要保留 36 步输出，只保留末状态，避免白占显存）
        _, (h_enc, c_enc) = self.encoder(x, return_outputs=False)

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
            _, (h_dec, c_dec) = self.decoder(dec_input_seq, (h_dec, c_dec), return_outputs=False)
            h_out = h_dec[-1]
            pred = self.conv_out(h_out)
            outputs.append(pred.unsqueeze(1))

            # 自回归
            dec_input = self.map_to_hidden(pred)

        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        if self.use_torch_compile:
            if self._compiled is None:
                try:
                    self._compiled = torch.compile(self._forward_impl)
                except Exception:
                    self._compiled = False
            if self._compiled is not False:
                try:
                    return self._compiled(x, target, teacher_forcing_ratio)
                except Exception:
                    # 编译环境不可用时自动回退到普通前向，不影响训练
                    self.use_torch_compile = False
                    self._compiled = None
        return self._forward_impl(x, target, teacher_forcing_ratio)


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
