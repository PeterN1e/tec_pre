import torch
import torch.nn as nn


class STLSTMCell(nn.Module):
    """
    Spatio-Temporal LSTM Cell (Predrnn, Wang et al. 2017)
    Dual memory: C (horizontal temporal) + M (vertical spatiotemporal)

    Args:
        input_dim:  number of input channels
        hidden_dim: number of hidden channels
        kernel_size: convolution kernel size (default 5)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2

        # ---- C gates (temporal memory) ----
        self.conv_xz = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_hz = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_xi = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_hi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_xf = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_hf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)

        # ---- M gates (spatiotemporal memory) ----
        self.conv_xzm = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_hzm = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_xim = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_him = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_xfm = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_hfm = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)

        # ---- output gate ----
        self.conv_xo = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_ho = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad)
        self.conv_co = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.conv_mo = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

        # ---- fusion of C and M ----
        self.conv_1x1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    def forward(self, x, h_prev, c_prev, m_prev):
        """
        Args:
            x:      (B, C_in, H, W)  current input to this layer
            h_prev: (B, D,    H, W)  hidden state from same layer, previous timestep
            c_prev: (B, D,    H, W)  cell state  from same layer, previous timestep
            m_prev: (B, D,    H, W)  M from lower layer, previous timestep
                    (for layer-0, this is M from the top layer at t-1, i.e. zigzag)
        Returns:
            h_t, c_t, m_t  each (B, D, H, W)
        """
        # C (temporal memory)
        z = torch.tanh(self.conv_xz(x) + self.conv_hz(h_prev))
        i = torch.sigmoid(self.conv_xi(x) + self.conv_hi(h_prev))
        f = torch.sigmoid(self.conv_xf(x) + self.conv_hf(h_prev))
        c_t = f * c_prev + i * z

        # M (spatiotemporal memory)
        z_m = torch.tanh(self.conv_xzm(x) + self.conv_hzm(m_prev))
        i_m = torch.sigmoid(self.conv_xim(x) + self.conv_him(m_prev))
        f_m = torch.sigmoid(self.conv_xfm(x) + self.conv_hfm(m_prev))
        m_t = f_m * m_prev + i_m * z_m

        # output
        o = torch.sigmoid(
            self.conv_xo(x) + self.conv_ho(h_prev)
            + self.conv_co(c_t) + self.conv_mo(m_t)
        )
        h_t = o * torch.tanh(self.conv_1x1(torch.cat([c_t, m_t], dim=1)))

        return h_t, c_t, m_t
