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


class STLSTMCellFused(nn.Module):
    """
    Mathematically identical to :class:`STLSTMCell`, but the 17 separate
    convolutions are packed into 4 grouped convolutions.

    Fusing is exact: a Conv2d computes each output channel independently, so
    splitting a wide convolution into channel chunks is the same arithmetic as
    running narrow convolutions side by side. Parameter count, receptive field
    and the equations are unchanged -- only the number of kernel launches and
    the effective batch size handed to cuDNN change. This matters a lot here
    because the encoder-decoder unrolls 36 timesteps x 3 layers, i.e. thousands
    of tiny convolutions per forward pass.

    Packing layout (all chunks are ``hidden_dim`` wide):

        conv_x_all  -> [z, i, f, z_m, i_m, f_m, o]        (7 * D)
        conv_h_all  -> [z, i, f, o]                       (4 * D)  from h_prev
        conv_m_all  -> [z_m, i_m, f_m]                    (3 * D)  from m_prev
        conv_cm_all -> [o_c, fusion]                      (2 * D)  from [c_t, m_t]
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        d = hidden_dim

        self.conv_x_all = nn.Conv2d(input_dim, 7 * d, kernel_size, padding=pad)
        self.conv_h_all = nn.Conv2d(d, 4 * d, kernel_size, padding=pad)
        self.conv_m_all = nn.Conv2d(d, 3 * d, kernel_size, padding=pad)
        self.conv_cm_all = nn.Conv2d(2 * d, 2 * d, kernel_size=1)

    def forward(self, x, h_prev, c_prev, m_prev):
        d = self.hidden_dim

        zx, ix, fx, zmx, imx, fmx, ox = self.conv_x_all(x).split(
            (d, d, d, d, d, d, d), dim=1
        )
        zh, ih, fh, oh = self.conv_h_all(h_prev).split(
            (d, d, d, d), dim=1
        )
        zmm, imm, fmm = self.conv_m_all(m_prev).split((d, d, d), dim=1)

        z = torch.tanh(zx + zh)
        i = torch.sigmoid(ix + ih)
        f = torch.sigmoid(fx + fh)
        c_t = f * c_prev + i * z

        z_m = torch.tanh(zmx + zmm)
        i_m = torch.sigmoid(imx + imm)
        f_m = torch.sigmoid(fmx + fmm)
        m_t = f_m * m_prev + i_m * z_m

        o_c, fusion = self.conv_cm_all(torch.cat([c_t, m_t], dim=1)).split(
            (d, d), dim=1
        )
        o = torch.sigmoid(ox + oh + o_c)
        h_t = o * torch.tanh(fusion)

        return h_t, c_t, m_t


# Order used when packing the reference cell's weights into the fused cell.
_X_PACK = ("conv_xz", "conv_xi", "conv_xf", "conv_xzm", "conv_xim", "conv_xfm", "conv_xo")
_H_PACK = ("conv_hz", "conv_hi", "conv_hf", "conv_ho")
_M_PACK = ("conv_hzm", "conv_him", "conv_hfm")


def fuse_st_lstm_weights(state_dict, hidden_dim=None):
    """
    Convert a state_dict produced by :class:`STLSTMCell` into one loadable by
    :class:`STLSTMCellFused`. Keys keep their prefix, e.g.

        ``encoder_cells.0.conv_hz.weight`` -> ``encoder_cells.0.conv_h_all.weight``

    ``conv_cm_all`` packs ``conv_co`` (c-part) and ``conv_mo`` (m-part) into the
    first ``hidden_dim`` output rows and ``conv_1x1`` into the remaining rows,
    which reproduces the reference computation exactly.
    """
    out = {}
    # group reference keys by (prefix, attribute)
    by_prefix = {}
    packed_attrs = _X_PACK + _H_PACK + _M_PACK + ("conv_co", "conv_mo", "conv_1x1")
    for key, tensor in state_dict.items():
        for attr in packed_attrs:
            # support both module-level keys ("conv_xz.weight") and nested
            # keys ("encoder_cells.0.conv_xz.weight")
            if key.startswith(f"{attr}."):
                prefix, tail = "", key[len(attr) + 1:]
            elif f".{attr}." in key:
                prefix, tail = key.split(f".{attr}.", 1)
            else:
                continue
            by_prefix.setdefault(prefix, {})[f"{attr}.{tail}"] = tensor
            break
        else:
            out[key] = tensor

    for prefix, parts in by_prefix.items():
        d = hidden_dim
        if d is None:
            d = parts["conv_hz.weight"].shape[0]

        def key(name):
            return f"{prefix}.{name}" if prefix else name

        out[key("conv_x_all.weight")] = torch.cat(
            [parts[f"{a}.weight"] for a in _X_PACK], dim=0
        )
        out[key("conv_x_all.bias")] = torch.cat(
            [parts[f"{a}.bias"] for a in _X_PACK], dim=0
        )
        out[key("conv_h_all.weight")] = torch.cat(
            [parts[f"{a}.weight"] for a in _H_PACK], dim=0
        )
        out[key("conv_h_all.bias")] = torch.cat(
            [parts[f"{a}.bias"] for a in _H_PACK], dim=0
        )
        out[key("conv_m_all.weight")] = torch.cat(
            [parts[f"{a}.weight"] for a in _M_PACK], dim=0
        )
        out[key("conv_m_all.bias")] = torch.cat(
            [parts[f"{a}.bias"] for a in _M_PACK], dim=0
        )

        cm_w = parts["conv_co.weight"].new_zeros(2 * d, 2 * d, 1, 1)
        cm_w[:d, :d] = parts["conv_co.weight"]
        cm_w[:d, d:] = parts["conv_mo.weight"]
        cm_w[d:, :] = parts["conv_1x1.weight"]
        out[key("conv_cm_all.weight")] = cm_w

        # rows[0:D] carry conv_co(c_t) + conv_mo(m_t), so their biases add
        o_bias = parts["conv_co.bias"] + parts["conv_mo.bias"]
        out[key("conv_cm_all.bias")] = torch.cat(
            [o_bias, parts["conv_1x1.bias"]], dim=0
        )

    return out
