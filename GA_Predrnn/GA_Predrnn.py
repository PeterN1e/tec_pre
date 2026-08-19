"""
GA-Predrnn Predictor
(Wang et al. Predrnn backbone + spatio-temporal Halo attention + dual output heads)

Public API
----------
GAPredrnnPredictor(tec, aux)
    tec : (B, T_in, H, W)   normalised TEC maps
    aux : (B, T_in, 6)      normalised physical indices
        indices [2,3,4] = dst, ap, f10.7 (broadcast to spatial grid)

Returns (for model_selector compatibility)
    (B, T_out, H, W)        predicted TEC maps

train_forward(tec, aux)
    returns (pred_tec, pred_aux)
        pred_tec : (B, T_out, H, W)
        pred_aux : (B, T_out, 3)         predicted [dst, ap, f10.7]
"""

import torch
import torch.nn as nn

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from GA_Predrnn.st_lstm_cell import STLSTMCell
from GA_Predrnn.spatiotemporal_attention import SpatioTemporalAttention


class GAPredrnnPredictor(nn.Module):
    """
    Encoder-decoder Predrnn with ST-Attention bridge and dual output heads.

    Args:
        input_dim:      channels of the fused input (TEC + 3 aux = 4)
        hidden_dim:     ST-LSTM hidden channels
        num_layers:     stacked ST-LSTM depth (default 3, matching paper)
        kernel_size:    ST-LSTM conv kernel (default 5)
        input_length:   number of input timesteps  (default 24 = 2 days x 12)
        output_length:  number of output timesteps  (default 12 = 1 day  x 12)
        aux_dim:        number of auxiliary channels to predict (default 3)
        block_size:     Halo attention block size (default 8)
        halo_size:      Halo attention halo size  (default 2)
        num_heads:      Halo attention heads       (default 4)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 5,
        input_length: int = 24,
        output_length: int = 12,
        aux_dim: int = 3,
        block_size: int = 8,
        halo_size: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_length = input_length
        self.output_length = output_length

        # --- encoder ST-LSTM stack ---
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.encoder_cells.append(STLSTMCell(in_dim, hidden_dim, kernel_size))

        # --- decoder ST-LSTM stack ---
        self.decoder_cells = nn.ModuleList()
        for _ in range(num_layers):
            self.decoder_cells.append(STLSTMCell(hidden_dim, hidden_dim, kernel_size))

        # --- ST-Attention bridge ---
        self.attention = SpatioTemporalAttention(
            hidden_dim, block_size, halo_size, num_heads
        )

        # --- output heads ---
        self.tec_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # aux head: GAP over spatial dims -> FC
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, aux_dim),
        )

        # projection used inside decoder autoregressive loop
        self.pred_to_hidden = nn.Conv2d(1, hidden_dim, kernel_size=1)

    # ------------------------------------------------------------------ #
    #  internal: encoder-decoder forward
    # ------------------------------------------------------------------ #
    def _forward_core(self, x):
        """
        x: (B, T_in, C, H, W)   fused 4-channel input
        returns:
            tec_out : (B, T_out, H, W)
            aux_out : (B, T_out, 3)
        """
        B, T_in, C, H, W = x.shape
        D = self.hidden_dim
        L = self.num_layers
        T_out = self.output_length

        # ---- zero-init states ----
        h = [x.new_zeros(B, D, H, W) for _ in range(L)]
        c = [x.new_zeros(B, D, H, W) for _ in range(L)]
        m = [x.new_zeros(B, D, H, W) for _ in range(L)]

        # ---- ENCODER ----
        for t in range(T_in):
            x_t = x[:, t]                               # (B, C, H, W)
            h_new_layers = []
            for l in range(L):
                inp = x_t if l == 0 else h_new_layers[-1]
                h_prev = h[l]
                c_prev = c[l]
                # M zigzag: layer-0 reads top-layer M from prev timestep
                m_prev = m[l - 1] if l > 0 else m[L - 1]
                h_t, c_t, m_t = self.encoder_cells[l](inp, h_prev, c_prev, m_prev)
                h[l] = h_t
                c[l] = c_t
                m[l] = m_t
                h_new_layers.append(h_t)

        # encoder hidden state at the last timestep, last layer
        enc_hidden = h[L - 1]                            # (B, D, H, W)

        # ---- DECODER (autoregressive) ----
        dec_input = x.new_zeros(B, D, H, W)             # initial decoder input

        tec_preds = []
        aux_preds = []

        for t in range(T_out):
            # apply ST-Attention at each decoder step
            dec_input = self.attention(enc_hidden, dec_input)

            h_new_layers = []
            for l in range(L):
                inp = dec_input if l == 0 else h_new_layers[-1]
                h_prev = h[l]
                c_prev = c[l]
                m_prev = m[l - 1] if l > 0 else m[L - 1]
                h_t, c_t, m_t = self.decoder_cells[l](inp, h_prev, c_prev, m_prev)
                h[l] = h_t
                c[l] = c_t
                m[l] = m_t
                h_new_layers.append(h_t)

            # predict TEC
            p_tec = self.tec_head(h[L - 1])              # (B, 1, H, W)
            tec_preds.append(p_tec)

            # predict aux
            p_aux = self.aux_head(h[L - 1])              # (B, 3)
            aux_preds.append(p_aux)

            # autoregressive: map prediction back to hidden space
            dec_input = self.pred_to_hidden(p_tec)        # (B, D, H, W)

        tec_out = torch.cat(tec_preds, dim=1).squeeze(2)  # (B, T_out, H, W)
        aux_out = torch.stack(aux_preds, dim=1)            # (B, T_out, 3)
        return tec_out, aux_out

    # ------------------------------------------------------------------ #
    #  build 4-channel fused input from dataset tensors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fuse_input(tec, aux):
        """
        tec : (B, T, H, W)
        aux : (B, T, 6)        columns [2,3,4] = dst, ap, f10.7
        returns: (B, T, 4, H, W)
        """
        # take dst, ap, f10.7 and broadcast to spatial grid
        aux3 = aux[:, :, [2, 3, 4]]                                     # (B, T, 3)
        H, W = tec.shape[2], tec.shape[3]
        aux_spatial = aux3.unsqueeze(-1).unsqueeze(-1)                   # (B, T, 3, 1, 1)
        aux_spatial = aux_spatial.expand(-1, -1, -1, H, W)              # (B, T, 3, H, W)
        tec_4d = tec.unsqueeze(2)                                       # (B, T, 1, H, W)
        return torch.cat([tec_4d, aux_spatial], dim=2)                  # (B, T, 4, H, W)

    # ------------------------------------------------------------------ #
    #  public forward — compatible with model_selector.py
    # ------------------------------------------------------------------ #
    def forward(self, tec, aux):
        """
        Args:
            tec : (B, T_in, H, W)   normalised TEC sequence
            aux : (B, T_in, 6)      normalised physical indices
        Returns:
            (B, T_out, H, W)        predicted TEC sequence
        """
        x = self._fuse_input(tec, aux)
        tec_out, _ = self._forward_core(x)
        return tec_out

    # ------------------------------------------------------------------ #
    #  GAN training forward — returns both TEC and aux predictions
    # ------------------------------------------------------------------ #
    def train_forward(self, tec, aux):
        """
        Same inputs as forward(), but returns both heads.
        Returns:
            pred_tec : (B, T_out, H, W)
            pred_aux : (B, T_out, 3)
        """
        x = self._fuse_input(tec, aux)
        return self._forward_core(x)


# ------------------------------------------------------------------ #
#  quick shape smoke test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_in, T_out, H, W = 2, 24, 12, 71, 73
    model = GAPredrnnPredictor(
        input_dim=4, hidden_dim=32, num_layers=2,
        kernel_size=3, input_length=T_in, output_length=T_out,
    )
    tec = torch.randn(B, T_in, H, W)
    aux = torch.randn(B, T_in, 6)
    with torch.no_grad():
        out = model(tec, aux)
        t_out, a_out = model.train_forward(tec, aux)
    print(f"forward      -> {out.shape}")
    print(f"train_forward-> tec {t_out.shape}, aux {a_out.shape}")
    print(f"params       -> {sum(p.numel() for p in model.parameters()):,}")
