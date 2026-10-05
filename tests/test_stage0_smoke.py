import unittest

import torch

from E_P_D.PredictionModel.Gru.convGRU import ConvGRU
from E_P_D.PredictionModel.LSTM.convLSTM.convLSTM import ConvLSTM
from GA_Predrnn.spatiotemporal_attention import HaloAttention


class Stage0SmokeTests(unittest.TestCase):
    def test_halo_attention_supports_tec_grid(self):
        model = HaloAttention(dim=8, block_size=8, halo_size=2, num_heads=2)
        x = torch.randn(2, 8, 71, 73)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))

    def test_conv_lstm_honors_hidden_channels(self):
        model = ConvLSTM(
            history_len=3,
            in_channels=5,
            hidden_channels=7,
            predict_len=2,
        )
        x = torch.randn(2, 3, 5, 8, 8)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 7, 8, 8))

    def test_conv_gru_imports_and_runs(self):
        model = ConvGRU(
            in_channels=5,
            hidden_channels=7,
            history_len=3,
            predict_len=2,
            gru_layers=1,
        )
        x = torch.randn(2, 3, 5, 8, 8)
        with torch.no_grad():
            y = model(x)
        self.assertEqual(tuple(y.shape), (2, 2, 5, 8, 8))


if __name__ == "__main__":
    unittest.main()
