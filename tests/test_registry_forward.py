import copy
import unittest

import torch

from core.config import load_config
from core.registry import build_model


def _small_config(model_name):
    """Start from the real model YAML and only shrink the knobs a test names."""
    config = copy.deepcopy(load_config(model_name))
    config["data"]["batch_size"] = 1
    config["data"]["input_length"] = 6
    config["data"]["output_length"] = 2
    return config


class RegistryForwardTests(unittest.TestCase):
    def _assert_forward(self, model_name, model_params, input_length=6):
        config = _small_config(model_name)
        config["data"]["input_length"] = input_length
        config["model"]["params"].update(model_params)
        model = build_model(config)
        tec = torch.randn(1, input_length, 71, 73)
        aux = torch.randn(1, input_length, 6)
        with torch.no_grad():
            output = model(tec, aux)
        self.assertEqual(tuple(output.shape), (1, 2, 71, 73))
        return model

    def test_epd_convlstm_forward(self):
        self._assert_forward(
            "E_P_D",
            {
                "predictor_name": "convlstm",
                "transmit_parameter": 1,
            },
        )

    def test_epd_alternate_predictors_forward(self):
        for predictor in ("transformer", "tcn", "convgru"):
            with self.subTest(predictor=predictor):
                self._assert_forward(
                    "E_P_D",
                    {
                        "predictor_name": predictor,
                        "transmit_parameter": 1,
                    },
                )

    def test_ed_cg_conv_lstm_forward(self):
        self._assert_forward(
            "ED_CGConvLSTM",
            {
                "hidden_dim": 4,
                "num_layers": 1,
                "kernel_size": 3,
            },
        )

    def test_ga_predrnn_forward(self):
        self._assert_forward(
            "GA_Predrnn",
            {
                "hidden_dim": 8,
                "num_layers": 1,
                "kernel_size": 3,
                "block_size": 4,
                "halo_size": 1,
                "num_heads": 2,
            },
        )

    def test_ed_autoformer_forward(self):
        self._assert_forward(
            "ED_Autoformer",
            {
                "d_model": 16,
                "n_heads": 2,
                "d_ff": 32,
                "e_layers": 1,
                "d_layers": 1,
                "moving_avg": 3,
                "encode_channels": [8, 16],
            },
        )

    def test_modelcanon_forward(self):
        self._assert_forward(
            "ModelCanon",
            {
                "d_model": 32,
                "n_heads": 4,
                "e_layers": 1,
                "decoder_layers": 1,
                "d_ff": 64,
                "patch_size": 4,
            },
        )


if __name__ == "__main__":
    unittest.main()
