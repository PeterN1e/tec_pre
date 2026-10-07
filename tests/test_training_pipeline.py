import logging
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_config
from core.inference import predict_split
from core.trainer import run_training


class TrainingPipelineTests(unittest.TestCase):
    def tearDown(self):
        logging.shutdown()

    def _write_year(self, tec_dir: Path, indices_dir: Path, year: int, days=2):
        rng = np.random.default_rng(year)
        tec = rng.normal(size=(days, 12, 4, 5)).astype(np.float32)
        np.save(tec_dir / f"{year}_igsg.npy", tec)
        rows = []
        for doy in range(1, days + 1):
            for hour in range(0, 24, 2):
                rows.append(
                    {
                        "year": year,
                        "doy": doy,
                        "hour": hour,
                        "kp": hour / 10,
                        "ssn": hour,
                        "dst": hour - 10,
                        "ap": hour + 2,
                        "f10.7": hour + 70,
                        "ae": hour + 20,
                    }
                )
        pd.DataFrame(rows).to_csv(
            indices_dir / f"{year}_indices.csv",
            index=False,
        )

    def test_one_epoch_and_prediction(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(logging.shutdown)
        root = Path(tmp)
        tec_dir = root / "tec"
        indices_dir = root / "indices"
        tec_dir.mkdir()
        indices_dir.mkdir()
        for year in (2020, 2021, 2022):
            self._write_year(tec_dir, indices_dir, year)

        config = load_config("ModelCanon")
        config["seed"] = 7
        config["device"] = "cpu"
        config["output_dir"] = str(root / "save")
        config["data"].update(
            {
                "data_root": str(root),
                "tec_subdir": "tec",
                "indices_subdir": "indices",
                "height": 4,
                "width": 5,
                # ModelCanon's three-day difference baseline needs
                # input_length >= 3 * output_length.
                "input_length": 12,
                "output_length": 4,
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "splits": {
                    "train": [202001, 202001],
                    "val": [202101, 202101],
                    "test": [202201, 202201],
                },
            }
        )
        config["model"]["params"] = {
            "d_model": 16,
            "n_heads": 4,
            "e_layers": 1,
            "decoder_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "patch_size": 4,
        }
        config["training"].update(
            {
                "epochs": 1,
                "patience": 1,
                "lr": 0.001,
                "use_amp": False,
                "scheduler": None,
            }
        )

        result = run_training(config)
        self.assertTrue(Path(result["checkpoint"]).exists())

        evaluation = predict_split(config, split="test")
        self.assertIn("aggregate", evaluation["metrics"])
        self.assertTrue(Path(evaluation["predictions_path"]).exists())
        self.assertTrue(Path(evaluation["metrics_path"]).exists())

    def test_one_epoch_gan_pipeline(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.addCleanup(logging.shutdown)
        root = Path(tmp)
        tec_dir = root / "tec"
        indices_dir = root / "indices"
        tec_dir.mkdir()
        indices_dir.mkdir()
        for year in (2020, 2021):
            self._write_year(tec_dir, indices_dir, year)

        config = load_config("GA_Predrnn")
        config["seed"] = 11
        config["device"] = "cpu"
        config["output_dir"] = str(root / "save")
        config["data"].update(
            {
                "data_root": str(root),
                "tec_subdir": "tec",
                "indices_subdir": "indices",
                "height": 4,
                "width": 5,
                "input_length": 6,
                "output_length": 2,
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "splits": {
                    "train": [202001, 202001],
                    "val": [202101, 202101],
                },
            }
        )
        config["model"]["params"] = {
            "hidden_dim": 8,
            "num_layers": 1,
            "kernel_size": 3,
            "block_size": 4,
            "halo_size": 1,
            "num_heads": 2,
            "aux_indices": [2, 3, 4],
        }
        config["training"].update(
            {
                "epochs": 1,
                "patience": 1,
                "use_amp": False,
                "scheduler": None,
                "gan": {
                    "enabled": True,
                    "disc_base_ch": 4,
                    "adv_weight": 0.1,
                    "lambda_tec": 1.0,
                    "lambda_aux": 0.1,
                    "g_lr": 0.001,
                    "d_lr": 0.001,
                },
            }
        )

        result = run_training(config)
        self.assertTrue(Path(result["checkpoint"]).exists())


if __name__ == "__main__":
    unittest.main()
