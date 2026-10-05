import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.tec_dataset import TecIonosphereDataset


class TecDatasetTests(unittest.TestCase):
    def _write_year(
        self,
        tec_dir: Path,
        indices_dir: Path,
        year: int,
        days: int,
    ):
        tec = np.arange(days * 12 * 4 * 5, dtype=np.float32).reshape(
            days, 12, 4, 5
        )
        np.save(tec_dir / f"{year}_igsg.npy", tec)
        rows = []
        for doy in range(1, days + 1):
            for hour in range(24):
                if hour % 2:
                    continue
                rows.append(
                    {
                        "year": year,
                        "doy": doy,
                        "hour": hour,
                        "kp": hour,
                        "ssn": hour + 1,
                        "dst": hour + 2,
                        "ap": hour + 3,
                        "f10.7": hour + 4,
                        "ae": hour + 5,
                    }
                )
        pd.DataFrame(rows).to_csv(
            indices_dir / f"{year}_indices.csv",
            index=False,
        )

    def test_window_shape_and_cross_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tec_dir = root / "tec"
            indices_dir = root / "indices"
            tec_dir.mkdir()
            indices_dir.mkdir()
            self._write_year(tec_dir, indices_dir, 2020, 1)
            self._write_year(tec_dir, indices_dir, 2021, 1)

            tec_scaler = StandardScaler()
            aux_scaler = StandardScaler()
            dataset = TecIonosphereDataset(
                tec_dir=tec_dir,
                indices_dir=indices_dir,
                start_month=202001,
                end_month=202101,
                input_length=6,
                output_length=6,
                is_train=True,
                tec_scaler=tec_scaler,
                aux_scaler=aux_scaler,
            )

            self.assertEqual(len(dataset), 13)
            tec_in, aux_in, tec_gt, aux_gt = dataset[6]
            self.assertEqual(tuple(tec_in.shape), (6, 4, 5))
            self.assertEqual(tuple(aux_in.shape), (6, 6))
            self.assertEqual(tuple(tec_gt.shape), (6, 4, 5))
            self.assertEqual(tuple(aux_gt.shape), (6, 6))
            self.assertTrue(np.isfinite(tec_in.numpy()).all())


if __name__ == "__main__":
    unittest.main()
