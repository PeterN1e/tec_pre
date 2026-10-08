import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.tec_dataset import TecIonosphereDataset, normalize_segments


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

    def _make_root(self, tmp: str):
        root = Path(tmp)
        tec_dir = root / "tec"
        indices_dir = root / "indices"
        tec_dir.mkdir()
        indices_dir.mkdir()
        return tec_dir, indices_dir

    def test_normalize_segments_formats(self):
        self.assertEqual(normalize_segments(200901), [(200901, 200901)])
        self.assertEqual(normalize_segments([200901, 201812]), [(200901, 201812)])
        self.assertEqual(
            normalize_segments([[200301, 200412], [200601, 200812]]),
            [(200301, 200412), (200601, 200812)],
        )
        # 三个及以上片段同样支持
        self.assertEqual(
            normalize_segments(
                [[200301, 200412], [200601, 200812], [201001, 201212]]
            ),
            [(200301, 200412), (200601, 200812), (201001, 201212)],
        )

    def test_multi_segment_windows_stay_inside_segment(self):
        # 2020 覆盖 1、2 月但只取 1 月，2022 取 1 月 -> 两个片段之间有真实缺口。
        with tempfile.TemporaryDirectory() as tmp:
            tec_dir, indices_dir = self._make_root(tmp)
            self._write_year(tec_dir, indices_dir, 2020, 40)
            self._write_year(tec_dir, indices_dir, 2022, 10)

            segments = [(202001, 202001), (202201, 202201)]
            common = {
                "tec_dir": tec_dir,
                "indices_dir": indices_dir,
                "segments": segments,
                "input_length": 6,
                "output_length": 6,
                "is_train": True,
                "tec_scaler": StandardScaler(),
                "aux_scaler": StandardScaler(),
            }
            stride1 = TecIonosphereDataset(window_step=1, **common)
            stride3 = TecIonosphereDataset(window_step=3, **common)

            # 片段 0 = 2020 年 1 月 372 帧，片段 1 = 2022 年 1 月 120 帧。
            self.assertEqual(
                [seg[1] for seg in stride1._segment_info], [372, 120]
            )
            self.assertEqual(
                [seg[2] for seg in stride1._segment_info], [361, 109]
            )
            self.assertEqual(len(stride1), 470)
            self.assertEqual(sum(s[2] for s in stride3._segment_info), 158)
            self.assertEqual(len(stride3), 158)

            for dataset in (stride1, stride3):
                self.assertEqual(
                    len(dataset),
                    sum(seg[2] for seg in dataset._segment_info),
                )
                window = dataset.input_length + dataset.output_length
                for seg_id, (gstart, count, valid) in enumerate(
                    dataset._segment_info
                ):
                    for local in (0, valid - 1):
                        index = sum(
                            ds[2] for ds in dataset._segment_info[:seg_id]
                        ) + local
                        start = gstart + local * dataset.window_step
                        self.assertLessEqual(start + window, gstart + count)
                        # 窗口首尾帧必须同年，说明没有跨片段取值。
                        self.assertEqual(
                            dataset._step_years[start],
                            dataset._step_years[start + window - 1],
                        )

    def test_legacy_start_end_matches_single_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tec_dir, indices_dir = self._make_root(tmp)
            self._write_year(tec_dir, indices_dir, 2020, 1)
            self._write_year(tec_dir, indices_dir, 2021, 1)

            legacy = TecIonosphereDataset(
                tec_dir=tec_dir,
                indices_dir=indices_dir,
                start_month=202001,
                end_month=202101,
                input_length=6,
                output_length=6,
                is_train=True,
                tec_scaler=StandardScaler(),
                aux_scaler=StandardScaler(),
            )
            explicit = TecIonosphereDataset(
                tec_dir=tec_dir,
                indices_dir=indices_dir,
                segments=[(202001, 202101)],
                window_step=1,
                input_length=6,
                output_length=6,
                is_train=True,
                tec_scaler=StandardScaler(),
                aux_scaler=StandardScaler(),
            )
            self.assertEqual(len(legacy), len(explicit))
            self.assertEqual(legacy.total_steps, explicit.total_steps)


if __name__ == "__main__":
    unittest.main()
