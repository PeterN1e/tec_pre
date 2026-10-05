from __future__ import annotations

import os
from os import listdir
from pathlib import Path
from typing import Sequence

import cdflib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data.tec_dataset import DEFAULT_AUX_COLUMNS, TecIonosphereDataset


class TecDataset1(Dataset):
    """Legacy in-memory dataset retained for ad-hoc experiments."""

    def __init__(self, data_tec, data_aux, seq_length):
        super().__init__()
        self.data_aux = data_aux.astype(np.float32)
        self.data_tec = data_tec.astype(np.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data_tec) - self.seq_length

    def __getitem__(self, index):
        return (
            self.data_tec[index:index + self.seq_length],
            self.data_aux[index:index + self.seq_length],
            [
                self.data_tec[index + self.seq_length],
                self.data_aux[index + self.seq_length],
            ],
        )


def _read_table(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        header = handle.read(4)
    if header == b"PK\x03\x04":
        return pd.read_excel(path)
    return pd.read_csv(path)


def data_reader(data_path, aux_columns: Sequence[str] = DEFAULT_AUX_COLUMNS):
    tec_batch = []
    tec_path_all = os.path.join(data_path, "tecMap")
    feature_path_all = os.path.join(data_path, "omni_2011_complete.csv")
    tec_data_list = listdir(tec_path_all)
    for filename in tec_data_list:
        tec_absolute_path = os.path.join(tec_path_all, filename)
        cdf = cdflib.CDF(tec_absolute_path)
        tec = cdf.varget("tecUHR")
        for hour in range(tec.shape[0]):
            tec_batch.append(tec[hour])

    feature_list = _read_table(Path(feature_path_all))
    return np.asarray(tec_batch), feature_list.loc[:, aux_columns].to_numpy()


__all__ = [
    "DEFAULT_AUX_COLUMNS",
    "TecDataset1",
    "TecIonosphereDataset",
    "data_reader",
]
