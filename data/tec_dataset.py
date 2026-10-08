from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import Dataset


DEFAULT_AUX_COLUMNS = ("kp", "ssn", "dst", "ap", "f10.7", "ae")
_YEAR_FILE_RE = re.compile(r"^(\d{4})_igsg\.npy$")


def _read_table(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        header = handle.read(4)
    if header == b"PK\x03\x04":
        return pd.read_excel(path)
    return pd.read_csv(path)


def _is_fitted(scaler: Any) -> bool:
    if scaler is None:
        return False
    try:
        check_is_fitted(scaler)
    except Exception:
        return False
    return True


def _month_in_range(
    years: np.ndarray,
    months: np.ndarray,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> np.ndarray:
    after_start = (years > start_year) | (
        (years == start_year) & (months >= start_month)
    )
    before_end = (years < end_year) | (
        (years == end_year) & (months <= end_month)
    )
    return after_start & before_end


def normalize_segments(value) -> List[Tuple[int, int]]:
    """Normalize split config to list of (start_ym, end_ym) tuples.
    
    Accepts:
        int              -> [(v, v)]
        [int, int]       -> [(a, b)]
        [[int,int], ...] -> [(a,b), ...]
    """
    if isinstance(value, int):
        return [(value, value)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Split config cannot be empty")
        if len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], int):
            return [(int(value[0]), int(value[1]))]
        return [(int(seg[0]), int(seg[1])) for seg in value]
    raise ValueError(f"Invalid split config format: {value}")


class TecIonosphereDataset(Dataset):
    """TEC sequence dataset with year-level caching and vectorized windows."""

    def __init__(
        self,
        tec_dir: str | Path,
        indices_dir: str | Path,
        start_month: Optional[int] = None,
        end_month: Optional[int] = None,
        segments: Optional[Sequence[Sequence[int]]] = None,
        window_step: int = 1,
        input_day_num: int = 3,
        output_day_num: int = 1,
        input_length: Optional[int] = None,
        output_length: Optional[int] = None,
        is_train: bool = False,
        tec_scaler: Any = None,
        aux_scaler: Any = None,
        aux_columns: Sequence[str] = DEFAULT_AUX_COLUMNS,
    ):
        super().__init__()
        self.tec_dir = Path(tec_dir)
        self.indices_dir = Path(indices_dir)
        self.input_length = int(input_length or input_day_num * 12)
        self.output_length = int(output_length or output_day_num * 12)
        self.input_day_num = input_day_num
        self.output_day_num = output_day_num
        self.is_train = is_train
        self.tec_scaler = tec_scaler
        self.aux_scaler = aux_scaler
        self.aux_columns = tuple(aux_columns)
        self.window_step = int(window_step)

        # Handle backward compatibility: if segments not provided, build from start_month/end_month
        if segments is None:
            if start_month is None or end_month is None:
                raise ValueError('Either segments or start_month/end_month must be provided')
            self.segments = [(int(start_month), int(end_month))]
        else:
            self.segments = normalize_segments(segments)
        
        # Derive year range from segments for _available_years
        self.start_y = min(s // 100 for s, _ in self.segments)
        self.end_y = max(e // 100 for _, e in self.segments)

        self._year_cache: Dict[
            Tuple[int, bool],
            Tuple[np.ndarray, np.ndarray],
        ] = {}
        self._build_index()
        self._fit_scalers_if_needed()

    def _available_years(self) -> List[int]:
        years = []
        for name in os.listdir(self.tec_dir):
            match = _YEAR_FILE_RE.match(name)
            if match:
                years.append(int(match.group(1)))
        return sorted(
            year
            for year in years
            if self.start_y <= year <= self.end_y
        )

    def _build_index(self) -> None:
        self.years = self._available_years()
        if not self.years:
            raise FileNotFoundError(f"No TEC year files found in {self.tec_dir}")

        time_index: List[List[int]] = []
        step_years: List[int] = []
        step_local_indices: List[int] = []
        step_segment_ids: List[int] = []
        year_offsets: Dict[int, int] = {}
        year_step_counts: Dict[int, int] = {}
        global_idx = 0

        for year in self.years:
            year_offsets[year] = global_idx
            csv_path = self.indices_dir / f"{year}_indices.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Index file not found: {csv_path}")
            frame = _read_table(csv_path)
            required = {"year", "doy", "hour", *self.aux_columns}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"Missing columns in {csv_path}: {sorted(missing)}"
                )

            dates = pd.to_datetime(
                frame["year"].astype(str)
                + frame["doy"].astype(str).str.zfill(3),
                format="%Y%j",
            )
            frame = frame.assign(date=dates, month=dates.dt.month)
            
            # Multi-segment filtering
            frame_ym = frame["year"].to_numpy() * 100 + frame["month"].to_numpy()
            segment_ids = np.full(len(frame), -1, dtype=np.int32)
            for seg_id, (seg_start, seg_end) in enumerate(self.segments):
                mask = (frame_ym >= seg_start) & (frame_ym <= seg_end)
                segment_ids[mask] = seg_id
            
            frame = frame.loc[(segment_ids >= 0) & (frame["hour"] % 2 == 0)]
            frame = frame.assign(segment_id=segment_ids[(segment_ids >= 0) & (frame["hour"] % 2 == 0)])
            frame = frame.reset_index(drop=True)

            count = 0
            for row in frame.itertuples(index=False):
                local_index = (int(row.doy) - 1) * 12 + int(row.hour) // 2
                time_index.append(
                    [
                        int(row.year),
                        int(row.month),
                        int(row.date.day),
                        int(row.hour),
                        global_idx,
                    ]
                )
                step_years.append(int(row.year))
                step_local_indices.append(local_index)
                step_segment_ids.append(int(row.segment_id))
                global_idx += 1
                count += 1
            year_step_counts[year] = count

        self.time_index = time_index
        self.total_steps = global_idx
        self.year_offset = year_offsets
        self.year_step_count = year_step_counts
        self._step_years = np.asarray(step_years, dtype=np.int32)
        self._step_local_indices = np.asarray(step_local_indices, dtype=np.int64)
        self._step_segment_ids = np.asarray(step_segment_ids, dtype=np.int32)
        
        # Build segment info: (global_start, frame_count, valid_samples)
        self._segment_info = []
        total_valid = 0
        for seg_id in range(len(self.segments)):
            seg_mask = self._step_segment_ids == seg_id
            seg_indices = np.where(seg_mask)[0]
            if len(seg_indices) == 0:
                self._segment_info.append((0, 0, 0))
                continue
            seg_global_start = int(seg_indices[0])
            seg_frame_count = int(seg_indices[-1] - seg_indices[0] + 1)
            seg_valid = max(0, (seg_frame_count - self.input_length - self.output_length) // self.window_step + 1)
            self._segment_info.append((seg_global_start, seg_frame_count, seg_valid))
            total_valid += seg_valid
        
        self.valid_samples = total_valid
        if self.valid_samples < 1:
            raise ValueError("Not enough time steps to build one training window")

    def _fit_scalers_if_needed(self) -> None:
        if not self.is_train:
            return
        if self.tec_scaler is None and self.aux_scaler is None:
            return

        for year in self.years:
            tec, aux = self._load_year_data(year, apply_scaler=False)
            if self.tec_scaler is not None and not _is_fitted(self.tec_scaler):
                self.tec_scaler.partial_fit(tec.reshape(tec.shape[0], -1))
            if self.aux_scaler is not None and not _is_fitted(self.aux_scaler):
                self.aux_scaler.partial_fit(aux)

        # Replace raw caches with scaled caches lazily on the next access.
        self._year_cache.clear()

    def _year_data_path(self, year: int) -> Path:
        return self.tec_dir / f"{year}_igsg.npy"

    def _load_year_data(
        self,
        year: int,
        apply_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = (year, apply_scaler)
        cached = self._year_cache.get(cache_key)
        if cached is not None:
            return cached

        tec_path = self._year_data_path(year)
        if not tec_path.exists():
            raise FileNotFoundError(f"TEC file not found: {tec_path}")
        tec = np.load(tec_path).astype(np.float32, copy=False)
        tec = tec.reshape(-1, tec.shape[-2], tec.shape[-1])
        spatial_shape = tec.shape[1:]

        csv_path = self.indices_dir / f"{year}_indices.csv"
        frame = _read_table(csv_path)
        frame = frame.loc[frame["hour"] % 2 == 0].reset_index(drop=True)
        aux = frame.loc[:, self.aux_columns].to_numpy(dtype=np.float32)

        if apply_scaler:
            if _is_fitted(self.tec_scaler):
                tec = self.tec_scaler.transform(
                    tec.reshape(tec.shape[0], -1)
                ).reshape(-1, *spatial_shape)
            if _is_fitted(self.aux_scaler):
                aux = self.aux_scaler.transform(aux)

        self._year_cache[cache_key] = (tec, aux)
        return tec, aux

    def _collect_sequence(
        self,
        start: int,
        length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        end = start + length
        years = self._step_years[start:end]
        local_indices = self._step_local_indices[start:end]
        if years.size == 0:
            raise IndexError("Empty sequence requested")

        boundaries = np.flatnonzero(years[1:] != years[:-1]) + 1
        tec_chunks = []
        aux_chunks = []
        for positions in np.split(np.arange(length), boundaries):
            year = int(years[positions[0]])
            tec, aux = self._load_year_data(year)
            local = local_indices[positions]
            if local.max() >= tec.shape[0] or local.max() >= aux.shape[0]:
                raise IndexError(
                    f"Local index {local.max()} is out of range for year {year}"
                )
            tec_chunks.append(tec[local])
            aux_chunks.append(aux[local])
        return np.concatenate(tec_chunks), np.concatenate(aux_chunks)

    def _get_step_data(self, global_index: int):
        year, month, day, hour, _ = self.time_index[global_index]
        del month, day, hour
        local_index = int(self._step_local_indices[global_index])
        tec, aux = self._load_year_data(year)
        return tec[local_index], aux[local_index]

    def __len__(self) -> int:
        return self.valid_samples

    def _resolve_sample(self, index: int) -> Tuple[int, int]:
        """Resolve flat sample index to (segment_id, offset_within_segment)."""
        cumulative = 0
        for seg_id, (_, _, seg_valid) in enumerate(self._segment_info):
            if index < cumulative + seg_valid:
                return seg_id, index - cumulative
            cumulative += seg_valid
        raise IndexError(f"Sample index {index} out of range")

    def __getitem__(self, index: int):
        if index < 0 or index >= self.valid_samples:
            raise IndexError(index)
        seg_id, local_offset = self._resolve_sample(index)
        seg_global_start = self._segment_info[seg_id][0]
        global_start = seg_global_start + local_offset * self.window_step
        tec_in, aux_in = self._collect_sequence(global_start, self.input_length)
        tec_gt, aux_gt = self._collect_sequence(
            global_start + self.input_length,
            self.output_length,
        )
        return (
            torch.from_numpy(tec_in).float(),
            torch.from_numpy(aux_in).float(),
            torch.from_numpy(tec_gt).float(),
            torch.from_numpy(aux_gt).float(),
        )
