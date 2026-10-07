from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import convolve1d as _scipy_convolve1d


# ============================================================
#  1. 精度指标：RMSE / MAE / R2 (NumPy 版本)
# ============================================================

def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def mae(pred, target):
    return np.mean(np.abs(pred - target))

def r2_score(pred, target):
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ============================================================
#  2. 空间结构指标：SSIM (NumPy 版本, 分块计算)
# ============================================================

# 每次送入卷积的帧数上限。71x73 一帧约 20 KB，
# 256 帧的中间量只有几十 MB，峰值内存与总帧数无关。
DEFAULT_SSIM_CHUNK = 256


def _gaussian_kernel_2d(kernel_size=11, sigma=1.5):
    return np.outer(_gaussian_kernel_1d(kernel_size, sigma),
                    _gaussian_kernel_1d(kernel_size, sigma))


def _gaussian_kernel_1d(kernel_size=11, sigma=1.5):
    coords = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.astype(np.float32)


def _as_1d_kernel(kernel):
    """Return the 1-D factor of a rank-1 (separable) 2-D kernel."""
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim == 1:
        return kernel
    if kernel.ndim != 2:
        raise ValueError("kernel must be 1-D or 2-D")
    # kernel == outer(g, g)  =>  kernel[:, 0] / sqrt(kernel[0, 0]) == g
    return (kernel[:, 0] / np.sqrt(kernel[0, 0])).astype(np.float32)


def _convolve_1d_pair(arr, kernel_1d):
    """Separable Gaussian smoothing over the last two axes (lat then lon)."""
    smoothed = _scipy_convolve1d(
        arr, kernel_1d, axis=-2, mode="constant", cval=0.0
    )
    return _scipy_convolve1d(
        smoothed, kernel_1d, axis=-1, mode="constant", cval=0.0
    )


def _convolve_2d(img, kernel, padding=0):
    """Backward-compatible entry point accepting 1-D or 2-D Gaussian windows."""
    del padding  # boundary handling comes from mode="constant"
    arr = np.asarray(img, dtype=np.float32)
    return _convolve_1d_pair(arr, _as_1d_kernel(kernel))


def _as_frame_stack(img, name):
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"{name} must be (H, W), (N, H, W) or (1, N, H, W); got {arr.shape}"
    )


def ssim_per_frame(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
    data_range=None,
    max_frames=DEFAULT_SSIM_CHUNK,
):
    """逐帧 SSIM 均值，峰值内存有界。

    旧实现把整个 ``(B*T, H, W)`` 展平后一次性卷积。本项目测试集有约 21 万
    帧，仅中间量就超过十 GB，进程因此被内核 SIGKILL(137)。按 ``max_frames``
    分块后，峰值内存与总帧数无关，逐帧结果完全一致。

    返回形状 ``(N,)`` 的 float64 数组，每个元素是该帧 ``ssim_map.mean()``。
    ``data_range`` 在分块前由 ``img2`` 整体求得，因此与不分块的结果一致。
    """
    a = _as_frame_stack(img1, "img1")
    b = _as_frame_stack(img2, "img2")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    n_frames = int(a.shape[0])
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    if data_range is None:
        data_range = float(np.nanmax(b) - np.nanmin(b))
    data_range = max(float(data_range), 1e-8)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    g = _gaussian_kernel_1d(window_size, sigma)

    out = np.empty(n_frames, dtype=np.float64)
    step = max(1, int(max_frames))
    for start in range(0, n_frames, step):
        stop = min(start + step, n_frames)
        x = np.asarray(a[start:stop], dtype=np.float32)
        y = np.asarray(b[start:stop], dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
            y = y[None]
        mu1 = _convolve_1d_pair(x, g)
        mu2 = _convolve_1d_pair(y, g)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = _convolve_1d_pair(x * x, g) - mu1_sq
        sigma2_sq = _convolve_1d_pair(y * y, g) - mu2_sq
        sigma12 = _convolve_1d_pair(x * y, g) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        out[start:stop] = ssim_map.reshape(stop - start, -1).mean(axis=1)
    return out


def ssim_single_frame(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
    data_range=None,
    max_frames=DEFAULT_SSIM_CHUNK,
):
    return float(
        ssim_per_frame(
            img1, img2, window_size, sigma, data_range, max_frames
        ).mean()
    )


def ssim_over_sequence(
    pred,
    target,
    window_size=11,
    sigma=1.5,
    max_frames=DEFAULT_SSIM_CHUNK,
):
    b, t, h, w = pred.shape
    return ssim_single_frame(
        pred.reshape(b * t, h, w),
        target.reshape(b * t, h, w),
        window_size,
        sigma,
        max_frames=max_frames,
    )


# ============================================================
#  3. 统一评估函数
# ============================================================

def evaluate_all(pred, target, ssim_window=11, ssim_sigma=1.5):
    results = {
        "RMSE": rmse(pred, target),
        "MAE": mae(pred, target),
        "R2": r2_score(pred, target),
        "SSIM": ssim_over_sequence(pred, target, ssim_window, ssim_sigma),
    }
    return results


def skill_score(pred, target, baseline):
    """Murphy skill score against a reference forecast."""
    baseline_mse = np.mean((baseline - target) ** 2)
    model_mse = np.mean((pred - target) ** 2)
    if baseline_mse <= 1e-12:
        return float("nan")
    return float(1.0 - model_mse / baseline_mse)


def evaluate_with_baseline(pred, target, baseline, ssim_window=11, ssim_sigma=1.5):
    results = evaluate_all(pred, target, ssim_window, ssim_sigma)
    results["SkillScore"] = skill_score(pred, target, baseline)
    results["BaselineRMSE"] = rmse(baseline, target)
    return results


def evaluate_by_latitude_band(pred, target, bands=6):
    """Report aggregate errors by equal-height latitude bands."""
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    height = pred.shape[-2]
    bands = max(1, min(int(bands), height))
    indices = np.array_split(np.arange(height), bands)
    results = {}
    for band_index, rows in enumerate(indices, start=1):
        pred_band = pred[..., rows, :]
        target_band = target[..., rows, :]
        results[f"lat_band_{band_index}"] = {
            "RMSE": rmse(pred_band, target_band),
            "MAE": mae(pred_band, target_band),
            "R2": r2_score(pred_band, target_band),
        }
    return results


# ============================================================
#  4. 逐步评估函数 (TEC预测论文标准)
# ============================================================

def evaluate_per_step(pred, target, ssim_window=11, ssim_sigma=1.5):
    B, T, H, W = pred.shape
    results = {"RMSE": [], "MAE": [], "R2": [], "SSIM": []}
    for t in range(T):
        p = pred[:, t, :, :]
        a = target[:, t, :, :]
        results["RMSE"].append(rmse(p, a))
        results["MAE"].append(mae(p, a))
        results["R2"].append(r2_score(p, a))
        results["SSIM"].append(ssim_single_frame(p, a, ssim_window, ssim_sigma))
    return results


# ============================================================
#  5. 流式评估：内存占用与样本数解耦
# ============================================================

def _band_rows(height: int, bands: int) -> List[np.ndarray]:
    bands = max(1, min(int(bands), int(height)))
    return np.array_split(np.arange(int(height)), bands)


def _blank_stats(size: int) -> Dict[str, np.ndarray]:
    return {
        "count": np.zeros(size, dtype=np.float64),
        "sum_t": np.zeros(size, dtype=np.float64),
        "sum_t2": np.zeros(size, dtype=np.float64),
        "sum_sq_err": np.zeros(size, dtype=np.float64),
        "sum_abs_err": np.zeros(size, dtype=np.float64),
    }


def _reduce_stats(box: Dict[str, np.ndarray]) -> Dict[str, float]:
    count = float(box["count"])
    if count <= 0:
        return {
            "RMSE": float("nan"),
            "MAE": float("nan"),
            "R2": float("nan"),
        }
    ss_tot = box["sum_t2"] - (box["sum_t"] ** 2) / count
    return {
        "RMSE": float(np.sqrt(box["sum_sq_err"] / count)),
        "MAE": float(box["sum_abs_err"] / count),
        "R2": float(1.0 - box["sum_sq_err"] / (ss_tot + 1e-8)),
    }


class StreamingMetrics:
    """流式 RMSE / MAE / R2 / SSIM 累加器。

    逐块喂入形状 ``(B, T, H, W)`` 的预测与真值，只保留充分统计量。峰值内存
    约为一个块的大小，与测试集总样本数无关。旧实现在测试集上要同时驻留
    pred/target/persistence 三个 4.35 GB 数组并再复制一份，峰值接近 48 GB，
    正是进程被 SIGKILL 的原因。

    产出的 dict 结构与 :func:`evaluate_arrays` 一致：``aggregate``、
    ``per_step``、``latitude_bands`` 三块。
    """

    def __init__(
        self,
        bands: int = 6,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
        compute_ssim: bool = False,
        data_range: Optional[float] = None,
    ):
        self.bands = int(bands)
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self.compute_ssim = bool(compute_ssim)
        self.data_range = None if data_range is None else float(data_range)

        self.count = 0
        self.sum_t = 0.0
        self.sum_t2 = 0.0
        self.sum_sq_err = 0.0
        self.sum_abs_err = 0.0
        self.baseline_count = 0
        self.baseline_sq_err = 0.0

        self._num_steps: Optional[int] = None
        self._height: Optional[int] = None
        self._step_stats: Optional[Dict[str, np.ndarray]] = None
        self._band_stats: Optional[Dict[str, np.ndarray]] = None

        # SSIM takes a global data_range. When the caller supplies it up front,
        # chunked SSIM equals the whole-array definition, so only running sums
        # are needed (no cached prediction stack).
        self.target_min = np.inf
        self.target_max = -np.inf
        self._ssim_sum = 0.0
        self._ssim_count = 0
        self._ssim_step_sum: Optional[np.ndarray] = None
        self._ssim_step_count: Optional[np.ndarray] = None

    # ---- ingestion ---------------------------------------------------

    def _check(self, pred: np.ndarray, target: np.ndarray) -> None:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must share a shape; got {pred.shape} vs "
                f"{target.shape}"
            )
        if pred.ndim != 4:
            raise ValueError(f"expected (B, T, H, W) chunk, got {pred.shape}")

    def _ensure_buffers(self, steps: int, height: int) -> None:
        if self._num_steps is None:
            self._num_steps = int(steps)
            self._height = int(height)
            self._step_stats = _blank_stats(int(steps))
            self._band_stats = _blank_stats(len(_band_rows(height, self.bands)))
            return
        if self._num_steps != int(steps):
            raise ValueError(
                f"inconsistent step count across chunks: "
                f"{self._num_steps} vs {steps}"
            )
        if self._height != int(height):
            raise ValueError(
                f"inconsistent grid height across chunks: "
                f"{self._height} vs {height}"
            )

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> None:
        self._check(pred, target)
        p = np.asarray(pred, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        b, steps, height, width = p.shape
        self._ensure_buffers(steps, height)

        diff = p - t
        sq = diff * diff
        ab = np.abs(diff)

        self.count += int(sq.size)
        self.sum_t += float(t.sum())
        self.sum_t2 += float(np.square(t).sum())
        self.sum_sq_err += float(sq.sum())
        self.sum_abs_err += float(ab.sum())

        axes = (0, 2, 3)
        self._step_stats["count"] += float(b * height * width)
        self._step_stats["sum_t"] += t.sum(axis=axes)
        self._step_stats["sum_t2"] += np.square(t).sum(axis=axes)
        self._step_stats["sum_sq_err"] += sq.sum(axis=axes)
        self._step_stats["sum_abs_err"] += ab.sum(axis=axes)

        for index, rows in enumerate(_band_rows(height, self.bands)):
            p_band = p[:, :, rows, :]
            t_band = t[:, :, rows, :]
            d_band = p_band - t_band
            self._band_stats["count"][index] += float(d_band.size)
            self._band_stats["sum_t"][index] += float(t_band.sum())
            self._band_stats["sum_t2"][index] += float(np.square(t_band).sum())
            self._band_stats["sum_sq_err"][index] += float((d_band * d_band).sum())
            self._band_stats["sum_abs_err"][index] += float(np.abs(d_band).sum())

        if baseline is not None:
            base = np.asarray(baseline, dtype=np.float64)
            if base.shape != t.shape:
                raise ValueError(
                    f"baseline shape {base.shape} does not match target {t.shape}"
                )
            base_diff = base - t
            self.baseline_count += int(base_diff.size)
            self.baseline_sq_err += float((base_diff * base_diff).sum())

        if self.compute_ssim:
            self._accumulate_ssim(p, t)

    def _accumulate_ssim(self, p: np.ndarray, t: np.ndarray) -> None:
        if self.data_range is None:
            raise ValueError(
                "compute_ssim=True requires a precomputed data_range so that "
                "chunked SSIM matches the whole-array definition"
            )
        b, steps, height, width = p.shape
        if self._ssim_step_sum is None:
            self._ssim_step_sum = np.zeros(steps, dtype=np.float64)
            self._ssim_step_count = np.zeros(steps, dtype=np.float64)
        values = ssim_per_frame(
            p.reshape(b * steps, height, width),
            t.reshape(b * steps, height, width),
            self.ssim_window,
            self.ssim_sigma,
            data_range=self.data_range,
        ).reshape(b, steps)
        self._ssim_step_sum += values.sum(axis=0)
        self._ssim_step_count += b
        self._ssim_sum += float(values.sum())
        self._ssim_count += int(values.size)

    @property
    def ssim(self) -> float:
        if self._ssim_count <= 0:
            return float("nan")
        return float(self._ssim_sum / self._ssim_count)

    def ssim_per_step(self) -> List[float]:
        if self._ssim_step_sum is None:
            return []
        return [
            float(self._ssim_step_sum[i] / self._ssim_step_count[i])
            if self._ssim_step_count[i]
            else float("nan")
            for i in range(self._ssim_step_sum.size)
        ]

    # ---- reduction ---------------------------------------------------

    def aggregate(self) -> Dict[str, float]:
        count = float(self.count)
        if count <= 0:
            return {
                "RMSE": float("nan"),
                "MAE": float("nan"),
                "R2": float("nan"),
                "SSIM": float("nan"),
                "SkillScore": float("nan"),
                "BaselineRMSE": float("nan"),
            }
        ss_tot = self.sum_t2 - (self.sum_t ** 2) / count
        model_mse = self.sum_sq_err / count
        out = {
            "RMSE": float(np.sqrt(model_mse)),
            "MAE": float(self.sum_abs_err / count),
            "R2": float(1.0 - self.sum_sq_err / (ss_tot + 1e-8)),
            "SSIM": self.ssim,
        }
        base_count = float(self.baseline_count)
        if base_count > 0:
            base_mse = self.baseline_sq_err / base_count
            out["BaselineRMSE"] = float(np.sqrt(base_mse))
            out["SkillScore"] = (
                float("nan") if base_mse <= 1e-12
                else float(1.0 - model_mse / base_mse)
            )
        else:
            out["BaselineRMSE"] = float("nan")
            out["SkillScore"] = float("nan")
        return out

    def per_step(self) -> Dict[str, List[float]]:
        if self._step_stats is None:
            return {"RMSE": [], "MAE": [], "R2": []}
        stats = self._step_stats
        out: Dict[str, List[float]] = {"RMSE": [], "MAE": [], "R2": []}
        for index in range(self._num_steps):
            box = {key: value[index] for key, value in stats.items()}
            reduced = _reduce_stats(box)
            out["RMSE"].append(reduced["RMSE"])
            out["MAE"].append(reduced["MAE"])
            out["R2"].append(reduced["R2"])
        ssim_steps = self.ssim_per_step()
        if ssim_steps:
            out["SSIM"] = ssim_steps
        return out

    def latitude_bands(self) -> Dict[str, Dict[str, float]]:
        if self._band_stats is None:
            return {}
        stats = self._band_stats
        out: Dict[str, Dict[str, float]] = {}
        for index in range(stats["count"].size):
            box = {key: value[index] for key, value in stats.items()}
            out[f"lat_band_{index + 1}"] = _reduce_stats(box)
        return out


def ssim_over_memmap(
    prediction,
    target,
    chunk: int = DEFAULT_SSIM_CHUNK,
    window_size: int = 11,
    sigma: float = 1.5,
    sample_chunk: int = 256,
):
    """在磁盘映射数组上分块计算 SSIM，返回 (整体SSIM, 每步SSIM列表)。

    只读入 ``sample_chunk`` 个样本，峰值内存几十 MB。
    """
    n_samples, steps, height, width = prediction.shape
    global_min = np.inf
    global_max = -np.inf
    for start in range(0, n_samples, sample_chunk):
        stop = min(start + sample_chunk, n_samples)
        block = np.asarray(target[start:stop], dtype=np.float32)
        if block.size:
            global_min = min(global_min, float(np.nanmin(block)))
            global_max = max(global_max, float(np.nanmax(block)))
    data_range = max(global_max - global_min, 1e-8)

    per_step_sum = np.zeros(steps, dtype=np.float64)
    per_step_count = np.zeros(steps, dtype=np.float64)
    total_sum = 0.0
    total_count = 0

    for start in range(0, n_samples, sample_chunk):
        stop = min(start + sample_chunk, n_samples)
        p = np.asarray(prediction[start:stop], dtype=np.float32)
        t = np.asarray(target[start:stop], dtype=np.float32)
        if p.size == 0:
            continue
        p_frames = p.reshape(-1, height, width)
        t_frames = t.reshape(-1, height, width)
        values = ssim_per_frame(
            p_frames,
            t_frames,
            window_size,
            sigma,
            data_range=data_range,
            max_frames=chunk,
        )
        values = values.reshape(stop - start, steps)
        per_step_sum += values.sum(axis=0)
        per_step_count += values.shape[0]
        total_sum += float(values.sum())
        total_count += int(values.size)

    overall = float(total_sum / total_count) if total_count else float("nan")
    step_values = [
        float(per_step_sum[i] / per_step_count[i]) if per_step_count[i] else float("nan")
        for i in range(steps)
    ]
    return overall, step_values


def evaluate_stream(
    prediction,
    target,
    persistence=None,
    bands: int = 6,
    ssim_window: int = 11,
    ssim_sigma: float = 1.5,
    sample_chunk: int = 256,
    compute_ssim: bool = True,
) -> Dict[str, Any]:
    """对流式/磁盘映射数组计算完整指标，内存占用有界。

    参数为实现了 ``shape`` 与切片语义的任意数组（``np.memmap`` 即可）。
    """
    accumulator = StreamingMetrics(
        bands=bands,
        ssim_window=ssim_window,
        ssim_sigma=ssim_sigma,
    )
    n_samples = int(prediction.shape[0])
    step = max(1, int(sample_chunk))
    for start in range(0, n_samples, step):
        stop = min(start + step, n_samples)
        p = np.asarray(prediction[start:stop], dtype=np.float32)
        t = np.asarray(target[start:stop], dtype=np.float32)
        if p.size == 0:
            continue
        base = None
        if persistence is not None:
            base = np.asarray(persistence[start:stop], dtype=np.float32)
        accumulator.update(p, t, base)

    aggregate = accumulator.aggregate()
    per_step = accumulator.per_step()
    if compute_ssim:
        overall, per_step_ssim = ssim_over_memmap(
            prediction,
            target,
            window_size=ssim_window,
            sigma=ssim_sigma,
            sample_chunk=sample_chunk,
        )
        aggregate["SSIM"] = overall
        per_step["SSIM"] = per_step_ssim
    else:
        aggregate["SSIM"] = float("nan")
        per_step["SSIM"] = [float("nan")] * int(prediction.shape[1])

    return {
        "aggregate": aggregate,
        "per_step": per_step,
        "latitude_bands": accumulator.latitude_bands(),
    }


# ============================================================
#  6. 打印 / 日志辅助
# ============================================================

def print_evaluation(pred, target, ssim_window=11, ssim_sigma=1.5):
    agg = evaluate_all(pred, target, ssim_window, ssim_sigma)
    print()
    print("=" * 60)
    print("  Aggregate Metrics (all prediction steps mixed)")
    print("=" * 60)
    for k, v in agg.items():
        print(f"  {k:>6s} : {v:.6f}")
    step = evaluate_per_step(pred, target, ssim_window, ssim_sigma)
    T = pred.shape[1]
    print()
    print("=" * 60)
    max_h = 2 * T
    print(f"  Per-Step Metrics (t+2h ~ t+{max_h}h)")
    print("=" * 60)
    print(f"  {'Step':>6s} {'Horizon':>8s} | {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'SSIM':>8s}")
    print("  " + "-" * 56)
    for t in range(T):
        h = 2 * (t + 1)
        horizon = f"t+{h}h"
        print(f"  {t+1:>6d} {horizon:>8s} | "
              f"{step['RMSE'][t]:8.4f} {step['MAE'][t]:8.4f} "
              f"{step['R2'][t]:8.4f} {step['SSIM'][t]:8.4f}")
    print("=" * 60)


def log_evaluation(logger, pred, target, ssim_window=11, ssim_sigma=1.5):
    """Log evaluation metrics to the provided logger."""
    agg = evaluate_all(pred, target, ssim_window, ssim_sigma)
    step = evaluate_per_step(pred, target, ssim_window, ssim_sigma)
    T = pred.shape[1]
    max_h = 2 * T

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Evaluation Metrics")
    logger.info("=" * 60)
    for k, v in agg.items():
        logger.info(f"  {k:>6s} : {v:.6f}")
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Per-Step Metrics (t+2h ~ t+{max_h}h)")
    logger.info("=" * 60)
    logger.info(f"  {'Step':>6s} {'Horizon':>8s} | {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'SSIM':>8s}")
    logger.info("  " + "-" * 56)
    for t in range(T):
        h = 2 * (t + 1)
        horizon = f"t+{h}h"
        logger.info(f"  {t+1:>6d} {horizon:>8s} | "
              f"{step['RMSE'][t]:8.4f} {step['MAE'][t]:8.4f} "
              f"{step['R2'][t]:8.4f} {step['SSIM'][t]:8.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    np.random.seed(42)
    B, T, H, W = 8, 12, 71, 73
    target = np.random.rand(B, T, H, W) * 80.0
    pred = target + np.random.randn(B, T, H, W) * 3.0
    print_evaluation(pred, target)
