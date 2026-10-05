import numpy as np
from typing import Optional, Tuple, Dict, List
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
#  2. 空间结构指标：SSIM (NumPy 版本)
# ============================================================

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


def _convolve_2d(img, kernel, padding=0):
    """Gaussian smoothing, applied separably.

    The Gaussian window is an exact outer product, so convolving along the
    rows and then along the columns with the 1-D factor gives the same result
    as a single 2-D pass while doing ``2k`` instead of ``k*k`` multiply-adds
    per pixel. For the 11x11 window used by SSIM that is roughly a 10x
    speedup, which matters because evaluation smooths every validation frame.

    ``kernel`` may be either the 2-D window or its 1-D factor.
    Zero padding outside the image is expressed by ``mode="constant"``, which
    matches the previous ``cval=0.0`` behaviour.
    """
    del padding  # boundary handling comes from mode="constant"
    arr = np.asarray(img, dtype=np.float32)
    g = _as_1d_kernel(kernel)
    smoothed = _scipy_convolve1d(arr, g, axis=-2, mode="constant", cval=0.0)
    return _scipy_convolve1d(smoothed, g, axis=-1, mode="constant", cval=0.0)


def ssim_single_frame(img1, img2, window_size=11, sigma=1.5, data_range=None):
    if img1.ndim == 2:
        img1 = img1[None, None, :, :]
        img2 = img2[None, None, :, :]
    elif img1.ndim == 3:
        img1 = img1[None, :, :, :]
        img2 = img2[None, :, :, :]
    if data_range is None:
        data_range = (img2.max() - img2.min()).clip(min=1e-8)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel_2d(window_size, sigma)
    pad = window_size // 2
    mu1 = _convolve_2d(img1, kernel, pad)
    mu2 = _convolve_2d(img2, kernel, pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _convolve_2d(img1 * img1, kernel, pad) - mu1_sq
    sigma2_sq = _convolve_2d(img2 * img2, kernel, pad) - mu2_sq
    sigma12 = _convolve_2d(img1 * img2, kernel, pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def ssim_over_sequence(pred, target, window_size=11, sigma=1.5):
    B, T, H, W = pred.shape
    return ssim_single_frame(
        pred.reshape(B * T, H, W),
        target.reshape(B * T, H, W),
        window_size,
        sigma,
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
# ============================================================
#  5. 测试代码
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    B, T, H, W = 8, 12, 71, 73
    target = np.random.rand(B, T, H, W) * 80.0
    pred = target + np.random.randn(B, T, H, W) * 3.0
    print_evaluation(pred, target)
