import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy.ndimage import convolve as _scipy_convolve


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
    coords = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = np.outer(g, g)
    return kernel


def _convolve_2d(img, kernel, padding=0):
    if img.ndim == 2:
        img = img[None, None, :, :]
    elif img.ndim == 3:
        img = img[None, :, :, :]
    B, C, H, W = img.shape
    output = np.zeros_like(img, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            output[b, c] = _scipy_convolve(img[b, c], kernel, mode='constant', cval=0.0)
    return output.squeeze() if output.shape[0] == 1 and output.shape[1] == 1 else output


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
    ssim_vals = []
    for t in range(T):
        frame_pred = pred[:, t, :, :]
        frame_target = target[:, t, :, :]
        ssim_vals.append(ssim_single_frame(frame_pred, frame_target, window_size, sigma))
    return np.mean(ssim_vals)


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


# ============================================================
#  5. 测试代码
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    B, T, H, W = 8, 12, 71, 73
    target = np.random.rand(B, T, H, W) * 80.0
    pred = target + np.random.randn(B, T, H, W) * 3.0
    print_evaluation(pred, target)
