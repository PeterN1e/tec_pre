import numpy as np
from typing import Optional, Tuple, Dict


# ============================================================
#  1. 精度指标：RMSE / MAE / R² (NumPy 版本)
# ============================================================

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    均方根误差 (Root Mean Square Error)
    pred, target: (B, T, H, W)  — 单位 TECU
    返回: 标量 float
    """
    return np.sqrt(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    平均绝对误差 (Mean Absolute Error)
    返回: 标量 float
    """
    return np.mean(np.abs(pred - target))


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    决定系数 R²
    返回: 标量, 越接近1越好
    """
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-8)


# ============================================================
#  2. 空间结构指标：SSIM (NumPy 版本)
# ============================================================

def _gaussian_kernel_2d(
        kernel_size: int = 11,
        sigma: float = 1.5,
) -> np.ndarray:
    """生成 2D 高斯卷积核 (NumPy 版本)"""
    coords = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = np.outer(g, g)  # (K, K)
    return kernel  # (K, K)


def _convolve_2d(
        img: np.ndarray,
        kernel: np.ndarray,
        padding: int = 0,
) -> np.ndarray:
    """
    使用 NumPy 实现 2D 卷积（带 padding）
    img: (B, C, H, W) 或 (H, W)
    kernel: (K, K)
    返回: (B, C, H, W) 或 (H, W)
    """
    # 统一维度
    if img.ndim == 2:
        img = img[None, None, :, :]  # (1, 1, H, W)
    elif img.ndim == 3:
        img = img[None, :, :, :]  # (1, C, H, W)

    B, C, H, W = img.shape
    K = kernel.shape[0]

    # Padding
    pad_h = padding
    pad_w = padding
    img_padded = np.pad(img, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # 为每个通道准备输出
    output = np.zeros((B, C, H, W), dtype=np.float32)

    # 对每个 batch 和每个通道进行卷积
    for b in range(B):
        for c in range(C):
            # 提取当前通道
            channel = img_padded[b, c]  # (H+2*pad, W+2*pad)

            # 滑窗卷积（使用 NumPy 的滑动窗口技巧）
            # 为了简化，使用 for 循环（对于小窗口足够）
            for i in range(H):
                for j in range(W):
                    # 提取窗口
                    window = channel[i:i + K, j:j + K]
                    # 卷积（点乘求和）
                    output[b, c, i, j] = np.sum(window * kernel)

    return output.squeeze() if output.shape[0] == 1 and output.shape[1] == 1 else output


def ssim_single_frame(
        img1: np.ndarray,
        img2: np.ndarray,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: Optional[float] = None,
) -> float:
    """
    单帧 SSIM (NumPy 版本)
    img1, img2: (B, 1, H, W) 或 (H, W)
    返回: 标量 float
    """
    # 统一维度为 (B, C, H, W)
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

    # 生成高斯核
    kernel = _gaussian_kernel_2d(window_size, sigma)
    pad = window_size // 2

    # 计算均值
    mu1 = _convolve_2d(img1, kernel, pad)
    mu2 = _convolve_2d(img2, kernel, pad)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差
    sigma1_sq = _convolve_2d(img1 * img1, kernel, pad) - mu1_sq
    sigma2_sq = _convolve_2d(img2 * img2, kernel, pad) - mu2_sq
    sigma12 = _convolve_2d(img1 * img2, kernel, pad) - mu1_mu2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def ssim_over_sequence(
        pred: np.ndarray,
        target: np.ndarray,
        window_size: int = 11,
        sigma: float = 1.5,
) -> float:
    """
    对整个 (B, T, H, W) 序列计算 SSIM (NumPy 版本)
    逐帧计算后取时间维平均
    返回: 标量 float
    """
    B, T, H, W = pred.shape
    ssim_vals = []

    for t in range(T):
        frame_pred = pred[:, t, :, :]  # (B, H, W)
        frame_target = target[:, t, :, :]  # (B, H, W)
        ssim_vals.append(
            ssim_single_frame(frame_pred, frame_target, window_size, sigma)
        )

    return np.mean(ssim_vals)


# ============================================================
#  3. 统一评估函数 (NumPy 版本)
# ============================================================

def evaluate_all(
        pred: np.ndarray,
        target: np.ndarray,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
) -> Dict[str, float]:
    """
    pred, target: (B, T, H, W)  例如 (32, 12, 71, 73)
    返回包含所有指标的字典 (NumPy 版本)
    """
    results = {
        "RMSE": rmse(pred, target),
        "MAE": mae(pred, target),
        "R2": r2_score(pred, target),
        "SSIM": ssim_over_sequence(pred, target, ssim_window, ssim_sigma),
    }
    return results


# ============================================================
#  4. 测试代码
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    B, T, H, W = 8, 12, 71, 73
    target = np.random.rand(B, T, H, W) * 80.0
    pred = target + np.random.randn(B, T, H, W) * 3.0

    metrics = evaluate_all(pred, target)
    print("=" * 50)
    print(f"  预测帧数 : {T} 帧  (每帧2h, 共{2 * T}h)")
    print(f"  空间网格 : {H}×{W}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:>6s} : {v:.6f}")
    print("=" * 50)