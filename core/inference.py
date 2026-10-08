from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from core.checkpoint import load_model_state
from core.config import (
    model_slug,
    resolve_data_paths,
    resolve_output_dir,
)
from core.evaluation import (
    ARRAY_NAMES,
    evaluate_arrays,
    load_predictions,
    run_inference,
    save_metrics,
)
from core.registry import build_model
from core.trainer import resolve_device, seed_everything
from data.tec_dataset import TecIonosphereDataset, normalize_segments


def model_directory(config: Dict[str, Any]) -> Path:
    return resolve_output_dir(config) / model_slug(config["model"]["name"])


def load_scalers(model_dir: str | Path):
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "tec_scaler.pkl"),
        joblib.load(model_dir / "aux_scaler.pkl"),
    )


def build_split_loader(
    config: Dict[str, Any],
    split: str,
    tec_scaler,
    aux_scaler,
    device: torch.device,
) -> DataLoader:
    paths = resolve_data_paths(config)
    data_cfg = config["data"]
    segments = normalize_segments(data_cfg["splits"][split])
    window_step = int(data_cfg["window_step_eval"])
    dataset = TecIonosphereDataset(
        tec_dir=paths["tec_dir"],
        indices_dir=paths["indices_dir"],
        segments=segments,
        window_step=window_step,
        input_length=int(data_cfg["input_length"]),
        output_length=int(data_cfg["output_length"]),
        is_train=False,
        tec_scaler=tec_scaler,
        aux_scaler=aux_scaler,
        aux_columns=data_cfg["aux_columns"],
    )
    pin_memory = bool(data_cfg["pin_memory"]) and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(data_cfg["num_workers"]),
        pin_memory=pin_memory,
    )


def _inverse_transform_inplace(
    array: np.ndarray,
    scaler,
    sample_chunk: int = 256,
    spatial: bool = True,
) -> np.ndarray:
    """逐块反标准化并原地写回磁盘映射数组。

    ``common.Data_Preprocessing.inverse_transform_predictions`` 会把整份数组
    reshape 成二维再返回一份拷贝。测试集单个数组 4.35 GB，复制后峰值翻倍，
    这是 137 的成因之一。这里按样本分块调用同一个 scaler，内存只有一块大小。
    """
    original_shape = array.shape
    if len(original_shape) < 2:
        raise ValueError(f"expected at least 2 dims, got {original_shape}")

    n_features = getattr(scaler, "n_features_in_", None)
    if spatial and len(original_shape) >= 4:
        feature_dim = int(original_shape[-2] * original_shape[-1])
    else:
        feature_dim = int(original_shape[-1])

    if n_features is not None and int(n_features) != feature_dim:
        raise ValueError(
            f"scaler expects {n_features} features but array layout implies "
            f"{feature_dim} for shape {original_shape}"
        )

    total = int(original_shape[0])
    step = max(1, int(sample_chunk))
    for start in range(0, total, step):
        stop = min(start + step, total)
        block = np.asarray(array[start:stop], dtype=np.float32)
        restored = scaler.inverse_transform(block.reshape(-1, feature_dim))
        np.asarray(array[start:stop])[:] = restored.reshape(block.shape)
    if hasattr(array, "flush"):
        array.flush()
    return array


def predict_split(
    config: Dict[str, Any],
    split: str = "test",
    checkpoint: Optional[str | Path] = None,
    sample_chunk: int = 256,
) -> Dict[str, Any]:
    seed_everything(int(config["seed"]))
    device = resolve_device(config["device"])
    model_dir = model_directory(config)
    tec_scaler, aux_scaler = load_scalers(model_dir)
    model = build_model(config).to(device)
    checkpoint_path = Path(checkpoint or model_dir / "model_state_dict.pth")
    checkpoint_payload, _, _ = load_model_state(
        model,
        checkpoint_path,
        map_location=device,
    )
    del checkpoint_payload

    loader = build_split_loader(
        config,
        split,
        tec_scaler,
        aux_scaler,
        device,
    )

    output_dir = model_dir / "evaluation" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 阶段一：推理，逐块写入磁盘映射。
    raw = run_inference(model, loader, device, dump_dir=output_dir)

    # 反标准化逐块原地完成，不产生第二份完整数组。
    _inverse_transform_inplace(raw["prediction"], tec_scaler, sample_chunk)
    _inverse_transform_inplace(raw["target"], tec_scaler, sample_chunk)
    _inverse_transform_inplace(raw["persistence"], tec_scaler, sample_chunk)
    _inverse_transform_inplace(
        raw["aux_target"], aux_scaler, sample_chunk, spatial=False
    )

    # Windows 不允许删除仍被映射的文件；先把写句柄放下。
    for array in raw.values():
        mmap_handle = getattr(array, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
    # 释放写模式映射的引用后，再以只读方式重新打开。
    del raw
    gc.collect()

    # 阶段二：以只读磁盘映射计算指标，不把整份数组读入内存。
    arrays = load_predictions(output_dir)
    metrics = evaluate_arrays(
        arrays["prediction"],
        arrays["target"],
        arrays["persistence"],
        sample_chunk=sample_chunk,
    )
    metrics_path = save_metrics(metrics, output_dir / "metrics.json")

    return {
        "model_dir": str(model_dir),
        "split": split,
        "metrics": metrics,
        "predictions_path": str(output_dir),
        "metrics_path": str(metrics_path),
        "arrays": arrays,
    }
