from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.Data_Preprocessing import inverse_transform_predictions
from core.checkpoint import load_model_state
from core.config import (
    model_slug,
    resolve_data_paths,
    resolve_output_dir,
)
from core.evaluation import evaluate_arrays, run_inference, save_metrics
from core.registry import build_model
from core.trainer import resolve_device, seed_everything
from data.tec_dataset import TecIonosphereDataset


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
    month_range = data_cfg["splits"][split]
    if isinstance(month_range, int):
        start_month = end_month = month_range
    else:
        start_month, end_month = month_range
    dataset = TecIonosphereDataset(
        tec_dir=paths["tec_dir"],
        indices_dir=paths["indices_dir"],
        start_month=int(start_month),
        end_month=int(end_month),
        input_length=int(data_cfg["input_length"]),
        output_length=int(data_cfg["output_length"]),
        is_train=False,
        tec_scaler=tec_scaler,
        aux_scaler=aux_scaler,
        aux_columns=data_cfg["aux_columns"],
    )
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )


def predict_split(
    config: Dict[str, Any],
    split: str = "test",
    checkpoint: Optional[str | Path] = None,
) -> Dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
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
    raw = run_inference(model, loader, device)
    prediction = inverse_transform_predictions(raw["prediction"], tec_scaler)
    target = inverse_transform_predictions(raw["target"], tec_scaler)
    persistence = inverse_transform_predictions(
        raw["persistence"],
        tec_scaler,
    )
    aux_target = inverse_transform_predictions(
        raw["aux_target"],
        aux_scaler,
    )

    metrics = evaluate_arrays(prediction, target, persistence)
    output_dir = model_dir / "evaluation" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions.npz",
        prediction=prediction,
        target=target,
        persistence=persistence,
        aux_target=aux_target,
    )
    metrics_path = save_metrics(metrics, output_dir / "metrics.json")
    return {
        "model_dir": str(model_dir),
        "split": split,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
        "predictions_path": str(output_dir / "predictions.npz"),
        "arrays": {
            "prediction": prediction,
            "target": target,
            "persistence": persistence,
        },
    }

