from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from common.EvaluationMetrics import (
    evaluate_by_latitude_band,
    evaluate_per_step,
    evaluate_with_baseline,
)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    predictions = []
    actuals = []
    aux_actuals = []
    persistence = []

    for tec_in, aux_in, tec_gt, aux_gt in loader:
        tec_in = tec_in.float().to(device)
        aux_in = aux_in.float().to(device)
        output = model(tec_in, aux_in)
        if isinstance(output, tuple):
            output = output[0]
        predictions.append(output.detach().cpu().numpy())
        actuals.append(tec_gt.numpy())
        aux_actuals.append(aux_gt.numpy())
        baseline = tec_in[:, -1:, :, :].repeat(
            1,
            output.shape[1],
            1,
            1,
        )
        persistence.append(baseline.detach().cpu().numpy())

    return {
        "prediction": np.concatenate(predictions, axis=0),
        "target": np.concatenate(actuals, axis=0),
        "aux_target": np.concatenate(aux_actuals, axis=0),
        "persistence": np.concatenate(persistence, axis=0),
    }


def evaluate_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    persistence: np.ndarray,
    latitude_bands: int = 6,
) -> Dict[str, Any]:
    return {
        "aggregate": evaluate_with_baseline(
            prediction,
            target,
            persistence,
        ),
        "per_step": evaluate_per_step(prediction, target),
        "latitude_bands": evaluate_by_latitude_band(
            prediction,
            target,
            bands=latitude_bands,
        ),
    }


def save_metrics(metrics: Dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            metrics,
            handle,
            indent=2,
            ensure_ascii=False,
            default=lambda value: (
                value.item() if hasattr(value, "item") else str(value)
            ),
        )
    return target
