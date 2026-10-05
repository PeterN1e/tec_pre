from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    scaler: Any = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload
    return {"model_state_dict": payload}


def load_model_state(
    model: torch.nn.Module,
    path: str | Path,
    map_location: Any = "cpu",
    strict: bool = True,
):
    payload = load_checkpoint(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(
        payload["model_state_dict"],
        strict=strict,
    )
    return payload, missing, unexpected


def resume_optimizer_state(
    checkpoint: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
) -> None:
    state = checkpoint.get("optimizer_state_dict")
    if state is not None:
        optimizer.load_state_dict(state)

