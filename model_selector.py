from __future__ import annotations

from typing import Optional

import torch.nn as nn

from core.config import load_config
from core.registry import build_model


def _legacy_model_name() -> str:
    from config import model_name

    return model_name


def _legacy_registry_overrides(model_name: str) -> list[str]:
    if model_name != "E_P_D":
        return []
    from config import EPDConfig

    return [f"model.params.predictor={EPDConfig().EPDmodel_name}"]


class ModelAll(nn.Module):
    """Backward-compatible adapter around the config-driven model registry."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        name = model_name or _legacy_model_name()
        if model is None:
            config = load_config(
                model_name=name,
                overrides=_legacy_registry_overrides(name),
            )
            model = build_model(config)
        self.model = model
        self.model_name = name

    def forward(self, tec, aux):
        return self.model(tec, aux)

    def train_forward(self, tec, aux):
        train_forward = getattr(self.model, "train_forward", None)
        if train_forward is not None:
            return train_forward(tec, aux)
        return self.model(tec, aux), None


__all__ = ["ModelAll", "build_model"]
