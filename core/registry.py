from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch.nn as nn

from core.config import load_config
from core.model_api import (
    EDCGConvLSTMAdapter,
    ForecastModel,
    TecAuxForecastModel,
)


ModelBuilder = Callable[[Dict[str, Any]], ForecastModel]
_MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Model already registered: {name}")
        _MODEL_REGISTRY[key] = builder
        return builder

    return decorator


def registered_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def _model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Structural hyperparameters, taken verbatim from the model YAML.

    No code-level defaults live here: a missing key is a config error, not a
    silently substituted value.
    """
    return dict(config.get("model", {}).get("params", {}))


def _io_config(config: Dict[str, Any], *keys: str) -> Dict[str, int]:
    """Shape parameters derived from the data section, never from model YAML."""
    data_cfg = config["data"]
    derived = {
        "input_length": int(data_cfg["input_length"]),
        "output_length": int(data_cfg["output_length"]),
        "aux_dim": len(data_cfg["aux_columns"]),
        "height": int(data_cfg["height"]),
        "width": int(data_cfg["width"]),
    }
    return {key: derived[key] for key in keys}


@register_model("E_P_D")
def build_epd(config: Dict[str, Any]) -> ForecastModel:
    from E_P_D.model_E_P_D import ModelEPD

    io = _io_config(config, "input_length", "output_length", "aux_dim")
    model = ModelEPD(**io, **_model_params(config))
    return TecAuxForecastModel(model)


@register_model("ED_CGConvLSTM")
def build_ed_cg_conv_lstm(config: Dict[str, Any]) -> ForecastModel:
    from ED_CGConvLSTM.ED_CGConvLSTM import EDCGConvLSTM

    io = _io_config(config, "input_length", "output_length")
    model = EDCGConvLSTM(**io, **_model_params(config))
    return EDCGConvLSTMAdapter(model)


@register_model("GA_Predrnn")
def build_ga_predrnn(config: Dict[str, Any]) -> ForecastModel:
    from GA_Predrnn.GA_Predrnn import GAPredrnnPredictor

    io = _io_config(config, "input_length", "output_length")
    model = GAPredrnnPredictor(**io, **_model_params(config))
    return TecAuxForecastModel(model)


@register_model("ED_Autoformer")
def build_ed_autoformer(config: Dict[str, Any]) -> ForecastModel:
    from ED_Autoformer.ED_Autoformer import EDAutoformer

    io = _io_config(config, "input_length", "output_length", "aux_dim")
    params = _model_params(config)
    if "encode_channels" in params:
        params["encode_channels"] = tuple(params["encode_channels"])
    model = EDAutoformer(**io, **params)
    return TecAuxForecastModel(model)


@register_model("ModelCanon")
def build_model_canon(config: Dict[str, Any]) -> ForecastModel:
    from ModelCanon.ModelCanon import ModelCanon

    io = _io_config(
        config, "input_length", "output_length", "aux_dim", "height", "width"
    )
    model = ModelCanon(**io, **_model_params(config))
    return TecAuxForecastModel(model)


def build_model(
    config: Dict[str, Any],
    model_name: Optional[str] = None,
) -> ForecastModel:
    name = model_name or config.get("model", {}).get("name")
    if not name:
        raise ValueError("Model name is missing from config")
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered models: {registered_models()}"
        )
    resolved = dict(config)
    resolved.setdefault("model", {})
    resolved["model"] = dict(resolved["model"])
    resolved["model"]["name"] = name
    model = _MODEL_REGISTRY[key](resolved)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Builder for {name} did not return an nn.Module")
    return model


def build_model_from_name(
    model_name: str,
    overrides: Optional[list[str]] = None,
) -> ForecastModel:
    return build_model(load_config(model_name=model_name, overrides=overrides))
