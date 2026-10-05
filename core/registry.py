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
    return dict(config.get("model", {}).get("params", {}))


def _io_config(config: Dict[str, Any]) -> Dict[str, int]:
    data_cfg = config["data"]
    return {
        "input_length": int(data_cfg["input_length"]),
        "output_length": int(data_cfg["output_length"]),
        "aux_dim": len(data_cfg["aux_columns"]),
        "height": int(data_cfg.get("height", 71)),
        "width": int(data_cfg.get("width", 73)),
    }


@register_model("E_P_D")
def build_epd(config: Dict[str, Any]) -> ForecastModel:
    from E_P_D.model_E_P_D import ModelEPD

    io = _io_config(config)
    params = _model_params(config)
    model = ModelEPD(
        transmit_parameter=params.get("transmit_parameter", 3),
        input_length=io["input_length"],
        output_length=io["output_length"],
        aux_dim=io["aux_dim"],
        predictor_name=params.get("predictor", "convlstm"),
    )
    return TecAuxForecastModel(model)


@register_model("ED_CGConvLSTM")
def build_ed_cg_conv_lstm(config: Dict[str, Any]) -> ForecastModel:
    from ED_CGConvLSTM.ED_CGConvLSTM import EDCGConvLSTM

    io = _io_config(config)
    params = _model_params(config)
    model = EDCGConvLSTM(
        input_dim=params.get("input_dim", 1),
        hidden_dim=params.get("hidden_dim", 60),
        output_dim=params.get("output_dim", 1),
        num_layers=params.get("num_layers", 4),
        kernel_size=params.get("kernel_size", 3),
        input_length=io["input_length"],
        output_length=io["output_length"],
        use_checkpoint=params.get("use_checkpoint"),
        use_torch_compile=params.get("use_torch_compile"),
    )
    return EDCGConvLSTMAdapter(model)


@register_model("GA_Predrnn")
def build_ga_predrnn(config: Dict[str, Any]) -> ForecastModel:
    from GA_Predrnn.GA_Predrnn import GAPredrnnPredictor

    io = _io_config(config)
    params = _model_params(config)
    model = GAPredrnnPredictor(
        input_dim=params.get("input_dim", 4),
        hidden_dim=params.get("hidden_dim", 64),
        num_layers=params.get("num_layers", 3),
        kernel_size=params.get("kernel_size", 5),
        input_length=io["input_length"],
        output_length=io["output_length"],
        aux_dim=params.get("aux_output_dim", 3),
        block_size=params.get("block_size", 8),
        halo_size=params.get("halo_size", 2),
        num_heads=params.get("num_heads", 4),
        aux_indices=params.get("aux_indices", (2, 3, 4)),
    )
    return TecAuxForecastModel(model)


@register_model("ED_Autoformer")
def build_ed_autoformer(config: Dict[str, Any]) -> ForecastModel:
    from ED_Autoformer.ED_Autoformer import EDAutoformer

    io = _io_config(config)
    params = _model_params(config)
    encode_channels = params.get("encode_channels")
    if encode_channels is not None:
        encode_channels = tuple(encode_channels)
    model = EDAutoformer(
        input_length=io["input_length"],
        output_length=io["output_length"],
        aux_dim=io["aux_dim"],
        d_model=params.get("d_model", 512),
        n_heads=params.get("n_heads", 8),
        d_ff=params.get("d_ff", 2048),
        e_layers=params.get("e_layers", 2),
        d_layers=params.get("d_layers", 1),
        moving_avg=params.get("moving_avg", 13),
        factor=params.get("factor", 3),
        dropout=params.get("dropout", 0.05),
        activation=params.get("activation", "gelu"),
        label_len=params.get("label_len"),
        encode_channels=encode_channels,
    )
    return TecAuxForecastModel(model)


@register_model("ModelCanon")
def build_model_canon(config: Dict[str, Any]) -> ForecastModel:
    from ModelCanon.ModelCanon import ModelCanon

    io = _io_config(config)
    params = _model_params(config)
    model = ModelCanon(
        input_length=io["input_length"],
        output_length=io["output_length"],
        aux_dim=io["aux_dim"],
        height=io["height"],
        width=io["width"],
        d_model=params.get("d_model", 256),
        n_heads=params.get("n_heads", 8),
        e_layers=params.get("e_layers", 4),
        decoder_layers=params.get("decoder_layers", 2),
        d_ff=params.get("d_ff", 1024),
        dropout=params.get("dropout", 0.05),
        patch_size=params.get("patch_size", 4),
    )
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
