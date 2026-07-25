from config import EDCGConvLSTMConfig, EPDConfig
from ED_CGConvLSTM.ED_CGConvLSTM import ED_CGConvLSTM
from E_P_D.model_E_P_D import ModelEPD


# =========================================================
#  注册表：每个模型名 → (模型类, 模型专属配置)
# =========================================================
MODEL_REGISTRY = {
    "ModelA": {
        "model_class": ED_CGConvLSTM,
        "model_cfg": EDCGConvLSTMConfig(),       # 默认参数
    },
    "ModelB": {
        "model_class": ModelEPD,
        "model_cfg": EPDConfig(),
    },
    # "ModelC": {
    #     "model_class": ModelC,
    #     "model_cfg": ModelCConfig(),
    # },
}


def build_model(model_name: str, train_cfg, dataset_cfg, overrides: dict = None):
    """
    统一模型构建接口

    Args:
        model_name:  "ModelA" / "ModelB" / "ModelC"
        train_cfg:   TrainConfig 实例
        dataset_cfg: DatasetConfig 实例
        overrides:   可选，覆盖模型默认参数 {"hidden_dim": 512, ...}

    Returns:
        已放到 device 上的模型实例
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}")

    entry = MODEL_REGISTRY[model_name]
    model_class = entry["model_class"]
    model_cfg = entry["model_cfg"]

    # 用 overrides 覆盖默认参数
    if overrides:
        for k, v in overrides.items():
            if not hasattr(model_cfg, k):
                raise KeyError(f"{model_name} 没有参数 '{k}'")
            setattr(model_cfg, k, v)

    # ---- 把所有配置合并为一个 dict 传给模型 ----
    params = {
        "transmit_parameter": model_cfg.transmit_parameter,
        "history_len": train_cfg.seq_length,
        "predict_len": train_cfg.pred_length,
        "aux_dim": dataset_cfg.aux_dim,
        "channel": model_cfg.channel,
    }
    # 追加模型专属参数（去掉已放入公共字段的）
    for k, v in vars(model_cfg).items():
        if k not in ("transmit_parameter", "channel"):
            params[k] = v

    model = model_class(**params).to(train_cfg.device)
    return model