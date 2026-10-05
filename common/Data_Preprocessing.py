import numpy as np


def _require_dims(data, allowed):
    if data.ndim not in allowed:
        raise ValueError(
            f"Expected array with dimensions {allowed}, got shape {data.shape}"
        )

def scale_tec_aux_data(data, scaler, fit_scaler=True):
    """
    将tec数据进行降维 步骤：reshape → 缩放 → 恢复形状
    :param data: tec数据或辅助特征aux
    :param scaler: 创建实例后的标准化器（MinMaxScaler/StandardScaler 等）
    :param fit_scaler: True则fit_transform（训练集），False则transform（测试集）
    :return:
    """
    _require_dims(data, {2, 3})
    if data.ndim == 3:
        num = data.shape[0]
        data_2d = data.reshape(num, -1)
        if fit_scaler:
            scaled = scaler.fit_transform(data_2d)
        else:
            scaled = scaler.transform(data_2d)
        return scaled.reshape(data.shape)
    if data.ndim == 2:
        data_2d = data
        if fit_scaler:
            scaled = scaler.fit_transform(data_2d)
        else:
            scaled = scaler.transform(data_2d)
        return scaled
    raise AssertionError("unreachable")

def inverse_transform_predictions(data,scaler):
    """
    作用：对数据进行反标准化，还原至输入状态
    :param data:
    :param scaler:设定的标准化
    :return:
    """
    #predictions:由[24,71,73]构成的列表
    #actual[24,71,73]构成的列表
    data_inv = []
    #act_inv = []#创建的是列表
    dim = len(data.shape)
    if dim not in {3, 4, 5}:
        raise ValueError(f"Expected a 3D, 4D or 5D array, got shape {data.shape}")
    original_shape = data.shape
    n_features = getattr(scaler, "n_features_in_", None)
    if dim == 3:
        feature_dim = original_shape[-1]
        data_2d = data.reshape(-1, feature_dim)
        return scaler.inverse_transform(data_2d).reshape(original_shape)
    spatial_features = original_shape[-2] * original_shape[-1]
    if dim == 5 or (n_features == spatial_features and n_features != original_shape[-1]):
        feature_dim = spatial_features
    elif n_features == original_shape[-1]:
        feature_dim = original_shape[-1]
    else:
        raise ValueError(
            "Cannot infer inverse-transform layout for "
            f"shape {original_shape} and scaler with {n_features} features"
        )
    data_2d = data.reshape(-1, feature_dim)
    data_inv = scaler.inverse_transform(data_2d).reshape(original_shape)
    return data_inv
