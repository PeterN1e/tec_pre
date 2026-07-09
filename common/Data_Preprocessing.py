import numpy as np

def scale_tec_aux_data(data, scaler, fit_scaler=True):
    """
    将tec数据进行降维 步骤：reshape → 缩放 → 恢复形状
    :param data: tec数据或辅助特征aux
    :param scaler: 创建实例后的MinMaxScaler
    :param fit_scaler: True则fit_transform（训练集），False则transform（测试集）
    :return:
    """
    dim = data.ndim
    if dim == 3:
        num, w, h = data.shape
        data_2d = data.reshape(num, 5183)
        if fit_scaler:
            scaled = scaler.fit_transform(data_2d)
        else:
            scaled = scaler.transform(data_2d)
        return scaled.reshape(num, w, h)
    elif dim == 2:#标准化二维特征数据
        #data_2d = data[:,1:]
        data_2d = data
        if fit_scaler:
            scaled = scaler.fit_transform(data_2d)
        else:
            scaled = scaler.transform(data_2d)
        #return np.concatenate((data[:,0].reshape(-1,1),scaled),axis = 1)
        return scaled
    else:
        print("输入维度错误，检查")
        exit()

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
    dim = data.ndim
    if dim == 5:#说明传入的数据是 标准化后的tec图
        original_shape = data.shape
        data_2d = data.reshape(-1,5183)#将数据转化为一行  5183 = 71*73
        data_inv=scaler.inverse_transform(data_2d).reshape(original_shape)
    elif dim == 4:#说明传入的数据是 特征参数
        original_shape = data.shape
        data_2d = data.reshape(-1,original_shape[-1])
        data_inv = scaler.inverse_transform(data_2d).reshape(original_shape)
    else:
        print("反标准化时参数维度传入错误")
    return data_inv