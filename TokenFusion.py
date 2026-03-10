
def feature_fusion1(tec_data, aux_data):
    """
    用于融合tec图像特征和辅助信息特征
    :param tec_data: (batch,seq_length,4104)
    :param aux_data: (batch,seq_length,4)
    :return:
    """
    batch_size = tec_data.shape[0]