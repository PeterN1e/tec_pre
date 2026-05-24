from config import ModelConfig,DatasetConfig,TrainConfig
cfg_model = ModelConfig()
cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()
USE_MODEL = cfg_model.model_name

# 统一导入对应模型
if USE_MODEL == "transformer":
    from Transformer.transformerModule import TecPreTransformer as Model_Predictor
elif USE_MODEL == "tcn":
    from TCN.tcnModule import TCNMiddlePredictor as Model_Predictor
elif USE_MODEL == "convlstm":
    from LSTM.convLSTM.convLSTM import ConvLSTM as Model_Predictor
elif USE_MODEL == "convgru":
    from Gru.convGRU import ConvGRU as Model_Predictor
else:
    raise ValueError("模型不存在")

# 对外统一暴露一个类名：MyModel
__all__ = ["Model_Predictor"]