from config import ModelConfig,DatasetConfig,TrainConfig
cfg_model = ModelConfig()
cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()
EPD_PredictMODEL = cfg_model.EPDmodel_name
MODEL = cfg_model.model_name
# 统一导入对应模型
if EPD_PredictMODEL == "transformer":
    from E_P_D.PredictionModel.Transformer.transformerModule import TecPreTransformer as EPD_Predictor
elif EPD_PredictMODEL == "tcn":
    from E_P_D.PredictionModel.TCN.tcnModule import TCNMiddlePredictor as EPD_Predictor
elif EPD_PredictMODEL == "convlstm":
    from E_P_D.PredictionModel.LSTM.convLSTM.convLSTM import ConvLSTM as EPD_Predictor
elif EPD_PredictMODEL == "convgru":
    from E_P_D.PredictionModel.Gru.convGRU import ConvGRU as EPD_Predictor
else:
    raise ValueError("模型不存在")

if MODEL == "E_P_D":
    from E_P_D.model_E_P_D import ModelEPD as Model
elif MODEL == "ED_CGConvLSTM":
    from ED_CGConvLSTM.ED_CGConvLSTM import ED_CGConvLSTM as Model
else:
    raise ValueError("模型不存在")

# 对外统一暴露类名：
__all__ = ["EPD_Predictor","Model"]