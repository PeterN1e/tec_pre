import torch.nn as nn
from config import EPDConfig,DatasetConfig,TrainConfig,model_name
cfg_EPD_model = EPDConfig()
cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()
EPD_PredictMODEL = cfg_EPD_model.EPDmodel_name
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

if model_name == "E_P_D":
    from E_P_D.model_E_P_D import ModelEPD as Model
elif model_name == "ED_CGConvLSTM":
    from ED_CGConvLSTM.ED_CGConvLSTM import EDCGConvLSTM as Model
else:
    raise ValueError("模型不存在")

class ModelAll(nn.Module):
    def __init__(self,model_name = model_name):
        super().__init__()
        self.model = Model()
        self.model_name = model_name
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():  # 遍历模型内所有参数
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self,tec,aux):
        if self.model_name == "E_P_D":
            x = self.model(tec,aux)
        elif self.model_name == "ED_CGConvLSTM":
            x = tec.unsqueeze(2)
            x = self.model(x)
            x = x.squeeze(2)
        else:
            raise ValueError("model_selector.py,模型选择错误")
        return x

# 对外统一暴露类名：
__all__ = ["EPD_Predictor","ModelAll"]