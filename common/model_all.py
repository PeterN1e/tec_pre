# from common.CNNDecoder4 import CnnDecoder
# from common.CNNEncoder2 import CnnEncoder
from common.CoordGate.CoordGateEncoder2 import CnnEncoder
from common.CoordGate.CoordGateDecoder4 import CnnDecoder
from model_selector import Model_Predictor
from config import TrainConfig
from common.TokenFusion import FilmFusion
cfg_train = TrainConfig()
import torch.nn as nn
class ModelAll(nn.Module):
    def __init__(self,
                 transmit_parameter,
                 predict_len,
                 history_len,
                 aux_dim,
                 channel):
        """
        :param transmit_parameter:
        :param predict_len:
        :param history_len:
        """
        super().__init__()
        self.encoder = CnnEncoder(transmit_parameter = transmit_parameter)
        self.predictor = Model_Predictor(history_len = history_len,
                                         predict_len = predict_len)
        self.decoder = CnnDecoder(transmit_parameter_de = transmit_parameter)
        self.Fusion = FilmFusion(aux_dim = aux_dim,channel = channel)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():#遍历模型内所有参数
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward(self,tec24,aux):
        """
        :param tec24: (batch_size,seq_length,71,73)
        :param aux: (batch_size,seq_length,4)
        :return:
        """
        x = self.encoder(tec24)
        x = self.Fusion(feat_map = x,aux = aux)
        x = self.predictor(x)
        x = self.decoder(x)

        return x

