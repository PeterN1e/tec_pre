# from common.CNNDecoder4 import CnnDecoder
# from common.CNNEncoder2 import CnnEncoder
from E_P_D.CoordGate.CoordGateEncoder2 import CnnEncoder
from E_P_D.CoordGate.CoordGateDecoder4 import CnnDecoder
from model_selector import EPD_Predictor
from config import TrainConfig,EPDConfig,DatasetConfig
from common.DataFusion import FilmFusion
cfg_train = TrainConfig()
EPD_cfg = EPDConfig()
DatasetCfg = DatasetConfig()
import torch.nn as nn
class ModelEPD(nn.Module):
    def __init__(self,
                 transmit_parameter = EPD_cfg.transmit_parameter,
                 input_length = cfg_train.input_length,
                 output_length = cfg_train.output_length,
                 aux_dim = DatasetCfg.aux_dim,
                 ):
        """
        :param transmit_parameter:
        :param output_length:
        :param input_length:
        """
        super().__init__()
        self.encoder = CnnEncoder(transmit_parameter = transmit_parameter)
        self.predictor = EPD_Predictor(history_len = input_length,
                                         predict_len = output_length)
        self.decoder = CnnDecoder(transmit_parameter_de = transmit_parameter)
        self.Fusion = FilmFusion(aux_dim = aux_dim)
    def forward(self,tec,aux):
        """
        :param tec: (batch_size,seq_length,71,73)
        :param aux: (batch_size,seq_length,4)
        :return:
        """
        x = self.encoder(tec)
        x = self.Fusion(feat_map = x,aux = aux)
        x = self.predictor(x)
        x = self.decoder(x)

        return x

