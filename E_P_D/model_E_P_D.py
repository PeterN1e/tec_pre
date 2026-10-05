from E_P_D.CoordGate.CoordGateEncoder2 import CnnEncoder
from E_P_D.CoordGate.CoordGateDecoder4 import CnnDecoder
from config import TrainConfig,EPDConfig,DatasetConfig
from common.DataFusion import FilmFusion
cfg_train = TrainConfig()
EPD_cfg = EPDConfig()
DatasetCfg = DatasetConfig()
import torch.nn as nn


def _build_predictor(
    predictor_name,
    input_length,
    output_length,
    encoder_channels,
):
    spatial_shape = (18, 19)
    flat_dim = encoder_channels * spatial_shape[0] * spatial_shape[1]
    if predictor_name == "transformer":
        from E_P_D.PredictionModel.Transformer.transformerModule import (
            TecPreTransformer,
        )
        return TecPreTransformer(
            history_len=input_length,
            predict_len=output_length,
            in_dim=flat_dim,
            output_dim=flat_dim,
            output_channels=encoder_channels,
            spatial_shape=spatial_shape,
        )
    if predictor_name == "tcn":
        from E_P_D.PredictionModel.TCN.tcnModule import TCNMiddlePredictor
        return TCNMiddlePredictor(
            input_dim=flat_dim,
            output_dim=flat_dim,
            predict_len=output_length,
            history_len=input_length,
            output_channels=encoder_channels,
            spatial_shape=spatial_shape,
        )
    if predictor_name == "convlstm":
        from E_P_D.PredictionModel.LSTM.convLSTM.convLSTM import ConvLSTM
        return ConvLSTM(
            history_len=input_length,
            in_channels=encoder_channels,
            hidden_channels=encoder_channels,
            predict_len=output_length,
        )
    if predictor_name == "convgru":
        from E_P_D.PredictionModel.Gru.convGRU import ConvGRU
        return ConvGRU(
            in_channels=encoder_channels,
            hidden_channels=encoder_channels,
            history_len=input_length,
            predict_len=output_length,
        )
    raise ValueError(f"Unknown E_P_D predictor: {predictor_name}")


class ModelEPD(nn.Module):
    def __init__(self,
                 transmit_parameter = EPD_cfg.transmit_parameter,
                 input_length = cfg_train.input_length,
                 output_length = cfg_train.output_length,
                 aux_dim = DatasetCfg.aux_dim,
                 predictor_name = EPD_cfg.EPDmodel_name,
                 ):
        """
        :param transmit_parameter:
        :param output_length:
        :param input_length:
        """
        super().__init__()
        self.predictor_name = predictor_name
        self.encoder = CnnEncoder(transmit_parameter = transmit_parameter)
        self.predictor = _build_predictor(
            predictor_name=predictor_name,
            input_length=input_length,
            output_length=output_length,
            encoder_channels=transmit_parameter * 4,
        )
        self.decoder = CnnDecoder(transmit_parameter_de = transmit_parameter)
        self.Fusion = FilmFusion(
            aux_dim=aux_dim,
            channel=transmit_parameter * 4,
        )
    def forward(self,tec,aux):
        """
        :param tec: (batch_size,seq_length,71,73)
        :param aux: (batch_size,seq_length,4)
        :return:
        """
        x = self.encoder(tec)
        x = self.Fusion(feat_map = x,aux = aux)
        if self.predictor_name in {"transformer", "tcn"}:
            x = x.flatten(2)
        x = self.predictor(x)
        x = self.decoder(x)

        return x

