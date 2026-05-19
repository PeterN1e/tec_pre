from CNNDecoder4 import CnnDecoder
from CNNEncoder2 import CnnEncoder
from LSTM.convLSTM.convLSTM3 import ConvLSTM
import torch.nn as nn
class ModelAll(nn.Module):
    def __init__(self,transmit_parameter,out_dim1,
              predict_len,history_len,in_channel2,hidden_channel2):
        """

        :param transmit_parameter:
        :param out_dim1:
        :param predict_len:
        :param history_len:

        """
        super().__init__()
        self.history_len = history_len
        self.transmit_parameter = transmit_parameter
        self.encoder = CnnEncoder(transmit_parameter=transmit_parameter,
                             out_dim=out_dim1)
        self.predictor = ConvLSTM(in_channels=in_channel2,
                                    hidden_channels =hidden_channel2,
                                    pred_len=predict_len)
        self.decoder = CnnDecoder(transmit_parameter_de=transmit_parameter)
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
        x = self.encoder(tec24,aux)
        x = self.predictor(x)
        x = self.decoder(x)

        return x

