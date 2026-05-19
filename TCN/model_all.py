from CNNDecoder4 import CnnDecoder
from CNNEncoder2 import CnnEncoder
from tcnModule3 import TCNMiddlePredictor
import torch.nn as nn
class ModelAll(nn.Module):
    def __init__(self,transmit_parameter,out_dim1,
              predict_len,history_len,in_dim2,d_model):
        """

        :param transmit_parameter:
        :param out_dim1:
        :param predict_len:
        :param history_len:
        :param in_dim2:
        :param d_model:
        """
        super().__init__()
        self.transmit_parameter = transmit_parameter
        self.encoder = CnnEncoder(transmit_parameter=transmit_parameter,
                             out_dim=out_dim1)
        self.tcn = TCNMiddlePredictor(input_dim=out_dim1,output_dim= 4104,seq_length= history_len,num_channels=[256, 256, 256],kernel_size=3,dropout=0.2,use_attention=False)
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
        x = self.tcn(x)
        x = self.decoder(x)

        return x

