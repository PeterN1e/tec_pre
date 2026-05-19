"""
分清nn.Conv2d中stride和Dilated的区别
"""
from numpy.ma.core import concatenate
import numpy as np
from config import batch_size
import torch
import torch.nn as nn
class CnnDecoder(nn.Module):
    def __init__(self,transmit_parameter_de=3):
        super().__init__()
        self.tec_decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=transmit_parameter_de*4, out_channels = transmit_parameter_de*2, padding=1, kernel_size=3, stride=2,output_padding=(1,0)),

            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=transmit_parameter_de*2, out_channels=transmit_parameter_de, kernel_size=3, stride=2, padding=1),

            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=transmit_parameter_de, out_channels=1, kernel_size=1, stride=1, padding=0),


        )
        self.transmit_parameter_de = transmit_parameter_de
    def forward(self,x):
        x_pred = None
        if x.dim() == 4:#(B,channels,h,w)
            x = self.tec_decoder(x)#(B,1,h,w)
            x = x.squeeze(1)
        elif x.dim() == 5:
            for i in range(x.size(1)):#(B,pred,channels,h,w)
                x_cell = x[:,i,:]
                x_cell = self.tec_decoder(x_cell)#(B,pred,1,h,w)
                if x_pred is not None:
                    x_pred = x_pred.concat(x_cell)
                else:
                    x_pred = x_cell
            x = x_pred.squeeze(1)
        else:
            print("解码器输入维度错误")
        return x

if __name__ == '__main__':
    test_a = CnnDecoder(transmit_parameter_de=3)
    test = torch.randn(batch_size, 12,12, 18, 19)
    test_a(test)