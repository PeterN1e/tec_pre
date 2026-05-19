"""
分清nn.Conv2d中stride和Dilated的区别
"""
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
        # test_out = self.tec_decoder(test)
        # test_out1 = test_out
        x = self.tec_decoder(x)
        # (batch_size,1,71,73)

        x = x.squeeze(1)
        # (batch_size,71,73)
        return x

if __name__ == '__main__':
    test_a = CnnDecoder(transmit_parameter_de=3)
    test = torch.randn(batch_size, 12, 18, 19)
    test_a(test)