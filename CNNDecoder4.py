"""
分清nn.Conv2d中stride和Dilated的区别
"""

import torch
import torch.nn as nn
class CnnDecoder(nn.Module):
    def __init__(self,transmit_parameter_de=1):
        super().__init__()
        self.tec_decoder = nn.Sequential(
            #nn.Linear(in_dim, out_dim),
            nn.ConvTranspose2d(in_channels=transmit_parameter_de*4, out_channels = transmit_parameter_de*2, padding=1, kernel_size=3, stride=2,output_padding=(1,0)),
            #(n,4c,18,19)->(n,2c,36,37)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=transmit_parameter_de*2, out_channels=transmit_parameter_de, kernel_size=3, stride=2, padding=1),
            #(n,2c,36,37)->(n,c,71,73)
            nn.ReLU(),
            nn.Conv2d(in_channels=transmit_parameter_de, out_channels=1, kernel_size=1, stride=1, padding=0)
            #(n,c,71,73)->(n,c,71,73)

        )
        self.transmit_parameter_de = transmit_parameter_de
    def forward(self,x):
        x = x.reshape(x.shape[0], self.transmit_parameter_de * 4, 18, 19)
        x = self.tec_decoder(x)
        return x#(batch_size,1,71,73)

# if __name__ == '__main__':
#     transmit_parameter=1
#     de = nn.ConvTranspose2d(in_channels=transmit_parameter*4, out_channels = transmit_parameter*2, padding=1, kernel_size=3, stride=2)
#     a = de(torch.randn((1,transmit_parameter*4,18,19)))
#     print(a.shape)
#     de1 = CnnDecoder(transmit_parameter)
#     print(de1(torch.randn((1,transmit_parameter*4,18,19))).shape)
