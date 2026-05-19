import numpy as np  #dd
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class CnnEncoder(nn.Module):
    """
    输入为（batch,seq_length,71,73）
    单个batch内具有时间顺序，倘若直接进行二维卷积
    会破环原有时间顺序，破坏了数据：

    解决：改变(reshape)维度为（batch*seq_length,1,71,73）进行计算
    最后在
    """
    def __init__(self,transmit_parameter,out_dim):
        super().__init__()
        self.transmit_parameter = transmit_parameter
        self.Tec_encoder = nn.Sequential(  #把几个层包装成块
            # nn.Sequential内：每个nn.Conv2d都是独立实例

            nn.Conv2d(in_channels=1, out_channels = transmit_parameter, padding=0, kernel_size=1, stride=1),
            # (B,C,H,W):(1,1,71,73)
            nn.ReLU(),#当网络特别深、特征图很大（如3D医学图像、高分辨率 TEC 地图）
            # ，显存吃紧用inplace=True  默认为False
            nn.Conv2d(in_channels = transmit_parameter, out_channels=transmit_parameter*2, padding=1, kernel_size=3, stride=2),
            #(1,64*2,36,37)
            nn.ReLU(),
            nn.Conv2d(transmit_parameter * 2, transmit_parameter * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),#(18,19)
            #nn.AvgPool2d(2)#(9,9)
            #nn.AdaptiveAvgPool2d()  #自适应池化操作
        )
        self.fc_tec = nn.Linear(4 * 18 * 19 * transmit_parameter,512)
        #self.fc_aux = nn.Linear(4,out_dim-1400)
    def forward(self,tec,aux):
        """
        :param tec: (24,24,460)
        :param aux: (24,24,out_dim-460)
        :return:
        """
        batch_size = tec.size(0)
        seq_length = tec.size(1)
        tec = tec.reshape(batch_size*seq_length,1,71,73)
        #（batch,seq_length,71,73）->（batch * seq_length,1,71,73）

        tec = self.Tec_encoder(tec)

        #（batch * seq_length,1,71,73）->（batch * seq_length,4 * transmit_parameter * 18 * 19）
        tec = tec.view(batch_size * seq_length,1,4 * self.transmit_parameter * 18 * 19)
        aux_dim = aux.shape[2]
        tec = tec[:,:,0:-aux_dim]#直接裁掉最后4个
        tec = tec.view(batch_size,seq_length,-1)
        x = torch.cat((aux,tec),dim = -1)
        x = self.fc_tec(x)
        """
        view()相当于reshape、resize，重新调整Tensor的形状
        """

        return x



#############测试
# if __name__ == '__main__':
#
#     dummy = torch.randn(1,24,71,73)
#     aux = torch.randn(1,24,4)
#     enc = CnnEncoder(transmit_parameter = 48,out_dim =500 )
#     vec = enc(dummy,aux)
#     print(vec)
#     print(vec.shape)



