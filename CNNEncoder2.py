import numpy as np  #dd
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from config import batch_size
class CnnEncoder(nn.Module):
    """
    利用卷积网络对tec灰度图进行特征提取，实际上就是编码
    输入为（batch，seq_length,71,73）
    但是，con2d只能传递4个维度的数据，无法容纳“频道”和”步长“参数了、
    所以，将时间步转化为频道信号
    """
    def __init__(self,transmit_parameter,out_dim):
        super().__init__()
        self.Tec_encoder = nn.Sequential(  #把几个层包装成块
            nn.Conv2d(in_channels=24, out_channels = transmit_parameter, padding=0, kernel_size=1, stride=1),
            # (B,C,H,W):(1,1,71,73)
            nn.ReLU(),#当网络特别深、特征图很大（如 3D 医学图像、高分辨率 TEC 地图）
            # ，显存吃紧用inplace=True  默认为False
            nn.Conv2d(in_channels = transmit_parameter, out_channels=transmit_parameter*2, padding=1, kernel_size=3, stride=2),
            #(1,64*2,36,37)
            nn.ReLU(),
            nn.Conv2d(transmit_parameter * 2, transmit_parameter * 4, kernel_size=3, stride=2, padding=1),
            #(1,64*4,18,19)
            nn.ReLU(),
            nn.AvgPool2d(2)
            #[1, , 9, 9]
            #nn.AdaptiveAvgPool2d()  #自适应池化操作
        )
        self.fc_tec = nn.Linear(2 * 4 * 9 * 9,460)
        self.fc_aux = nn.Linear(4,out_dim-460)
    def forward(self,tec,aux):
        """
        :param tec: (24,24,460)
        :param aux: (24,24,out_dim-460)
        :return:
        """
        tec = self.Tec_encoder(tec)

        tec = tec.view(batch_size,24,2 * 4 * 9 * 9)

        tec = self.fc_tec(tec)

        aux = self.fc_aux(aux)

        # print("aux:", aux.shape)
        x = torch.cat((aux,tec),-1)
        """
        view()相当于reshape、resize，重新调整Tensor的形状
        """

        return x



#############测试
# if __name__ == '__main__':
#     dummy = torch.randn(1,24,71,73)
#     aux = torch.randn(1,24,3)
#     enc = CnnEncoder(transmit_parameter = 48,out_dim =500 )
#     vec = enc(dummy,aux)
#     print(vec)
#     print(vec.shape)


