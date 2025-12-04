import numpy as np  #dd
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
class CnnEncoder(nn.Module):
    """
    利用卷积网络对tec灰度图进行特征提取，实际上就是编码
    输入为二维数组
    """
    def __init__(self,transmit_parameter,out_dim):
        super().__init__()
        self.Tec_encoder = nn.Sequential(  #把几个层包装成块
            nn.Conv2d(in_channels=1, out_channels = transmit_parameter, padding=0, kernel_size=1, stride=1),
            # (B,C,H,W):(1,1,71,73)
            nn.ReLU(),#当网络特别深、特征图很大（如 3D 医学图像、高分辨率 TEC 地图）
            # ，显存吃紧用inplace=True  默认为False
            nn.Conv2d(in_channels = transmit_parameter, out_channels=transmit_parameter*2, padding=1, kernel_size=3, stride=2),
            #(1,64*2,36,37)
            nn.ReLU(),
            nn.Conv2d(transmit_parameter * 2, transmit_parameter * 4, kernel_size=3, stride=2, padding=1),
            #(1,64*4,18,19)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  #自适应池化操作
          #  nn.flat
        )
        self.fc = nn.Linear(transmit_parameter * 4,out_dim)
    def forward(self,x):
        x = self.Tec_encoder(x)
        # print(x.shape)
        # exit()
        x = self.fc(x.view(-1,256))
        return x

#############测试
if __name__ == '__main__':
    dummy = torch.randn(1,1,71,73)
    enc = CnnEncoder(transmit_parameter = 64,out_dim =128 )
    vec = enc(dummy)
    print(vec)


