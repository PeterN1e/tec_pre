import torch
from torch import nn
import numpy as np
class TecPredict(nn.Module):
    def __init__(self,model,test_loader):
        super().__init__()
        self.model = model
        self.test_loader = test_loader
    def forward(self,frame_num=24):
        """

        :param frame_num: 预测的帧数
        :return: 实际值和预测值
        """
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for batch_in_tec,batch_in_aux,batch_exp in self.test_loader:
                batch_in_tec = batch_in_tec.float().cuda()  # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux = batch_in_aux.float().cuda()
                batch_exp = batch_exp.float().cuda()
                output = self.model(batch_in_tec,batch_in_aux)
                print(f"预测第{frame_num}帧")
                predictions.append(output.cpu().numpy())
                actuals.append(batch_exp.cpu().numpy())
                if frame_num==1:
                    break
                frame_num-=1
        return np.array(predictions),np.array(actuals)

