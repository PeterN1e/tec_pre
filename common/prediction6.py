import torch
from torch import nn
import numpy as np
from config import TrainConfig
cfg_train = TrainConfig()
class TecPredict(nn.Module):
    def __init__(self,model,test_loader):
        super().__init__()
        self.model = model
        self.test_loader = test_loader
        self.device = cfg_train.device
    def forward(self,frame_num=1):
        """

        :param frame_num: 预测的帧数
        :return: 实际值和预测值
        """
        self.model.eval()
        predictions = []
        actuals = []
        physical_value = []
        with torch.no_grad():
            for batch_in_tec,batch_in_aux,batch_exp_tec,batch_exp_aux in self.test_loader:
                batch_in_tec = batch_in_tec.float().to(self.device) #(batch_size,seq_length,71,73)
                # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux = batch_in_aux.float().to(self.device)
                batch_exp_tec = batch_exp_tec.float().to(self.device)
                batch_exp_aux = batch_exp_aux.float().to(self.device)
                output = self.model(batch_in_tec,batch_in_aux)
                print(f"预测第{frame_num}组")
                frame_num += 1
                #prediction6.py 使用 np.array(delta) 时未先转移到 CPU
                # NumPy 不支持 GPU 数据，要将数据放到cpu上并转化为np数据
                predictions.append(output.cpu().numpy())
                actuals.append(batch_exp_tec.cpu().numpy())
                physical_value.append(batch_exp_aux.cpu().numpy())

        return np.array(predictions),np.array(actuals),np.array(physical_value)

