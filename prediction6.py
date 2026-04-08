import torch
from torch import nn
import numpy as np
from config import device
class TecPredict(nn.Module):
    def __init__(self,model,test_loader):
        super().__init__()
        self.model = model
        self.test_loader = test_loader
    def forward(self,frame_num=1):
        """

        :param frame_num: 预测的帧数
        :return: 实际值和预测值
        """
        self.model.eval()
        predictions = []
        actuals = []
        datetime = []
        with torch.no_grad():
            for batch_in_tec,batch_in_aux,batch_exp in self.test_loader:
                batch_in_tec = batch_in_tec.float().to(device) #(batch_size,seq_length,71,73)
                # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux_no_date = batch_in_aux[:,:,1:].float().to(device)
                batch_exp_tec = batch_exp[0].float().to(device)#(batch_size,71,73)
                batch_exp_aux = batch_exp[1].float().to(device)  # (batch_size,71,73)

                output = self.model(batch_in_tec,batch_in_aux_no_date)#(batch_size,71,73)
                frame_num+=1
                print(f"预测第{frame_num}组")
                predictions.append(output.cpu().numpy())
                actuals.append(batch_exp_tec.cpu().numpy())
                datetime.append(batch_exp_aux.cpu().numpy())

        return np.array(predictions),np.array(actuals),np.array(datetime)

