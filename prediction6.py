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
        delta = []
        with torch.no_grad():
            for batch_in_tec,batch_in_aux,batch_exp in self.test_loader:
                batch_in_tec = batch_in_tec.float().to(device) #(batch_size,seq_length,71,73)
                # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux_no_date = batch_in_aux[:,:,1:].float().to(device)
                batch_exp_tec = batch_exp[0].float().to(device)#(batch_size,71,73)
                batch_exp_aux = batch_exp[1].float().to(device)  # (batch_size,71,73)

                output = self.model(batch_in_tec,batch_in_aux_no_date)#(batch_size,71,73)
                print(f"预测第{frame_num}组")
                frame_num += 1

                delta_one = batch_exp_tec-output #计算每个批次真实值与实际值之差
                delta.append(delta_one.cpu().numpy())#delta 列表中的张量仍在 GPU（cuda:0）上
                                                    #prediction6.py 使用 np.array(delta) 时未先转移到 CPU
                #但 NumPy 不支持 GPU 数据，要将数据放到cpu上并转化为np数据
                predictions.append(output.cpu().numpy())
                actuals.append(batch_exp_tec.cpu().numpy())
                datetime.append(batch_exp_aux.cpu().numpy())

        return np.array(predictions),np.array(actuals),np.array(datetime),np.array(delta)

