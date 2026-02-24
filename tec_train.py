import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader
import numpy as np
import os  #处理文件和目录
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging  #跟踪程序的运行状态、调试错误以及记录重要信息
import time
import json
from config import device

# 导入进度条模块，tqdm可以让我们在训练过程中看到每个epoch的进度
from tqdm import tqdm
from typing import Optional,Dict,List,Any

logging.basicConfig(
    level=logging.INFO,#只有 INFO 级别及以上（INFO、WARNING、ERROR、CRITICAL）的日志会被处理；DEBUG 会被忽略。
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',#依次是时间、logger 名、级别、正文。
    handlers=[                                  # 同时把日志送到两个地方
        logging.FileHandler("training.log"),  #所保存的日志
        logging.StreamHandler()               #控制台输出，可以实时查看进度
    ]
)
logger = logging.getLogger(__name__)


class TrainModel(nn.Module):
    def __init__(self,model,train_loader,test_loader,criterion,optimizer):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer


    def forward(self,num_epochs):
        train_losses = []
        test_losses=[]

        for epoch in range(1,num_epochs+1):
            self.model.train()
            train_loss ,num= 0.0,0
            pbar = tqdm(self.train_loader,
                        total=len(self.train_loader),
                        ncols=100,
                        desc=f'Epoch {epoch}/{num_epochs}',
                        leave=False)
            for batch_in_tec,batch_in_aux,batch_exp in pbar:
                """
                batch_in是输入模型的输入，batch_exp：用于计算模型的损失，期待的输出
                out是模型产生的输出 
                batch_in_tec: torch.Size([batch_size, 24, 71, 73])
                batch_in_aux: torch.Size([batch_size, 24, 4])
                """


                batch_in_tec = batch_in_tec.float().to(device)#转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                batch_in_aux = batch_in_aux.float().to(device)
                batch_exp = batch_exp.float().to(device)
                #batch_exp(24,71,73)
                output = self.model(batch_in_tec,batch_in_aux)
                #output:(24,1,71,73)
                #loss = self.criterion(output.squeeze(1), batch_exp)
                #batch_exp和output尽管维度不同，但是如果criterion不要求类别索引，也不看通道维度，而是逐元素做回归差值，就只需考虑元素个数一致就ok
                loss = self.criterion(output, batch_exp)
                #挤压掉output:(24,1,71,73)中的1，保持对齐
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                num+=1
                pbar.set_postfix({'loss':f'{loss.item():.4f}',
                                  'avg':f'{train_loss / num:.4f}'})

            ###############验证阶段##############
            self.model.eval()  # 模型切换为评估模式
            test_loss = 0.0
            with torch.no_grad():
                for batch_in_tec,batch_in_aux,batch_exp in self.test_loader:
                    batch_in_tec = batch_in_tec.float().cuda()  # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
                    batch_in_aux = batch_in_aux.float().cuda()
                    batch_exp = batch_exp.float().cuda()

                    outputs = self.model(batch_in_tec,batch_in_aux)
                    #loss = self.criterion(outputs.squeeze(1), batch_exp)
                    loss = self.criterion(outputs, batch_exp)
                    test_loss += loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            pbar.close()
            logger.info(f'Epoch {epoch:3d} | '
                        f'Train {avg_train_loss:.5f} | '
                        f'Test  {avg_test_loss:.5f}')
            epoch+=1
        return train_losses, test_losses
