#########dataloader1.py
train_path = "D:\\Dataset______________\\tec\\2011\\TrainDataset" #tec图cdf文件夹路径
#由于第一天的数据错误
test_path = "D:\\Dataset______________\\tec\\2011\\TestDataset"#特征路径
day_seq = 1 #定义一个batch_size由多少天的数据构成
#########CNNEncoder2.py
transmit_parameter = 3  #卷积编码层的通道数大小
out_dim = 128            #卷积编码层最终线性层的输出维度
#########TokenFusion.py
         #数据集序列长度
######### main.py
import torch
epochs_num = 10
batch_size = 24
seq_length = 6    #一个batch序列时间步数

model_name = "transformer"

######### transformerModule3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')