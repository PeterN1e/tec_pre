#########dataloader1.py
dataset_year = 2011#使用数据集的年份
train_path = f"D:\\Dataset______________\\tec\\{dataset_year}\\TrainDataset" #tec图cdf文件夹路径
test_path = f"D:\\Dataset______________\\tec\\{dataset_year}\\TestDataset"
val_path = "D:\\Dataset______________\\tec\\2012\\ValDataset"

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
seq_length = 24    #一个batch序列时间步数
pred_length = 1    #所预测的时间步

model_name = "transformer"

######### transformerModule3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')