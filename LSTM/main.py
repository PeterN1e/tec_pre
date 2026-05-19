import torch
import torch.nn as nn
from config import train_path,test_path,device,batch_size,seq_length,epochs_num, model_name,lr
import numpy as np
from torch.utils.data import DataLoader
from common.dataloader1 import TecDataset1,data_reader
from common.tec_train import TrainModel
from common.pic_show7 import pic_show,datagram
import torch.optim as optim
import warnings
from sklearn.preprocessing import MinMaxScaler
from LSTM.convLSTM.model_all import ModelAll
import matplotlib.pyplot as plt
from prediction6 import TecPredict
from DataProcessing8 import data_save
from Data_Preprocessing import scale_tec_aux_data,inverse_transform_predictions

def main():
    torch.manual_seed(42)  # 42是生命、宇宙和一切终极问题的答案
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    print(f"使用设备：{device}")
    train_data_tec, train_data_aux= data_reader(train_path)
    print("训练集装载完毕")
    test_data_tec, test_data_aux= data_reader(test_path)
    print("测试集装载完毕")
    tec_scaler = MinMaxScaler()#不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()

    train_scaled_tec = scale_tec_aux_data(train_data_tec,tec_scaler,fit_scaler=True) #归一化后转变成原来的形状
    test_scaled_tec = scale_tec_aux_data(test_data_tec,tec_scaler,fit_scaler=False)

    train_scaled_aux = scale_tec_aux_data(train_data_aux,aux_scaler,fit_scaler=True)
    test_scaled_aux = scale_tec_aux_data(test_data_aux,aux_scaler,fit_scaler=False)

    print("test_scaled_tec:", train_scaled_tec.shape)
    print("test_scaled_aux:", test_scaled_tec.shape)
    print("train_scaled_tec:", train_scaled_aux.shape)
    print("test_scaled_tec:", test_scaled_aux.shape)

    train_dataset = TecDataset1(train_scaled_tec, train_scaled_aux, seq_length=seq_length)
    test_dataset = TecDataset1(test_scaled_tec, test_scaled_aux, seq_length=seq_length)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True,drop_last = True)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False,drop_last = True)
    print("训练数据集总步长：", train_dataset.__len__())
    print("测试数据集总步长：", test_dataset.__len__())
    print(f"批次大小：{batch_size}")

    model = ModelAll(transmit_parameter=3,
                     history_len=seq_length,
                     predict_len=1,
                     in_channel2=12,
                     hidden_channel2=12,
                     out_dim1=512
                     ).to(device)
    model = model.to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_l1smooth = nn.SmoothL1Loss()

    optimizer=optim.Adam(model.parameters(),lr=lr)   #优化器对象
    print("基础模型创建完成!")
    print(f"模型参数量:{sum(p.numel() for p in model.parameters() ):}")

    print("开始训练模型...")

    tec_train = TrainModel(model = model,train_loader=train_dataloader,test_loader=test_dataloader,criterion=criterion_mae,optimizer=optimizer)
    train_losses, test_losses = tec_train(epochs_num)

    torch.save(model.state_dict(), f"convlstm_model\\{model_name}.pth")
    print("the model training is completed and saved")

    plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'Arial Unicode MS']  #称之为rc配置或rc参数。通过rc参数可以修改默认的属性，
# 包括窗体大小、每英寸的点数、线条宽度、颜色、样式、坐标轴、坐标和网络属性、文本、字体等。
    plt.rcParams['axes.unicode_minus'] = False  #Matplotlib将使用ASCII字符U+002D来表示负号，从而避免负号显示为方块或乱码的问题。

    plt.figure(figsize=(24, 8))
    # fig = plt.figure(figsize=(a, b), dpi=dpi)
    # 其中：figsize 设置图形的大小，a 为图形的宽， b 为图形的高，单位为英寸
    # dpi 为设置图形每英寸的点数
    plt.subplot(1, 2, 1)  # 是 Matplotlib 中用于创建子图的一个函数，
    # 它允许在同一个图形窗口中绘制多个子图
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.title("model loss")
    plt.xlabel('Epoch')  # 分别为x ，y轴添加标签
    plt.ylabel('loss')
    plt.legend()  # 用于为图表添加图例，适用于 区分不同数据系列，提高可读性。
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.title('model loss(logarithmic scale)')
    plt.xlabel('Epoch')
    plt.ylabel('loss(logarithmic scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def model_predict_only():
    train_data_tec, train_data_aux = data_reader(train_path)
    test_data_tec, test_data_aux = data_reader(test_path)

    tec_scaler = MinMaxScaler()  # 不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()

    ##########将tec数据进行降维 步骤：reshape → 缩放 → 恢复形状

    train_scaled_tec = scale_tec_aux_data(train_data_tec, tec_scaler, fit_scaler=True)  # 归一化后转变成原来的形状
    test_scaled_tec = scale_tec_aux_data(test_data_tec, tec_scaler, fit_scaler=False)

    train_scaled_aux = scale_tec_aux_data(train_data_aux, aux_scaler, fit_scaler=True)
    test_scaled_aux = scale_tec_aux_data(test_data_aux, aux_scaler, fit_scaler=False)
    print("test_scaled_aux:", test_scaled_tec.shape)
    print("test_scaled_tec:", test_scaled_aux.shape)

    test_dataset = TecDataset1(test_scaled_tec, test_scaled_aux, seq_length=seq_length)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = ModelAll(transmit_parameter=3,
                     history_len=seq_length,
                     predict_len=1,
                     in_channel2=12,
                     hidden_channel2=12,
                     out_dim1=512
                     ).to(device)
    model.load_state_dict(torch.load(f"convlstm_model\\{model_name}.pth", map_location=device,weights_only=True))

    tec_predict = TecPredict(model,test_dataloader)
    pre, act,aux,delta= tec_predict()
    pre = inverse_transform_predictions(pre,tec_scaler)
    act = inverse_transform_predictions(act,tec_scaler)
    aux = inverse_transform_predictions(aux,aux_scaler)
    delta = inverse_transform_predictions(delta,tec_scaler)

    data_save(delta)

    delta_abs = np.abs(delta)   #计算绝对值

    delta_average_one_hour = np.mean(delta_abs,axis =(2, 3) )#对单张差值取平均值
    delta_average_one_day = np.mean(delta_average_one_hour,axis = 1)
    datagram(delta_average_one_day)

    print(pre.shape,act.shape)
    print("预测完成")
    for i in range(10): #允许检索10次
        retrival = int(input(f"输入检索值0~{pre.shape[0]-1}："))#输入的字符转换为数字
        if 0<=retrival<pre.shape[0]:
            pic_show(act[retrival], pre[retrival], aux[retrival],delta[retrival])  # 引用“图片展示”实例
            print("完成绘制")
        else:
            print("输入错误")
            break



if __name__ == "__main__":
    a = input("训练模式输入0，推理模式输入1：")
    if a=="0":
        print("开始进行训练")
        main()
        b = input("是否要接着推理？输入1则进行")
        if b=="1":
            model_predict_only()
        else:
            print("不进行推理，训练结束。")
            exit()
    elif a=="1":
        print("开始进行推理")
        model_predict_only()
        c = input("开始数据分析输入：0")
        if c=="0":
            exit()

    else:
        print("输入错误")
