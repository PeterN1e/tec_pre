import torch
import torch.nn as nn
from keras import device


from config import train_path,test_path,device,batch_size,seq_length,epochs_num
import numpy as np
from torch.utils.data import Dataset,DataLoader
from dataloader1 import TecDataset,TecDataset1,data_reader
from tec_train import TrainModel
import torch.optim as optim
import warnings
from sklearn.preprocessing import MinMaxScaler
from model_all import ModelAll
import matplotlib.pyplot as plt
from prediction6 import TecPredict
import cdflib

def main():
    torch.manual_seed(42)  # 42是生命、宇宙和一切终极问题的答案
    np.random.seed(42)
    warnings.filterwarnings('ignore')
    print(f"使用设备：{device}")
    train_data_tec, train_data_aux= data_reader(train_path)
    test_data_tec, test_data_aux= data_reader(test_path)


    tec_scaler = MinMaxScaler()#不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()

    ##########将tec数据进行降维 步骤：reshape → 缩放 → 恢复形状
    original_shape_tec_train = train_data_tec.shape    #保存原始形状
    train_data_tec_2d = train_data_tec.reshape(original_shape_tec_train[0], -1)# 变为 (6528, 71*73)
    original_shape_tec_test = test_data_tec.shape
    test_data_tec_2d = test_data_tec.reshape(original_shape_tec_test[0], -1)  # 变为 (2208, 71*73)

    print("train_data_tec1:", train_data_tec_2d.shape)
    print("test_data_tec1:", test_data_tec_2d.shape)

    train_scaled_tec= tec_scaler.fit_transform(train_data_tec_2d)
    train_scaled_aux= aux_scaler.fit_transform(train_data_aux)

    test_scaled_tec = tec_scaler.transform(test_data_tec_2d)
    test_scaled_aux = aux_scaler.transform(test_data_aux)

    train_scaled_tec = train_scaled_tec.reshape(original_shape_tec_train) #归一化后转变成原来的形状
    test_scaled_tec = test_scaled_tec.reshape(original_shape_tec_test)

    print("test_scaled_tec:", train_scaled_tec.shape)
    print("test_scaled_aux:", test_scaled_tec.shape)
    print("train_scaled_tec:", train_scaled_aux.shape)
    print("test_scaled_tec:", test_scaled_aux.shape)

    train_dataset = TecDataset1(train_scaled_tec, train_scaled_aux, seq_length=seq_length)
    test_dataset = TecDataset1(test_scaled_tec, test_scaled_aux, seq_length=seq_length)

    train_dataloader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False,drop_last = False)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False,drop_last = False)
    print("训练数据集总步长：", train_dataset.__len__())
    print("测试数据集总步长：", test_dataset.__len__())
    print(f"批次大小：{batch_size}")

    model = ModelAll(transmit_parameter=48,
                     history_len = 24,
                     predict_len = 1,
                     d_model = 512,
                     in_dim2 = 500,
                     out_dim1 = 500
                     )

    model = model.to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_l1smooth = nn.SmoothL1Loss()


    optimizer=optim.Adam(model.parameters(),lr=0.001)   #优化器对象
    print("基础模型创建完成!")
    print(f"模型参数量:{sum(p.numel() for p in model.parameters() ):}")

    print("开始训练模型...")

    tec_train = TrainModel(model = model,train_loader=train_dataloader,test_loader=test_dataloader,criterion=criterion_l1smooth,optimizer=optimizer)
    train_losses, test_losses = tec_train(epochs_num)

    torch.save(model, "model.pth")
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
    plt.ylabel('label')
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

    ########预测
    tec_predict = TecPredict(model,test_dataloader)
    pre,act = tec_predict()
    print(pre.shape,act.shape)#(frame, batch_size, seq_length, 71, 73) (frame, batch_size, 71, 73)


def inverse_transform_predictions(predictions,actual,scaler):
    """
    作用：对数据进行反标准化，还原至输入状态
    :param predictions:预测
    :param actual:实际
    :param scaler:设定的标准化
    :return:
    """
    pre = predictions[10,0,:,:,:]
    act= actual[10,0,:,:]
    # pre = pre.squeeze(0)
    # act = act.squeeze(0)
    pre = pre.reshape(1,-1)
    act = act.reshape(1,-1)
    pre=scaler.inverse_transform(pre)
    act=scaler.inverse_transform(act)
    pre = pre.reshape(1, 71, 73)
    act = act.reshape(1, 71, 73)
    return pre,act


def model_predict_only():
    train_data_tec, train_data_aux = data_reader(train_path)
    test_data_tec, test_data_aux = data_reader(test_path)

    tec_scaler = MinMaxScaler()  # 不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()

    ##########将tec数据进行降维 步骤：reshape → 缩放 → 恢复形状
    original_shape_tec_train = train_data_tec.shape  # 保存原始形状
    train_data_tec_2d = train_data_tec.reshape(original_shape_tec_train[0], -1)  # 变为 (6528, 71*73)
    original_shape_tec_test = test_data_tec.shape
    test_data_tec_2d = test_data_tec.reshape(original_shape_tec_test[0], -1)  # 变为 (2208, 71*73)

    print("train_data_tec1:", train_data_tec_2d.shape)
    print("test_data_tec1:", test_data_tec_2d.shape)

    train_scaled_tec = tec_scaler.fit_transform(train_data_tec_2d)
    train_scaled_aux = aux_scaler.fit_transform(train_data_aux)

    test_scaled_tec = tec_scaler.transform(test_data_tec_2d)
    test_scaled_aux = aux_scaler.transform(test_data_aux)

    #train_scaled_tec = train_scaled_tec.reshape(original_shape_tec_train)  # 归一化后转变成原来的形状
    test_scaled_tec = test_scaled_tec.reshape(original_shape_tec_test)

    #print("test_scaled_tec:", train_scaled_tec.shape)
    print("test_scaled_aux:", test_scaled_tec.shape)
    #print("train_scaled_tec:", train_scaled_aux.shape)
    print("test_scaled_tec:", test_scaled_aux.shape)

    #train_dataset = TecDataset1(train_scaled_tec, train_scaled_aux, seq_length=seq_length)
    test_dataset = TecDataset1(test_scaled_tec, test_scaled_aux, seq_length=seq_length)

    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    loaded_model = torch.load("model.pth")
    tec_predict = TecPredict(loaded_model,test_dataloader)
    pre, act = tec_predict()
    pre,act =inverse_transform_predictions(pre,act,tec_scaler)
    print(pre.shape,act.shape)

    plt.figure(figsize=(10, 10))
    # proj = ccrs.PlateCarree()
    plt.subplot(2, 1, 1)
    plt.pcolormesh(pre[0,:,:], shading='auto', cmap='jet')

    plt.colorbar(label='TECU')
    plt.title("tec pre")
    #plt.savefig(f'tecUHR_{0}.png', dpi=150)

    plt.subplot(2, 1, 2)
    plt.pcolormesh( act[0, :, :], shading='auto', cmap='jet')
    plt.colorbar(label='TECU')
    plt.title('tec act')
    #plt.savefig(f'tecUHR_{0}.png', dpi=150)

    plt.show()
if __name__ == "__main__":
    model_predict_only()
    #main()