import torch
import torch.nn as nn
from config import ModelConfig,DatasetConfig,TrainConfig
import numpy as np
from torch.utils.data import DataLoader
import joblib
import torch.optim as optim
import warnings
from sklearn.preprocessing import MinMaxScaler
from common.dataloader1 import TecDataset1,data_reader,TecIonosphereDataset
from common.tec_train import TrainModel
from common.pic_show7 import pic_show,datagram
from common.model_all import ModelAll
from common.prediction6 import TecPredict
from common.Data_Preprocessing import scale_tec_aux_data,inverse_transform_predictions

import os
import matplotlib.pyplot as plt

cfg_model = ModelConfig()
cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    tec_scaler = MinMaxScaler()  # 不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()

    train_dataset = TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_year=2002, end_year=2010,
    start_month=200201, end_month=200812,
    k=3,
    is_train = True,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler

    )

    val_dataset =  TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_year=2002, end_year=2010,
    start_month=200901, end_month=201012,
    k=3,
    is_train=False,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler
    )

    train_dataloader = DataLoader(train_dataset,batch_size=cfg_train.batch_size, shuffle=True,drop_last = True)
    val_dataloader = DataLoader(val_dataset,batch_size=cfg_train.batch_size, shuffle=False,drop_last = True)

    print("训练数据集总步长：", train_dataset.__len__())
    print("测试数据集总步长：", val_dataset.__len__())
    print(f"批次大小：{cfg_train.batch_size}")

    model = ModelAll(transmit_parameter = cfg_model.transmit_parameter,
                     history_len = cfg_train.seq_length,
                     predict_len = cfg_train.pred_length,
                     aux_dim = 5,
                     channel = 12
                     ).to(cfg_train.device)
    model = model.to(cfg_train.device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_l1smooth = nn.SmoothL1Loss()

    optimizer=optim.Adam(model.parameters(),lr = cfg_train.lr)   #优化器对象

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    print("模型创建完成!")
    print(f"模型参数量:{sum(p.numel() for p in model.parameters() ):}")
    print("开始训练模型...")

    tec_train = TrainModel(model = model,
                           train_loader = train_dataloader,
                           test_loader = val_dataloader,
                           criterion = criterion_mae,
                           optimizer =optimizer,
                           scheduler =scheduler,
                           save_best = True)
    train_losses, test_losses = tec_train.train(cfg_train.epochs_num,)

#############保存模型参数和
    if not os.path.exists(os.path.join(cfg_train.model_path,cfg_model.model_name)):
        os.makedirs(os.path.join(cfg_train.model_path,cfg_model.model_name))
    torch.save(model.state_dict(), os.path.join(cfg_train.model_path,cfg_model.model_name, "model_state_dict.pth"))
    joblib.dump(tec_scaler, os.path.join(cfg_train.model_path,"tec_scaler.pkl"))
    joblib.dump(aux_scaler, os.path.join(cfg_train.model_path,"aux_scaler.pkl"))

    print("模型训练结束")

    plt.rcParams['font.sans-serif']=['Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.title("model loss")
    plt.xlabel('Epoch')  # 分别为x ，y轴添加标签
    plt.ylabel('loss')
    plt.legend()
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
    os.makedirs(cfg_train.pic_path, exist_ok=True)
    file_path = os.path.join(cfg_train.pic_path, f'{cfg_model.model_name}train_loss.png')
    plt.savefig(file_path)
    plt.show()

def model_predict_only():

    tec_scaler = joblib.load(os.path.join(cfg_train.model_path , "tec_scaler.pkl"))
    aux_scaler = joblib.load(os.path.join(cfg_train.model_path, "aux_scaler.pkl"))

    test_dataset = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_year=2002, end_year=2010,
        start_month=200201, end_month=200812,
        k=3,
        is_train=False,
        tec_scaler = tec_scaler,
        aux_scaler = aux_scaler
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg_train.batch_size, shuffle=False, drop_last=True)
    model = ModelAll(transmit_parameter = cfg_model.transmit_parameter,
                     history_len = cfg_train.seq_length,
                     predict_len = cfg_train.pred_length,
                     aux_dim = 5,
                     channel = 12
                     ).to(cfg_train.device)
    save_dir = cfg_model.model_name
    model.load_state_dict(torch.load(os.path.join("model_dict",save_dir, "model_state_dict.pth"), map_location=cfg_train.device,weights_only=True))

    tec_predict = TecPredict(model,test_dataloader)

    pre, act,aux= tec_predict()
    pre = inverse_transform_predictions(pre,tec_scaler)
    act = inverse_transform_predictions(act,tec_scaler)
    aux = inverse_transform_predictions(aux,aux_scaler)
    delta = act - pre
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
