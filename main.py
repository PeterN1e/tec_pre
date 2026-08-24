import torch
import torch.nn as nn
from config import DatasetConfig,TrainConfig
import numpy as np
from torch.utils.data import DataLoader
import joblib
import torch.optim as optim
import warnings
from sklearn.preprocessing import MinMaxScaler
from common.dataloader1 import TecIonosphereDataset
from common.tec_train import TrainModel
from common.pic_show7 import pic_show,datagram
from model_selector import ModelAll
from common.prediction6 import TecPredict
from common.Data_Preprocessing import inverse_transform_predictions

import os
import matplotlib.pyplot as plt
from common.EvaluationMetrics import print_evaluation

cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()
plt.rcParams['font.sans-serif'] = [
    'SimHei',  # Windows 黑体
    'WenQuanYi Micro Hei',  # Linux 文泉驿
]
plt.rcParams['axes.unicode_minus'] = False


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    tec_scaler = MinMaxScaler()
    aux_scaler = MinMaxScaler()

    train_dataset = TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_month=cfg_dataset.start_month_train, end_month=cfg_dataset.end_month_train,
    input_day_num=cfg_train.input_day_num,
    is_train = True,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler
    )

    val_dataset =  TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_month=cfg_dataset.start_month_val, end_month=cfg_dataset.end_month_val,
    input_day_num=cfg_train.input_day_num,
    is_train=False,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler
    )

    train_dataloader = DataLoader(train_dataset,batch_size=cfg_train.batch_size, shuffle=True,drop_last = True)
    val_dataloader = DataLoader(val_dataset,batch_size=cfg_train.batch_size, shuffle=False, drop_last = True)

    print("训练数据集总步长：", train_dataset.__len__())
    print("测试数据集总步长：", val_dataset.__len__())
    print(f"批次大小：{cfg_train.batch_size}")

    model = ModelAll()
    model = model.to(cfg_train.device)
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    criterion_l1smooth = nn.SmoothL1Loss()
    optimizer=optim.Adam(model.parameters(),lr = cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    print("模型创建完成!")
    print(f"模型参数量:{sum(p.numel() for p in model.parameters() ):}")
    print("开始训练模型...")

    if not os.path.exists(os.path.join(cfg_train.model_path,cfg_train.model_name)):
        os.makedirs(os.path.join(cfg_train.model_path,cfg_train.model_name))

    tec_train = TrainModel(model = model,
                           train_loader = train_dataloader,
                           test_loader = val_dataloader,
                           criterion = criterion_mae,
                           criterion_name = "L1Loss",
                           optimizer =optimizer,
                           scheduler =scheduler,
                           save_best = True,
                           patience = cfg_train.patience,
                           model_save_path = os.path.join(cfg_train.model_path,cfg_train.model_name, "model_state_dict.pth"))
    train_losses, test_losses = tec_train.train(cfg_train.epochs_num)

#############保存和标准化映射关系
    joblib.dump(tec_scaler, os.path.join(cfg_train.model_path,"tec_scaler.pkl"))
    joblib.dump(aux_scaler, os.path.join(cfg_train.model_path,"aux_scaler.pkl"))

    print("模型训练结束")

    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='training loss')
    plt.plot(test_losses, label='test loss')
    plt.title("model loss")
    plt.xlabel('Epoch')
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
    file_path = os.path.join(cfg_train.pic_path, f'{cfg_train.model_name}train_loss.png')
    plt.savefig(file_path)
    plt.show()

def model_predict_only():

    tec_scaler = joblib.load(os.path.join(cfg_train.model_path , "tec_scaler.pkl"))
    aux_scaler = joblib.load(os.path.join(cfg_train.model_path, "aux_scaler.pkl"))

    test_dataset = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=cfg_dataset.start_month_test, end_month=cfg_dataset.end_month_test,
        input_day_num=cfg_train.input_day_num,
        is_train=False,
        tec_scaler = tec_scaler,
        aux_scaler = aux_scaler
    )
    test_dataloader = DataLoader(test_dataset, batch_size=cfg_train.batch_size, shuffle=False, drop_last=True)
    model =ModelAll()
    model = model.to(cfg_train.device)
    save_dir = cfg_train.model_name
    model.load_state_dict(torch.load(os.path.join(r"save/model_dict",save_dir, "model_state_dict.pth"), map_location=cfg_train.device,weights_only=True))

    tec_predict = TecPredict(model,test_dataloader)

    pre, act,aux= tec_predict()
    pre = inverse_transform_predictions(pre,tec_scaler)
    act = inverse_transform_predictions(act,tec_scaler)
    aux = inverse_transform_predictions(aux,aux_scaler)
    delta = act - pre

    # 将五维张量 (num_batches, batch_size, T, H, W) 合并为四维 (B, T, H, W)
    # prediction6.py输出形状为 (num_batches, batch_size, 12, 71, 73)
    # 需要reshape为 (num_batches*batch_size, 12, 71, 73) 才能用于评估
    num_batches, batch_size, T, H, W = pre.shape
    pre_4d = pre.reshape(num_batches * batch_size, T, H, W)
    act_4d = act.reshape(num_batches * batch_size, T, H, W)
    aux_3d = aux.reshape(num_batches * batch_size, T, aux.shape[-1])
    print(pre_4d.shape, act_4d.shape)
    print("预测完成")

    # 使用新的逐步评估函数，符合TEC预测论文标准
    print_evaluation(pre_4d, act_4d)

    # delta用于图片展示
    delta_4d = act_4d - pre_4d

    for i in range(10): #允许检索10次
        retrival = int(input(f"输入检索值0~{pre_4d.shape[0]}："))
        if 0<=retrival<pre_4d.shape[0]:
            pic_show(act_4d[retrival,:,:,:], pre_4d[retrival,:,:,:], aux_3d[retrival,:,:],delta_4d[retrival,:,:,:])
            print("完成绘制")
        else:
            print("输入错误")
            break

if __name__ == "__main__":
    a = input("训练后推理模式输入0，单推理模式输入1：")
    if a=="0":
        print("开始进行训练")
        main()
        model_predict_only()
        exit()
    elif a=="1":
        print("开始进行推理")
        model_predict_only()

    else:
        print("输入错误")
