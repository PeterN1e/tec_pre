import torch
import torch.nn as nn
from config import DatasetConfig, TrainConfig, get_train_config
import numpy as np
from torch.utils.data import DataLoader
import joblib
import torch.optim as optim
import warnings
from sklearn.preprocessing import StandardScaler
from common.dataloader1 import TecIonosphereDataset
from common.tec_train import TrainModel, train_gan
from common.pic_show7 import pic_show,datagram
from model_selector import ModelAll
from common.prediction6 import TecPredict
from common.Data_Preprocessing import inverse_transform_predictions

import os
import matplotlib.pyplot as plt
from common.EvaluationMetrics import print_evaluation, log_evaluation
from core.checkpoint import load_checkpoint

cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()


def build_discriminator(model_name, device):
    """Factory for GAN discriminators; extend here for new GAN models."""
    if model_name == "GA_Predrnn":
        from GA_Predrnn.discriminator import Discriminator
        from config import GAPredrnnConfig
        return Discriminator(base_ch=GAPredrnnConfig().disc_base_ch).to(device)
    raise ValueError(f"暂无该 GAN 模型的判别器预设: {model_name}")


plt.rcParams['font.sans-serif'] = [
    'SimHei',  # Windows 黑体
    'WenQuanYi Micro Hei',  # Linux 文泉驿
]
plt.rcParams['axes.unicode_minus'] = False


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    preset = get_train_config(cfg_train.model_name)
    tec_scaler = StandardScaler()
    aux_scaler = StandardScaler()

    train_dataset = TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_month=cfg_dataset.start_month_train, end_month=cfg_dataset.end_month_train,
    input_day_num=preset.input_day_num,
    is_train = True,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler
    )

    val_dataset =  TecIonosphereDataset(
    tec_dir=cfg_dataset.tec_dir,
    indices_dir=cfg_dataset.indices_dir,
    start_month=cfg_dataset.start_month_val, end_month=cfg_dataset.end_month_val,
    input_day_num=preset.input_day_num,
    is_train=False,
    tec_scaler = tec_scaler,
    aux_scaler = aux_scaler
    )

    train_dataloader = DataLoader(train_dataset,batch_size=preset.batch_size, shuffle=True,drop_last = True)
    val_dataloader = DataLoader(val_dataset,batch_size=preset.batch_size, shuffle=False, drop_last=False)

    print("训练数据集总步长：", train_dataset.__len__())
    print("测试数据集总步长：", val_dataset.__len__())
    print(f"批次大小：{preset.batch_size}")

    model = ModelAll()
    model = model.to(cfg_train.device)
    model_dir = os.path.join(cfg_train.model_path, cfg_train.model_name)
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "model_state_dict.pth")

    if preset.use_gan:
        discriminator = build_discriminator(cfg_train.model_name, cfg_train.device)
        g_optimizer = optim.Adam(model.parameters(), lr=preset.g_lr)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=preset.d_lr)
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=3)
        scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=3)
        print("GAN 模型创建完成!")
        print(f"生成器参数量:{sum(p.numel() for p in model.parameters()):}")
        print(f"判别器参数量:{sum(p.numel() for p in discriminator.parameters()):}")
        history = train_gan(
            generator=model,
            discriminator=discriminator,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            model_name=cfg_train.model_name,
            model_save_path=model_save_path,
            device=cfg_train.device,
            epochs=preset.epochs,
            patience=preset.patience,
            lambda_tec=preset.lambda_tec,
            lambda_aux=preset.lambda_aux,
            adv_weight=preset.adv_weight,
            clip_grad=preset.clip_grad,
            use_amp=getattr(cfg_train, "use_amp", False),
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
        )
        train_losses = history["train_g"]
        test_losses = history["val_loss"]
    else:
        criterion_mse = nn.MSELoss()
        criterion_mae = nn.L1Loss()
        criterion_l1smooth = nn.SmoothL1Loss()
        criterion_name = "L1Loss"
        if cfg_train.model_name == "ModelCanon":
            from common.loss_function import DeltaCriterion
            criterion_mae = DeltaCriterion(model.model)
            criterion_name = "DeltaCriterion"
        optimizer = optim.Adam(model.parameters(), lr=preset.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        print("模型创建完成!")
        print(f"模型参数量:{sum(p.numel() for p in model.parameters()):}")
        print("开始训练模型...")
        tec_train = TrainModel(model = model,
                               train_loader = train_dataloader,
                               test_loader = val_dataloader,
                               criterion = criterion_mae,
                               criterion_name = criterion_name,
                               optimizer = optimizer,
                               scheduler = scheduler,
                               save_best = True,
                               patience = preset.patience,
                               model_save_path = model_save_path)
        train_losses, test_losses = tec_train.train(preset.epochs)

#############保存和标准化映射关系
    joblib.dump(tec_scaler, os.path.join(model_dir, "tec_scaler.pkl"))
    joblib.dump(aux_scaler, os.path.join(model_dir, "aux_scaler.pkl"))

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

def _load_state_dict(path, device):
    return load_checkpoint(path, map_location=device)["model_state_dict"]


def _resolve_artifact_dir(model_name):
    model_dir = os.path.join(cfg_train.model_path, model_name)
    if os.path.isdir(model_dir):
        return model_dir
    return cfg_train.model_path


def model_predict_only(preset=None, model=None, interactive=True):
    preset = preset or get_train_config(cfg_train.model_name)
    artifact_dir = _resolve_artifact_dir(cfg_train.model_name)

    tec_scaler = joblib.load(os.path.join(artifact_dir, "tec_scaler.pkl"))
    aux_scaler = joblib.load(os.path.join(artifact_dir, "aux_scaler.pkl"))

    test_dataset = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=cfg_dataset.start_month_test, end_month=cfg_dataset.end_month_test,
        input_day_num=preset.input_day_num,
        is_train=False,
        tec_scaler = tec_scaler,
        aux_scaler = aux_scaler
    )
    test_dataloader = DataLoader(test_dataset, batch_size=preset.batch_size, shuffle=False, drop_last=False)
    if model is None:
        model = ModelAll()
    model = model.to(cfg_train.device)
    model.load_state_dict(
        _load_state_dict(
            os.path.join(artifact_dir, "model_state_dict.pth"),
            cfg_train.device,
        )
    )

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

    # 将评估指标写入当前模型对应的训练日志末尾
    import logging
    eval_logger = logging.getLogger(f"train.{cfg_train.model_name}")
    if not eval_logger.handlers:
        log_file = cfg_train.log_path / f"{cfg_train.model_name}.log"
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        eval_logger.addHandler(fh)
        eval_logger.setLevel(logging.INFO)
    log_evaluation(eval_logger, pre_4d, act_4d)

    # delta用于图片展示
    delta_4d = act_4d - pre_4d

    if interactive:
        for i in range(10): #允许检索10次
            retrival = int(input(f"输入检索值0~{pre_4d.shape[0]}："))
            if 0<=retrival<pre_4d.shape[0]:
                pic_show(act_4d[retrival,:,:,:], pre_4d[retrival,:,:,:], aux_3d[retrival,:,:],delta_4d[retrival,:,:,:])
                print("完成绘制")
            else:
                print("输入错误")
                break

if __name__ == "__main__":
    preset = get_train_config(cfg_train.model_name)
    a = input("训练后推理模式输入0，单推理模式输入1：")
    if a=="0":
        print("开始进行训练")
        main()
        model_predict_only(preset=preset)
        exit()
    elif a=="1":
        print("开始进行推理")
        model_predict_only(preset=preset)

    else:
        print("输入错误")
