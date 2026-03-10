import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
from keras import device
from sklearn.model_selection import train_test_split
from dataloader1 import TecDataset1,data_reader
from config import train_path,test_path,device,batch_size,seq_length,epochs_num
from sklearn.preprocessing import MinMaxScaler
from main import scale_tec_aux_data,inverse_transform_predictions


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================== 数据集定义 ======================
class TECDataset(Dataset):
    def __init__(self, tec_data, params_data, target_data):
        """
        初始化数据集
        :param tec_data: 时序TEC数据，shape=(样本数, 12, 71, 73)
        :param params_data: 4维参数数据，shape=(样本数, 4)
        :param target_data: 目标TEC图像，shape=(样本数, 71, 73)
        """
        self.tec_data = torch.FloatTensor(tec_data).unsqueeze(2)  # 增加通道维度: (N,12,1,71,73)
        self.params_data = torch.FloatTensor(params_data)  # (N,4)
        self.target_data = torch.FloatTensor(target_data).unsqueeze(1)  # (N,1,71,73)

        # 数据归一化（重要：避免梯度爆炸）
        self.tec_mean = self.tec_data.mean()
        self.tec_std = self.tec_data.std()
        self.params_mean = self.params_data.mean(dim=0)
        self.params_std = self.params_data.std(dim=0)

        self.tec_data = (self.tec_data - self.tec_mean) / self.tec_std
        self.params_data = (self.params_data - self.params_mean) / (self.params_std + 1e-8)
        self.target_data = (self.target_data - self.tec_mean) / self.tec_std

    def __len__(self):
        return len(self.tec_data)

    def __getitem__(self, idx):
        return self.tec_data[idx], self.params_data[idx], self.target_data[idx]


# ====================== 模型定义 ======================
class TECEncoder(nn.Module):
    """编码器：处理12帧时序TEC图像和4维参数"""

    def __init__(self, input_channels=1, params_dim=4, hidden_dim=64):
        super().__init__()

        # 时空特征提取（处理12帧图像）
        self.temporal_conv = nn.Sequential(
            # 输入: (B, 12, 1, 71, 73) -> 融合时间维度
            nn.Conv3d(in_channels=12, out_channels=hidden_dim // 2,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # (B,32,1,35,36)

            nn.Conv3d(in_channels=hidden_dim // 2, out_channels=hidden_dim,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B,64,1,17,18)
        )

        # ========== 时序参数特征提取 ==========
        # 输入: (B,12,4) -> 输出: (B,64)
        self.params_encoder = nn.Sequential(
            nn.Linear(12 * params_dim, 64),  # 展平12帧×4维=48维
            nn.ReLU(),
            nn.Linear(64, hidden_dim),  # (B,64)
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 17 * 18 + hidden_dim, hidden_dim * 16),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, tec_frames, params):
        """
        前向传播
        :param tec_frames: (B, 12, 1, 71, 73)
        :param params: (B, 4)
        :return: 融合特征 (B, 1024)
        """
        # 处理时序图像
        tec_frames = tec_frames.reshape(tec_frames.shape[0], 12, 1, 71, 73)
        tec_features = self.temporal_conv(tec_frames)  # (B,64,1,17,18)
        tec_features = tec_features.flatten(1)  # (B,64*17*18)

        # 处理参数
        params_feat = params.flatten(1)
        params_features = self.params_encoder(params_feat)  # (B,64)

        # 特征融合
        combined = torch.cat([tec_features, params_features], dim=1)  # (B,64*17*18+64)
        fused_features = self.fusion(combined)  # (B,64*16=1024)

        return fused_features


class TECPredictor(nn.Module):
    """中间预测层：特征映射"""
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.predictor(x)


class TECDecoder(nn.Module):
    """解码器：从特征重建71×73图像"""
    def __init__(self, input_dim=512, output_channels=1):
        super().__init__()

        # 将一维特征映射为低维空间特征图
        self.linear_proj = nn.Sequential(
            nn.Linear(input_dim, 64 * 18 * 18),
            nn.ReLU()
        )

        # 上采样恢复尺寸
        self.decoder = nn.Sequential(
            # 输入: (B,64,18,18)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 36x36
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 72x72
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # 调整到71x73
            nn.Conv2d(16, output_channels, kernel_size=3, padding=1),
            nn.Upsample(size=(71, 73), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        """
        前向传播
        :param x: (B, 512)
        :return: 重建图像 (B,1,71,73)
        """
        batch_size = x.shape[0]
        features = self.linear_proj(x)  # (B,64*18*18)
        features = features.view(batch_size, 64, 18, 18)  # (B,64,18,18)
        output = self.decoder(features)  # (B,1,71,73)
        return output


class TECForecastModel(nn.Module):
    """完整的TEC预测模型"""

    def __init__(self):
        super().__init__()
        self.encoder = TECEncoder()
        self.predictor = TECPredictor()
        self.decoder = TECDecoder()

    def forward(self, tec_frames, params):
        """
        完整前向传播
        :param tec_frames: (B,12,1,71,73) 12帧时序图像
        :param params: (B,4) 4维参数
        :return: 预测图像 (B,1,71,73)
        """
        enc_features = self.encoder(tec_frames, params)
        pred_features = self.predictor(enc_features)
        output = self.decoder(pred_features)
        return output


# ====================== 训练与评估函数 ======================
def train_model(model, train_loader, val_loader, epochs=50, lr=1e-4):
    """模型训练函数"""
    criterion = nn.MSELoss()  # 回归任务使用MSE损失
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    model.to(device)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for tec_frames, params, target in pbar:
            tec_frames = tec_frames.to(device)
            params = params.to(device)
            target = target.to(device)

            # 前向传播
            outputs = model(tec_frames, params)
            loss = criterion(outputs, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item() * tec_frames.size(0)
            pbar.set_postfix({'train_loss': loss.item()})

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tec_frames, params, target in val_loader:
                tec_frames = tec_frames.to(device)
                params = params.to(device)
                target = target.to(device)

                outputs = model(tec_frames, params)
                loss = criterion(outputs, target)
                val_loss += loss.item() * tec_frames.size(0)

        # 计算平均损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # 更新学习率
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # 保存最佳模型
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, f'tec_model_epoch_{epoch + 1}.pth')


# ====================== 数据加载与模型运行 ======================
if __name__ == "__main__":
    train_data_tec, train_data_aux = data_reader(train_path)
    test_data_tec, test_data_aux = data_reader(test_path)
    tec_scaler = MinMaxScaler()  # 不同数据缩放器不允许共同使用
    aux_scaler = MinMaxScaler()
    train_scaled_tec = scale_tec_aux_data(train_data_tec, tec_scaler, fit_scaler=True)  # 归一化后转变成原来的形状
    test_scaled_tec = scale_tec_aux_data(test_data_tec, tec_scaler, fit_scaler=False)

    train_scaled_aux = scale_tec_aux_data(train_data_aux, aux_scaler, fit_scaler=True)
    test_scaled_aux = scale_tec_aux_data(test_data_aux, aux_scaler, fit_scaler=False)

    # 创建数据集
    train_dataset = TecDataset1(train_scaled_tec, train_scaled_aux, seq_length=seq_length)
    test_dataset = TecDataset1(test_scaled_tec, test_scaled_aux, seq_length=seq_length)
    # 创建数据加载器
    batch_size = 24  # 根据GPU显存调整
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化模型
    model = TECForecastModel()

    # 训练模型
    train_model(model, train_loader, test_loader, epochs=50, lr=1e-4)

    # 模型推理示例
    model.eval()
    with torch.no_grad():
        # 取一个测试样本
        for batch_in_tec, batch_in_aux, batch_exp in test_loader:
            batch_in_tec= batch_in_tec.float().to(device)  # 转换前的数据类型为float64，为了和之后权重（float32）偏置计算
            batch_in_aux = batch_in_aux.float().to(device)
            batch_exp = batch_exp.float().to(device)
            pred_tec = model(batch_in_tec, batch_in_aux)

        # 反归一化恢复原始尺度
            pred_tec = inverse_transform_predictions(pred_tec,batch_exp,tec_scaler)
            print(f"预测图像形状: {pred_tec.shape}")  # 应输出 (batch_size,1,71,73)