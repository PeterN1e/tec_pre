import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    transmit_parameter : int = 3  # 卷积编码层的通道数大小
    out_dim : int = 128  # 卷积编码层最终线性层的输出维度
    model_name: str = "lstm"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@dataclass
class DatasetConfig:
    dataset_year : int = 2011  # 使用数据集的年份
    train_path : str = f"D:\\Dataset______________\\tec\\{dataset_year}\\TrainDataset"  # tec图cdf文件夹路径
    test_path : str = f"D:\\Dataset______________\\tec\\{dataset_year}\\TestDataset"
    val_path : str = "D:\\Dataset______________\\tec\\2012\\ValDataset"

@dataclass
class TrainConfig:
    epochs_num: int = 10
    batch_size: int = 24
    seq_length: int = 24
    pred_length : int = 1 
    lr : float = 1e-3

