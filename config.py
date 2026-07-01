import torch
from dataclasses import dataclass
from pathlib import Path
import platform
###########################################
# 获取当前 .py 文件所在的目录（）
BASE_DIR = Path(__file__).resolve().parent
# 在当前 .py 文件的同级目录下创建 log 文件夹
log_dir = BASE_DIR / "save" / "log"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "training.log"

pic_dir = BASE_DIR / "save"/"pic"
pic_dir.mkdir(parents=True, exist_ok=True)
pic_path = pic_dir

model_dir = BASE_DIR / "save"/"model_dict"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir
###############################################

if platform.system() == "Windows":
    dataset_base_path = Path("D:/Dataset_tec_NLY")
else:
    dataset_base_path = Path("/mnt/d/Dataset_tec_NLY")  # 或你的 Linux 挂载路径

n = 3
if n == 1:
    model_name = "transformer"
elif n == 2:
    model_name = "tcn"
elif n == 3:
    model_name = "convlstm"
elif n == 4:
    model_name = "convgru"

@dataclass
class ModelConfig:
    transmit_parameter : int = 3  # 卷积编码层的通道数大小
    out_dim : int = 128  # 卷积编码层最终线性层的输出维度
    model_name: str = model_name


@dataclass
class DatasetConfig:
    dataset_year : int = 2011  # 使用数据集的年份
    tec_dir = dataset_base_path/"tec_ionex_npy/igsg"  # tec图cdf文件夹路径
    indices_dir = dataset_base_path/"indices"

@dataclass
class TrainConfig:
    epochs_num: int = 10
    batch_size: int = 24
    seq_length: int = 24
    pred_length : int = 1 
    lr : float = 1e-3
    log_path : str = log_path
    pic_path : str = pic_path
    model_path : str = model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
