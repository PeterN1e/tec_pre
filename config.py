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
log_path = log_dir

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

model_select = 4
if model_select == 1:
    model_name = "E_P_D"
elif model_select == 2:
    model_name = "ED_CGConvLSTM"
elif model_select == 3:
    model_name = "GA_Predrnn"
elif model_select == 4:
    model_name = "ED_Autoformer"
else:
    raise ValueError("模型不存在")

n = 3
if n == 1:
    EPDmodel_name = "transformer"
elif n == 2:
    EPDmodel_name = "tcn"
elif n == 3:
    EPDmodel_name = "convlstm"
elif n == 4:
    EPDmodel_name = "convgru"
else:
    raise ValueError("模型不存在")

@dataclass
class FusionConfig:
    channel : int = 12

@dataclass
class DatasetConfig:
    start_month_train = 200201
    end_month_train = 200208
    start_month_val = 200209
    end_month_val = 200210
    start_month_test = 200211
    end_month_test = 200211
    aux_dim : int = 6
    tec_dir = dataset_base_path/"tec_ionex_npy/igsg"  # tec图cdf文件夹路径
    indices_dir = dataset_base_path/"indices"

@dataclass
class TrainConfig:
    model_name: str = model_name
    epochs_num: int = 10
    patience: int = 5
    batch_size: int = 4
    input_day_num: int = 3
    output_day_num: int = 1
    input_length: int = input_day_num*12
    output_length : int = output_day_num*12
    lr : float = 1e-3
    log_path : str = log_path
    pic_path : str = pic_path
    model_path : str = model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class EDCGConvLSTMConfig:
    input_dim: int = 1
    output_dim: int = 1
    num_layers: int = 4
    kernel_size: int = 3
    hidden_dim: int = 24
    H : int =  71
    W : int = 73
@dataclass
class EPDConfig:
    transmit_parameter : int = 3  # 卷积编码层的通道数大小
    out_dim : int = 128  # 卷积编码层最终线性层的输出维度
    EPDmodel_name: str = EPDmodel_name
@dataclass
class GAPredrnnConfig:
    input_dim: int = 4            # TEC + dst + ap + f10.7
    hidden_dim: int = 64
    num_layers: int = 3           # stacked ST-LSTM layers
    kernel_size: int = 5
    input_length: int = 24        # 2 days x 12 steps
    output_length: int = 12       # 1 day  x 12 steps
    aux_output_dim: int = 3       # dst, ap, f10.7
    # Halo attention
    block_size: int = 8
    halo_size: int = 2
    num_heads: int = 4
    # Discriminator
    disc_base_ch: int = 64
    # Loss weights (a >> b per paper)
    lambda_tec: float = 1.0
    lambda_aux: float = 0.1
    # Training
    d_lr: float = 1e-4
    g_lr: float = 1e-3
    epochs: int = 50
    patience: int = 10
    clip_grad: float = 1.0

@dataclass
class EDAutoformerConfig:
    d_model: int = 512            # Autoformer 隐藏维度
    n_heads: int = 8              # 自相关多头数
    d_ff: int = 2048              # FFN 中间维度
    e_layers: int = 2             # 编码器层数
    d_layers: int = 1             # 解码器层数
    moving_avg: int = 13          # 系列分解滑动平均核大小
    factor: int = 3               # top-k 时延 = factor * ln(L)
    dropout: float = 0.05
    activation: str = "gelu"
    encode_channels: tuple = (64, 128, 256, 512)  # TEC 编码器通道数
    label_len: int = None         # 解码器已知标签长度, None 时取 input_length // 2
