import cdflib
from torch.utils.data import Dataset,DataLoader
from os import listdir
import os
import numpy as np
import pandas as pd  #分析结构化数据
import torch

from common.Data_Preprocessing import scale_tec_aux_data
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



class TecDataset1(Dataset):
    """
    输出：tec图[前seq_length张]，特征数据[前seq_length张]，（tec图[下一张]，特征数据[下一张]）
    """
    def __init__(self,data_tec,data_aux,seq_length):
        super().__init__()
        self.data_aux = data_aux.astype(np.float32)
        self.data_tec = data_tec.astype(np.float32)
        self.seq_length =seq_length
    def __len__(self):
        return len(self.data_tec)-self.seq_length
    def __getitem__(self, index):
        return self.data_tec[index:index+self.seq_length],self.data_aux[index:index+self.seq_length],[self.data_tec[index+self.seq_length],self.data_aux[index+self.seq_length]]

def data_reader(data_path):
    tec_batch = []
    tec_path_all = os.path.join(data_path, "tecMap")
    feature_path_all = os.path.join(data_path, "omni_2011_complete.csv")
    tec_data_list = listdir(tec_path_all)
    for day in range(len(tec_data_list)):
        tec_absolute_path = os.path.join(data_path, "tecMap", tec_data_list[day])
        cdf = cdflib.CDF(f"{tec_absolute_path}")
        tec = cdf.varget('tecUHR')
        for h in range(tec.shape[0]):
            tec_batch.append(tec[h])

    with open(feature_path_all, 'rb') as f:
        header = f.read(4)
    if header == b'PK\x03\x04':
        feature_list = pd.read_excel(feature_path_all)
    else:
        feature_list = pd.read_csv(feature_path_all)
    tec_batch = np.array(tec_batch)
    aux = ["doy","hour","ssn","dst","f10.7"]
    #detect_all_same_value(tec_batch)

    return tec_batch,np.array(feature_list[aux].values) #np.array可接受字符串，数字 布尔类型

class TecIonosphereDataset(Dataset):
    def __init__(self,
                 tec_dir,               # 存放 2005_igsg.npy 的文件夹
                 indices_dir,           # 存放 2005_indices.csv 的文件夹
                 start_year=2002,       # 全局起始年
                 end_year=2010,         # 全局结束年
                 start_month=200201,    # 数据集起始年月（如200201）
                 end_month=201012,      # 数据集结束年月
                 k=3,                   # 用k天作为输入
                 is_train=False,
                 tec_scaler = None,
                 aux_scaler = None):
        super().__init__()
        self.tec_dir = tec_dir
        self.indices_dir = indices_dir
        self.k = k
        self.is_train = is_train

        # 解析起止年月
        self.start_y = start_month // 100
        self.start_m = start_month % 100
        self.end_y = end_month // 100
        self.end_m = end_month % 100

        # ===================== 2. 扫描所有年份文件 =====================
        self.years = sorted([
            int(f.split("_")[0])
            for f in os.listdir(tec_dir)
            if f.endswith("igsg.npy")
        ])
        self.years = [y for y in self.years if self.start_y <= y <= self.end_y]

        # ===================== 3. 构建全局时间索引表（核心） =====================
        self.time_index = []  # 每条记录: [year, month, day, hour, global_idx]
        self.global_idx = 0  # 整个时序的唯一索引
        self.year_offset = {}  # 记录每年在全局索引中的起始位置

        for year in self.years:
            self.year_offset[year] = self.global_idx  # 记录该年的起始全局索引
            # 加载该年CSV，获取完整时间轴
            csv_path = os.path.join(indices_dir, f"{year}_indices.csv")
            # df = pd.read_csv(csv_path)
            with open(csv_path, 'rb') as f:
                header = f.read(4)
            if header == b'PK\x03\x04':
                 feature_list = pd.read_excel(csv_path)
            else:
                feature_list = pd.read_csv(csv_path)

            df = feature_list[["year", "doy", "hour", "kp", "ssn", "dst", "ap", "f10.7", "ae"]]

            # 筛选：在指定月份范围内
            df["date"] = pd.to_datetime(df["year"].astype(str) + df["doy"].astype(str).str.zfill(3), format="%Y%j")
            df["month"] = df["date"].dt.month
            df = df[(df["year"] > self.start_y) |
                    ((df["year"] == self.start_y) & (df["month"] >= self.start_m))]
            df = df[(df["year"] < self.end_y) |
                    ((df["year"] == self.end_y) & (df["month"] <= self.end_m))]

            # 对齐TEC：2小时间隔取物理数据 (0,2,4,...22)
            df = df[df["hour"] % 2 == 0]
            df = df.reset_index(drop=True)

            # 加入全局索引
            for _, row in df.iterrows():
                y = int(row["year"])
                m = int(row["month"])
                d = pd.to_datetime(f"{y}{int(row['doy']):03d}", format="%Y%j").day
                h = int(row["hour"])
                self.time_index.append([y, m, d, h, self.global_idx])
                self.global_idx += 1

        self.total_steps = self.global_idx
        self.valid_samples = self.total_steps - self.k * 12  # 每步=2小时，12步=1天
        self.tec_scaler = tec_scaler  # 不同数据缩放器不允许共同使用
        self.aux_scaler = aux_scaler
    # ===================== 按需加载单年数据（核心） =====================
    def _load_year_data(self, year):
        """ 只在需要时加载该年npy和csv，不常驻内存 """
        # 加载TEC
        tec_path = os.path.join(self.tec_dir, f"{year}_igsg.npy")
        tec_data = np.load(tec_path)  # (365,12,71,73)

        # 加载物理参数并2小时采样
        csv_path = os.path.join(self.indices_dir, f"{year}_indices.csv")
        df = pd.read_csv(csv_path)
        df = df[df["hour"] % 2 == 0].reset_index(drop = True)
        phys = df[["kp", "ssn", "dst", "ap", "f10.7", "ae"]].values.astype(np.float32)

        return tec_data, phys, df

    # ===================== 根据时间获取单步数据 =====================
    def _get_step_data(self, g_idx):
        """ 根据全局时序索引获取：tec(71,73) + phys(6,) """
        y, m, d, h, _ = self.time_index[g_idx]
        doy = pd.Timestamp(year=y, month=m, day=d).dayofyear
        step_in_day = h // 2  # 0~11

        # 加载年份数据
        tec_year, phys_year, df_year = self._load_year_data(y)

        # 定位数据
        tec = tec_year[doy - 1, step_in_day]  # (71,73)
        phys = phys_year[g_idx - self.year_offset[y]]

        return tec, phys

    # ===================== 数据集长度 =====================
    def __len__(self):
        return self.valid_samples

    # ===================== 获取样本 =====================
    def __getitem__(self, idx):
        """
        输入：idx ~ idx+k*12-1 共k天 → reshape(12*k,71,73)
        物理量：同步k天 → (12*k, 6)
        输出：未来1步 → (12,71,73)
        """
        # 1) 取连续 k*12 步（k天）
        tec_seq = []
        phys_seq = []
        for i in range(self.k * 12):
            tec, phys = self._get_step_data(idx + i)
            tec_seq.append(tec)
            phys_seq.append(phys)

        # 2) 取预测目标：第k天之后的1天（12步）
        tec_label = []
        phys_label = []
        for i in range(self.k * 12, self.k * 12 + 12):
            tec, phys = self._get_step_data(idx + i)
            tec_label.append(tec)
            phys_label.append(phys)

        # ===================== 维度拼接 =====================
        tec_in = np.stack(tec_seq, axis=0)  # (12*k, 71,73)
        phys_in = np.stack(phys_seq, axis=0)# (12*k,6)
        tec_gt = np.stack(tec_label, axis=0)# (12,71,73)
        phys_gt = np.stack(phys_label, axis=0) # (12,6)
        # =====================调用你的标准化函数=====================
        tec_in = scale_tec_aux_data(tec_in, self.tec_scaler, fit_scaler=self.is_train)
        tec_gt = scale_tec_aux_data(tec_gt, self.tec_scaler, fit_scaler=False)
        phys_in = scale_tec_aux_data(phys_in, self.aux_scaler, fit_scaler=self.is_train)        # 转张量
        phys_gt = scale_tec_aux_data(phys_gt, self.aux_scaler, fit_scaler=False)
        return (
            torch.from_numpy(tec_in).float(),
            torch.from_numpy(phys_in).float(),
            torch.from_numpy(tec_gt).float(),
            torch.from_numpy(phys_gt).float()
        )

if __name__ == "__main__":
    tec = TecIonosphereDataset(r"D:\Dataset_tec_NLY\tec_ionex_npy\igsg",r"D:\Dataset_tec_NLY\indices")