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
                 tec_dir,  # 存放 2005_igsg.npy 的文件夹
                 indices_dir,  # 存放 2005_indices.csv 的文件夹
                 start_month=200201,  # 数据集起始年月（如200201）
                 end_month=201012,  # 数据集结束年月
                 his_day_num=3,  # 用k天作为输入
                 is_train=False,
                 tec_scaler=None,
                 aux_scaler=None):
        super().__init__()
        self.tec_dir = tec_dir
        self.indices_dir = indices_dir
        self.his_day_num = his_day_num
        self.is_train = is_train
        self.tec_scaler = tec_scaler
        self.aux_scaler = aux_scaler
        self._year_cache = {}  # 年份数据缓存：key=年份, value=(tec_data, phys_data, df)

        # 解析起止年月
        self.start_y = start_month // 100
        self.start_m = start_month % 100
        self.end_y = end_month // 100
        self.end_m = end_month % 100

        # ===================== 扫描所有年份文件 =====================
        self.years = sorted([
            int(f.split("_")[0])
            for f in os.listdir(tec_dir)
            if f.endswith("igsg.npy")
        ])
        self.years = [y for y in self.years if self.start_y <= y <= self.end_y]

        # ===================== 构建全局时间索引表（核心） =====================
        self.time_index = []  # 每条记录: [year, month, day, hour, global_idx]
        self.global_idx = 0  # 整个时序的唯一索引
        self.year_offset = {}  # 记录每年在全局索引中的起始位置

        for year in self.years:
            self.year_offset[year] = self.global_idx  # 记录该年的起始全局索引
            # 加载该年CSV，获取完整时间轴
            csv_path = os.path.join(indices_dir, f"{year}_indices.csv")
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
        # 修正样本数：输入长度 + 预测长度 不能超过总步数
        self.valid_samples = self.total_steps - self.his_day_num * 12 - 12 + 1

        # ===================== 训练集专属：预加载+拟合Scaler =====================
        if self.is_train:
            print("正在预加载训练集数据并拟合标准化器...")
            all_tec = []
            all_aux = []
            for year in self.years:
                tec_data, phys_data, _ = self._load_year_data(year)
                # 截取当前年份在训练范围内的时间步
                year_start = self.year_offset[year]
                if year == self.years[-1]:
                    year_end = self.total_steps
                else:
                    year_end = self.year_offset[year + 1]
                step_count = year_end - year_start
                all_tec.append(tec_data.reshape(-1, 71, 73)[:step_count])
                all_aux.append(phys_data[:step_count])

            all_tec = np.concatenate(all_tec, axis=0)
            all_aux = np.concatenate(all_aux, axis=0)
            # 一次性拟合标准化器
            scale_tec_aux_data(all_tec, self.tec_scaler, fit_scaler=True)
            scale_tec_aux_data(all_aux, self.aux_scaler, fit_scaler=True)
            print(f"标准化器拟合完成，训练集共 {len(all_tec)} 个时间步已缓存")

    # ===================== 带缓存的年份数据加载 =====================
    def _load_year_data(self, year):
        """ 加载过的年份直接从缓存返回，不再重复读硬盘 """
        if year in self._year_cache:
            return self._year_cache[year]

        tec_path = os.path.join(self.tec_dir, f"{year}_igsg.npy")
        tec_data = np.load(tec_path)  # shape: (365, 12, 71, 73)

        csv_path = os.path.join(self.indices_dir, f"{year}_indices.csv")
        df = pd.read_csv(csv_path)
        df = df[df["hour"] % 2 == 0].reset_index(drop=True)
        phys = df[["kp", "ssn", "dst", "ap", "f10.7", "ae"]].values.astype(np.float32)

        self._year_cache[year] = (tec_data, phys, df)
        return tec_data, phys, df

    # ===================== 根据全局索引取单步数据 =====================
    def _get_step_data(self, g_idx):
        y, m, d, h, _ = self.time_index[g_idx]
        doy = pd.Timestamp(year=y, month=m, day=d).dayofyear
        step_in_day = h // 2
        tec_year, phys_year, _ = self._load_year_data(y)
        return tec_year[doy - 1, step_in_day], phys_year[g_idx - self.year_offset[y]]

    # ===================== 数据集长度 =====================
    def __len__(self):
        return self.valid_samples

    # ===================== 获取单个样本 =====================
    def __getitem__(self, idx):
        # 输入序列：his_day_num 天
        tec_seq, phys_seq = [], []
        for i in range(self.his_day_num * 12):
            tec, phys = self._get_step_data(idx + i)
            tec_seq.append(tec)
            phys_seq.append(phys)

        # 标签序列：未来1天（12步）
        tec_label, phys_label = [], []
        for i in range(self.his_day_num * 12, self.his_day_num * 12 + 12):
            tec, phys = self._get_step_data(idx + i)
            tec_label.append(tec)
            phys_label.append(phys)

        # 维度拼接
        tec_in = np.stack(tec_seq, axis=0)
        phys_in = np.stack(phys_seq, axis=0)
        tec_gt = np.stack(tec_label, axis=0)
        phys_gt = np.stack(phys_label, axis=0)

        # 标准化（全部仅transform，不再fit）
        tec_in = scale_tec_aux_data(tec_in, self.tec_scaler, fit_scaler=False)
        tec_gt = scale_tec_aux_data(tec_gt, self.tec_scaler, fit_scaler=False)
        phys_in = scale_tec_aux_data(phys_in, self.aux_scaler, fit_scaler=False)
        phys_gt = scale_tec_aux_data(phys_gt, self.aux_scaler, fit_scaler=False)

        return (
            torch.from_numpy(tec_in).float(),
            torch.from_numpy(phys_in).float(),
            torch.from_numpy(tec_gt).float(),
            torch.from_numpy(phys_gt).float()
        )

if __name__ == "__main__":
    tec = TecIonosphereDataset(r"D:\Dataset_tec_NLY\tec_ionex_npy\igsg",r"D:\Dataset_tec_NLY\indices")