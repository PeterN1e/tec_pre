import cdflib
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import Dataset,DataLoader
from os import listdir
import os
import torch
import numpy as np
import pandas as pd  #分析结构化数据

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'




def detect_all_same_value(tec_data: np.ndarray) -> list:
    """
    检测71x73矩阵中所有元素为同一值（方差=0）的异常数据
    :param tec_data: 输入数据，shape=(n,71,73)
    :return: 异常矩阵的索引列表
    """
    anomaly_indices = []
    # 遍历每个71x73矩阵
    for idx in range(len(tec_data)):
        mat = tec_data[idx]
        # 方差为0 说明所有元素值相同
        if np.var(mat) == 0:
            # 获取该矩阵的唯一值
            same_value = mat[0, 0]
            anomaly_indices.append(idx)
            # 实时提示异常信息
            print(f"⚠️  索引{idx}的71x73矩阵：所有元素均为 {same_value}（方差=0，异常）")

    # 汇总检测结果
    print("=" * 70)
    if anomaly_indices:
        print(f"检测完成！共发现{len(anomaly_indices)}个异常矩阵，索引：{anomaly_indices}")
    else:
        print("检测完成！未发现所有元素为同一值的异常矩阵（所有矩阵方差均>0）")
    print("=" * 70)
    return anomaly_indices



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
        # cdf = cdflib.CDF(f"D:/Dataset______________/tec/tec_2011/{tec_data[0]}")
        tec = cdf.varget('tecUHR')
        for h in range(tec.shape[0]):
            tec_batch.append(tec[h])

    feature_list = pd.read_csv(feature_path_all)
    feature_list["datetime"] = pd.to_datetime(feature_list["datetime"]).dt.strftime('%Y%m%d')
    #将原先datatime数据更改格式
    tec_batch = np.array(tec_batch)
    aux = ["datetime","hour","ssn","dst","f10.7"]
    #detect_all_same_value(tec_batch)

    return tec_batch,np.array(feature_list[aux].values) #np.array可接受字符串，数字 布尔类型





# plt.show()
#
# plt.figure(figsize=(10,5))
# #proj = ccrs.PlateCarree()
# plt.pcolormesh(lon, lat, tec[0, :, :], shading='auto', cmap='plasma')
#
# plt.colorbar(label='TECU')
# plt.title(f'Global TEC (tecUHR) – {0}st epoch')
# plt.savefig(f'tecUHR_{0}.png', dpi=150)
# plt.show()
#
# proj = ccrs.PlateCarree()          # 等经纬度投影
# fig  = plt.figure(figsize=(9, 4.5))
# ax   = plt.axes(projection=proj)
#
# mesh = ax.pcolormesh(lon, lat, tec[1, :, :], transform=proj,
#                      shading='auto', cmap='jet')
# ax.coastlines()                    # 海岸线
# ax.gridlines(draw_labels=True)
# ax.set_global()                    # 显示全球
# fig.colorbar(mesh, ax=ax, label='TECU', orientation='horizontal')
# plt.show()