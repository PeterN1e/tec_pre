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

class TecDataset(Dataset):
    def __init__(self,day_seq, data_path):
        super().__init__()
        self.day_seq = day_seq
        self.data_path = data_path
        self.tec_batch =[]

        tec_path_all = os.path.join(self.data_path, "tecMap")
        feature_path_all = os.path.join(self.data_path, "omni_2011_complete.csv")
        tec_data_list = listdir(tec_path_all)
        # tec_data.sort()
        if day_seq == -1:
            day_seq = len(tec_data_list)
        # 用于读取tec图
        for day in range(self.day_seq):
            tec_absolute_path = os.path.join(self.data_path, "tecMap", tec_data_list[day])
            cdf = cdflib.CDF(f"{tec_absolute_path}")
            # cdf = cdflib.CDF(f"D:/Dataset______________/tec/tec_2011/{tec_data[0]}")
            tec = cdf.varget('tecUHR')
            for h in range(tec.shape[0]):
                self.tec_batch.append(tec[h])
        self.feature_list=pd.read_csv(feature_path_all)
    def __len__(self):
        return self.day_seq*24
    def __getitem__(self, index):
        return self.tec_batch[index],[self.feature_list["hour"][index],self.feature_list["ssn"][index],self.feature_list["dst"][index],self.feature_list["f10.7"][index]]


class TecDataset1(Dataset):
    def __init__(self,data_tec,data_aux,seq_length):
        super().__init__()
        self.data_aux = data_aux.astype(np.float32)
        self.data_tec = data_tec.astype(np.float32)
        self.seq_length =seq_length
    def __len__(self):
        return len(self.data_tec)-self.seq_length
    def __getitem__(self, index):
        return self.data_tec[index:index+self.seq_length],self.data_aux[index:index+self.seq_length],self.data_tec[index+self.seq_length]


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

    aux = ["hour","ssn","dst","f10.7"]
    return np.array(tec_batch),np.array(feature_list[aux].values)
    #<class 'numpy.ndarray'>:(6528, 71,73),(6528, 4)

#print(y.shape)

# i=0
# for x,y in train_dataloader:
#     print(x)
#     print(x)
#     i+=1
# print(i)
# exit()
#def tec_dataloader():
"""
TEC数据集类
"""


# tec1 = TecDataset(1,1,"D:\\Dataset______________\\tec\\2011\\TestDataset")
# print(tec1[0])
# exit()



# cdf  = cdflib.CDF('D:/Dataset______________/tec/tec_2011/gps_tec1hr_igs_20110101_v01.cdf')
#
# tec  = cdf.varget('tecUHR')   #.varget提取出指定变量 或者 tecEHR / tecCOD / tecCOR
#
# lat  = cdf.varget('lat')
# print(lat)
# print('lat shape :', lat.shape)  #lat shape : (71,)
#
# lon = cdf.varget('lon')         #lon shape : (73,)
# img = Image.fromarray(tec[0,:,:])
#
# print(tec[0,:,:])
# img.show()
# print('tec[0,:,:] shape :',tec[0,:,:].shape)
# print('tec[0,:,:] 类型 :',type(tec[0,:,:]))
# #exit()
#
# print('tec shape :', tec.shape)   # 预期 (time, lat, lon)
# print("tec size :", tec.size)
# print('lat shape :', lat.shape)
#
# print('lon shape :', lon.shape)
# print(type(lon))                  # 查看类型<class 'numpy.ndarray'>
# print(lon.dtype)                   #查看数据类型  float32
#
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