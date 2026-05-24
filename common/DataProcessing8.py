from config import ModelConfig,DatasetConfig,TrainConfig
cfg_model = ModelConfig()
cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()
import numpy as np
from os import listdir
import os
from common.pic_show7 import datagram

def data_evaluation(data):
    data_results = []
    if data.ndim==4:
        delta_abs = np.abs(data)  # 计算绝对值
        delta_average_one_hour = np.mean(delta_abs, axis=(2, 3))  # 对单张差值取平均值
        delta_average_one_day = np.mean(delta_average_one_hour, axis=1)  # 对24小时（每天）差值取平均值
    elif data.ndim==5:
        for i in range(data.shape[0]):
            delta_abs = np.abs(data[i])  # 计算绝对值
            delta_average_one_hour = np.mean(delta_abs, axis=(2, 3))  # 对单张差值取平均值
            delta_average_one_day = np.mean(delta_average_one_hour, axis=1)
            data_results.append(delta_average_one_day)
        delta_average_one_day = np.stack(data_results,axis = 0)
    else:
        print(f"{data_evaluation}维度传入错误")
        exit()
    return delta_average_one_day


def data_save(data):  #存入数据,格式为数组
    if not os.path.exists(f"Summarization_data\\{cfg_train.seq_length}to{cfg_train.pred_length}"):
        os.makedirs(f"Summarization_data\\{cfg_train.seq_length}to{cfg_train.pred_length}")
    np.savez(f"Summarization_data\\{cfg_train.seq_length}to{cfg_train.pred_length}\\{cfg_model.model_name}_data", a=data)

def data_loaded():#导出数据

    data = []
    data_list = listdir(f"Summarization_data\\{cfg_train.seq_length}to{cfg_train.pred_length}")
    for i in range(len(data_list)):
        data_cell = np.load(f"Summarization_data\\{cfg_train.pred_length}to{cfg_train.pred_length}\\{data_list[i]}", allow_pickle=True)
        data_cell = data_cell['a']
        data.append(data_cell)
        data_list[i] = data_list[i].split('_')[0]
    return np.stack(data,axis = 0),data_list

if __name__ == "__main__":

    a,b = data_loaded()
    c= data_evaluation(a)
    print(c.shape)

    datagram(data=c,label = b)

