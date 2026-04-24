import pandas as pd
import matplotlib.pyplot as plt
from config import model_name
import numpy as np

def data_save(data):  #存入数据
    np.savez(f"Summarization_data\\{model_name}_data", a=data)

if __name__ == "__main__":
    a = np.random.randn(5,6,7,8)
    data_save(a)
    loaded_arr = np.load(f"Summarization_data\\{model_name}_data.npz",allow_pickle=True)


    b = loaded_arr['a']
    print(b.shape)
    print(type(b))