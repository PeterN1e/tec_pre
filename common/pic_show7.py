import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec
from config import TrainConfig,ModelConfig
cfg_model = ModelConfig()
cfg_train = TrainConfig()
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
def pic_show(act,pre,aux,delta):
    """
    :param delta:
    :param aux:
    :param pre: 接收格式为：(num,71,73)
    :param act: 接收格式为：(num,71,73)
    :return: 图片
    """
    lat = np.arange(87.5, -90, -2.5)
    lon = np.arange(-180, 185, 5)
    picture_num = act.shape[0]
    fig = plt.figure(figsize=(250, 15))
    gs = GridSpec(3,24,figure = fig,
                  width_ratios = [1]*24,
                  height_ratios=[1, 1, 1],
                  wspace=0.15,  # 水平间距
                  hspace=0.3)
    vmax_tec = np.max(act)
    vmax_error = np.max(delta)
    for i in range(picture_num):
        ax1 = fig.add_subplot(gs[0, i])#使用行列引索
        im1 =ax1.pcolormesh(lon, lat, act[i, :, :], shading='auto', cmap='jet', vmin=0, vmax=vmax_tec)

        plt.colorbar(im1,ax = ax1,label='TECU',shrink=1)
        title_right = f"SSN:{aux[i,2]:.0f}|  DST:{aux[i,3]:.0f}|  F10.7:{aux[i,4]:.1f}"
        plt.title(title_right,loc = 'right',fontsize=20)
        plt.title("真实图",loc = 'left', fontsize=20)

        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.pcolormesh(lon, lat, pre[i, :, :], shading='auto', cmap='jet', vmin=0, vmax=vmax_tec)

        plt.colorbar(im2,ax = ax2,label='TECU',shrink=1)
        plt.title("预测图", loc='left', fontsize=20)

        ax3 = fig.add_subplot(gs[2, i])
        im3 = ax3.pcolormesh(lon, lat, delta[i, :, :], shading='auto', cmap='jet', vmin=-vmax_error, vmax=vmax_error)
        plt.colorbar(im3,ax = ax3,label='TECU',shrink=1)
        average = np.mean(np.abs(delta[i, :, :]))#

        plt.title("差值图", loc='left', fontsize=20)
        plt.title(f"{average}", loc='center',fontsize=20)
        os.makedirs(cfg_train.pic_path, exist_ok=True)
        file_path = os.path.join(cfg_train.pic_path, 'r-p-d.png')
        plt.savefig(file_path)
    plt.show()

def datagram(data,label=None):
    colors = ['r','g','b','c','y']
    x = np.arange(0, data.shape[-1], 1)#定义x轴长度

    plt.figure(figsize=(15, 4))
    if data.ndim==2:
        for i in range(len(data)):
            y = data[i]
            y_mean = float(np.mean(y))
            #label='误差'
            plt.plot(x, y, color=colors[i], linewidth=1)
            plt.axhline(y=y_mean, color=colors[i], linestyle='--', linewidth=2, label=f'{label[i]}均值: {y_mean:.2f}')
    elif data.ndim==1:
        y = data
        y_mean = float(np.mean(y))
        plt.plot(x, y, label='误差', color='steelblue', linewidth=1)
        plt.axhline(y=y_mean, color='r', linestyle='--', linewidth=2, label=f'均值: {y_mean:.2f}')
    else:
        print("datagram作图处，传入维度错误")
        exit()
    plt.xlabel('天')
    plt.ylabel('差值/TECU')
    plt.title('误差变化曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model-compare.png')
    plt.show()