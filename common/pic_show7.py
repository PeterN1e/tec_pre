import matplotlib.pyplot as plt

from config import TrainConfig,ModelConfig
cfg_model = ModelConfig()
cfg_train = TrainConfig()
plt.rcParams['font.sans-serif'] = [
    'SimHei',  # Windows 黑体
    'WenQuanYi Micro Hei',  # Linux 文泉驿
]
plt.rcParams['axes.unicode_minus'] = False


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def pic_show(act, pre, aux, delta):
    """
    :param delta: (num,71,73)
    :param aux: (num, ...)
    :param pre: (num,71,73)
    :param act: (num,71,73)
    :return: 保存并显示图片
    """
    lat = np.arange(87.5, -90, -2.5)
    lon = np.arange(-180, 185, 5)
    picture_num = act.shape[0]  # 12

    # ========== 核心：画布尺寸 ==========
    fig = plt.figure(figsize=(42, 7), dpi=120)

    # GridSpec：3行，13列（12个数据子图 + 1个色标列）
    gs = GridSpec(3, 13, figure=fig,
                  width_ratios=[1] * 12 + [0.08],  # 最后一列给色标
                  height_ratios=[1, 1, 1],
                  wspace=0.18,  # 水平间距
                  hspace=0.35)  # 垂直间距

    vmax_tec = np.max(act)
    vmax_error = np.max(np.abs(delta))  # 差值用对称范围

    # 行标题（放在每行首部）
    row_titles = ["真实图", "预测图", "差值图"]

    for i in range(picture_num):
        # ---- 第1行：真实图 ----
        ax1 = fig.add_subplot(gs[0, i])
        im1 = ax1.pcolormesh(lon, lat, act[i, :, :],
                             shading='auto', cmap='jet',
                             vmin=0, vmax=vmax_tec)
        # 右上角参数信息
        title_right = f"SSN:{aux[i, 2]:.0f} | DST:{aux[i, 3]:.0f} | F10.7:{aux[i, 4]:.1f}"
        ax1.set_title(title_right, loc='right', fontsize=8)
        # 只在第1列加行标题（首部）
        if i == 0:
            ax1.set_title(row_titles[0], loc='left', fontsize=8, fontweight='bold')

        # ---- 第2行：预测图 ----
        ax2 = fig.add_subplot(gs[1, i])
        im2 = ax2.pcolormesh(lon, lat, pre[i, :, :],
                             shading='auto', cmap='jet',
                             vmin=0, vmax=vmax_tec)
        if i == 0:
            ax2.set_title(row_titles[1], loc='left', fontsize=8, fontweight='bold')

        # ---- 第3行：差值图 ----
        ax3 = fig.add_subplot(gs[2, i])
        im3 = ax3.pcolormesh(lon, lat, delta[i, :, :],
                             shading='auto', cmap='jet',
                             vmin=-vmax_error, vmax=vmax_error)
        # 子图中间显示平均绝对误差
        average = np.mean(np.abs(delta[i, :, :]))
        ax3.set_title(f"MAE:{average:.2f}", loc='center', fontsize=8)
        if i == 0:
            ax3.set_title(row_titles[2], loc='left', fontsize=8, fontweight='bold')

    # ---- 每行添加共享色标（放在最右侧） ----
    cbar_ax1 = fig.add_subplot(gs[0, 12])
    fig.colorbar(im1, cax=cbar_ax1, label='TECU')

    cbar_ax2 = fig.add_subplot(gs[1, 12])
    fig.colorbar(im2, cax=cbar_ax2, label='TECU')

    cbar_ax3 = fig.add_subplot(gs[2, 12])
    fig.colorbar(im3, cax=cbar_ax3, label='TECU')

    # 保存图片（移到循环外，避免覆盖）
    os.makedirs(cfg_train.pic_path, exist_ok=True)
    file_path = os.path.join(cfg_train.pic_path, 'r-p-d.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=150)
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