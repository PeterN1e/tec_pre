import matplotlib.pyplot as plt

def pic_show(act,pre):
    """
    :param pre: 接收格式为：(num,71,73)
    :param act: 接收格式为：(num,71,73)
    :return: 图片
    """
    picture_num = act.shape[0]
    error = pre - act
    plt.figure(figsize=(120, 10))
    for i in range(picture_num):

        plt.subplot(2, 12, i+1)
        plt.pcolormesh(act[i-1, :, :], shading='auto', cmap='jet',vmin = 0, vmax = 100)
        plt.colorbar(label='TECU')
        plt.title("tec act")
    plt.show()

    picture_num = pre.shape[0]
    plt.figure(figsize=(120, 10))
    for i in range(picture_num):
        plt.subplot(2, 12, i + 1)
        plt.pcolormesh(pre[i - 1, :, :], shading='auto', cmap='jet',vmin = 0, vmax = 100)
        plt.colorbar(label='TECU')
        plt.title("tec pre")
        # plt.savefig(f'tecUHR_{0}.png', dpi=150)
    plt.show()

    picture_num = error.shape[0]
    plt.figure(figsize=(120, 10))
    for i in range(picture_num):
        plt.subplot(2, 12, i + 1)
        plt.pcolormesh(error[i - 1, :, :], shading='auto', cmap='jet',vmin = 0, vmax = 100)
        plt.colorbar(label='TECU')
        plt.title("tec error")
        # plt.savefig(f'tecUHR_{0}.png', dpi=150)
    plt.show()