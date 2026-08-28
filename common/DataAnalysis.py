"""
TEC 日间差值分析：计算前一天和后一天同一时刻 TEC 图像的差值并展示。

Δtec = tec(day_d, hour_h) - tec(day_d-1, hour_h)

同一时刻在不同天相差 12 帧（2h 分辨率，一天 12 帧）。
"""

import numpy as np
from config import DatasetConfig, dataset_base_path,DataAnalysisConfig
from common.dataloader1 import TecIonosphereDataset
from common.pic_show7 import pic_show


def compute_daily_diff(ds, start_global_idx):
    """
    取连续两天同一时刻的 TEC，计算差值。

    Args:
        ds: TecIonosphereDataset 实例
        start_global_idx: 起始全局索引（前一天 0h 对应位置）

    Returns:
        act: 后一天 TEC, shape (12, 71, 73)
        pre: 前一天 TEC, shape (12, 71, 73)
        aux: 后一天辅助数据, shape (12, 6)
        delta: 差值 act - pre, shape (12, 71, 73)
    """
    tec_prev = []
    tec_curr = []
    aux_curr = []

    for h in range(12):
        tec_p, phys_p = ds._get_step_data(start_global_idx + h)
        tec_c, phys_c = ds._get_step_data(start_global_idx + 12 + h)

        tec_prev.append(tec_p)
        tec_curr.append(tec_c)
        aux_curr.append(phys_c)

    act = np.array(tec_curr)
    pre = np.array(tec_prev)
    aux = np.array(aux_curr)
    delta = act - pre

    return act, pre, aux, delta


def main():
    cfg_dataset = DatasetConfig()

    # 构建数据集（仅用于获取 time_index 和 _get_step_data，不调用 __getitem__）
    ds = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=DataAnalysisConfig.start_month_analysis,
        end_month=DataAnalysisConfig.end_month_analysis,
        input_day_num=3,
        is_train=False,
        tec_scaler=None,d
        aux_scaler=None,
    )

    total_steps = ds.total_steps
    max_start = total_steps - 24  # 需要 24 帧（两天）
    print(f"数据集总步数: {total_steps}，可分析范围: [0, {max_start}]")

    while True:
        try:
            idx = int(input(f"输入起始全局索引（0 ~ {max_start}，-1 退出）: "))
        except ValueError:
            print("请输入整数")
            continue

        if idx == -1:
            break
        if idx < 0 or idx > max_start:
            print(f"索引超出范围，请输入 0 ~ {max_start}")
            continue

        info_prev = ds.time_index[idx]
        info_curr = ds.time_index[idx + 12]
        print(f"前一天: {info_prev[0]}-{info_prev[1]:02d}-{info_prev[2]:02d} {info_prev[3]:02d}h")
        print(f"后一天: {info_curr[0]}-{info_curr[1]:02d}-{info_curr[2]:02d} {info_curr[3]:02d}h")

        act, pre, aux, delta = compute_daily_diff(ds, idx)
        print(f"ΔTEC 范围: [{delta.min():.2f}, {delta.max():.2f}] TECU, 均值: {delta.mean():.2f} TECU")

        pic_show(act, pre, aux, delta)


if __name__ == "__main__":
    main()
