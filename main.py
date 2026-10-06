import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from core.config import load_config, model_slug, resolve_output_dir
from core.trainer import resolve_device, run_training
from core.inference import predict_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

MODEL_CHOICES = ["E_P_D", "ED_CGConvLSTM", "GA_Predrnn", "ED_Autoformer", "ModelCanon"]
EPD_PREDICTORS = {"1": "convlstm", "2": "convgru", "3": "tcn", "4": "transformer"}


def _plot_loss(history, model_dir):
    if isinstance(history, dict):
        train_losses = history.get("train_loss") or history.get("train_g", [])
        val_losses = history.get("val_loss", [])
    elif isinstance(history, (tuple, list)) and len(history) == 2:
        train_losses, val_losses = history
    else:
        return

    if not train_losses:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
    ax1.plot(train_losses, label='training loss')
    ax1.plot(val_losses, label='test loss')
    ax1.set_title("model loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_losses, label='training loss')
    ax2.plot(val_losses, label='test loss')
    ax2.set_title('model loss(logarithmic scale)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('loss(logarithmic scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pic_dir = os.path.join(model_dir, "pic")
    os.makedirs(pic_dir, exist_ok=True)
    plt.savefig(os.path.join(pic_dir, "train_loss.png"))
    plt.show()


def _interactive_visualize(result):
    from common.pic_show7 import pic_show

    prediction = result["arrays"]["prediction"]
    target = result["arrays"]["target"]
    aux_target = np.load(result["predictions_path"])["aux_target"]
    delta = target - prediction

    total = prediction.shape[0]
    print(f"\n共 {total} 个样本，可输入索引 0~{total - 1} 查看可视化")
    for _ in range(10):
        try:
            idx = int(input(f"输入检索值 0~{total - 1}（q 退出）: "))
        except (ValueError, EOFError):
            break
        if 0 <= idx < total:
            pic_show(target[idx], prediction[idx], aux_target[idx], delta[idx])
            print("完成绘制")
        else:
            print("输入超出范围，退出")
            break


def main():
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    print("=" * 40)
    print("  电离层 TEC 预测系统")
    print("=" * 40)

    print("\n选择模型:")
    for i, name in enumerate(MODEL_CHOICES, 1):
        print(f"  {i}. {name}")
    model_idx = input("输入模型编号 (1-5): ").strip()
    if model_idx not in [str(i) for i in range(1, 6)]:
        print("输入错误，退出")
        return
    model_name = MODEL_CHOICES[int(model_idx) - 1]

    overrides = []
    if model_name == "E_P_D":
        print("\nE_P_D 时序预测器:")
        print("  1. ConvLSTM")
        print("  2. ConvGRU")
        print("  3. TCN")
        print("  4. Transformer")
        pred_idx = input("选择预测器 (1-4，默认 1): ").strip() or "1"
        predictor = EPD_PREDICTORS.get(pred_idx, "convlstm")
        overrides.append(f"model.params.predictor={predictor}")
        print(f"已选择: {predictor}")

    print("\n选择操作:")
    print("  1. 训练 + 训练后推理")
    print("  2. 仅训练")
    print("  3. 仅推理评估")
    op = input("输入操作编号 (1-3): ").strip()
    if op not in ("1", "2", "3"):
        print("输入错误，退出")
        return

    config = load_config(model_name=model_name, overrides=overrides)
    device = resolve_device(config.get("device", "auto"))
    output_dir = resolve_output_dir(config)
    slug = model_slug(model_name)
    model_dir = str(output_dir / slug)

    if op in ("1", "2"):
        print(f"\n开始训练 [{model_name}] ...")
        result = run_training(config)
        print("训练完成!")
        _plot_loss(result["history"], model_dir)

    if op in ("1", "3"):
        print(f"\n开始推理评估 [{model_name}] ...")
        result = predict_split(config, split="test")

        agg = result["metrics"].get("aggregate", {})
        print("\n===== 评估结果 =====")
        for key in ("rmse", "mae", "r2", "ssim"):
            if key in agg:
                print(f"  {key.upper()}: {agg[key]:.4f}")
        print(f"  指标已保存: {result['metrics_path']}")

        try:
            _interactive_visualize(result)
        except Exception as e:
            print(f"可视化跳过: {e}")


if __name__ == "__main__":
    main()
