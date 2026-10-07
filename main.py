import logging
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

from core.config import load_config, model_slug, resolve_output_dir
from core.registry import registered_models
from core.trainer import run_training
from core.inference import predict_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 仅用于菜单展示与排序；实际可用的模型以 registry 为准（见 _menu_models）
CANONICAL_MODELS = ["E_P_D", "ED_CGConvLSTM", "GA_Predrnn", "ED_Autoformer", "ModelCanon"]
EPD_PREDICTORS = {
    "1": "convlstm",
    "2": "convgru",
    "3": "tcn",
    "4": "transformer",
}


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


def _interactive_visualize(result, save_dir=None):
    """按需读取单一样本，避免把整份预测数组载入内存。"""
    from common.pic_show7 import pic_show

    arrays = result["arrays"]
    prediction = arrays["prediction"]
    target = arrays["target"]
    aux_target = arrays["aux_target"]

    total = int(prediction.shape[0])
    print(f"\n共 {total} 个样本，可输入索引 0~{total - 1} 查看可视化")
    for _ in range(10):
        try:
            idx = int(input(f"输入检索值 0~{total - 1}（q 退出）: "))
        except (ValueError, EOFError):
            break
        if 0 <= idx < total:
            pred_frame = np.asarray(prediction[idx], dtype=np.float32)
            target_frame = np.asarray(target[idx], dtype=np.float32)
            delta = target_frame - pred_frame
            pic_show(
                target_frame,
                pred_frame,
                np.asarray(aux_target[idx]),
                delta,
                save_dir=save_dir,
            )
            print("完成绘制")
        else:
            print("输入超出范围，退出")
            break


def _log_evaluation(metrics, model_name, output_dir):
    """将评估指标格式化后同时输出到控制台和 save/log/{model_name}.log。"""
    log_dir = os.path.join(str(output_dir), "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{model_name}.log")

    logger = logging.getLogger(f"eval.{model_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 显式走 sys.stdout，与 print()/input() 同一条流，避免 stdout/stderr 缓冲
    # 不一致导致评估结果被拼在可视化输入提示之后。
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    agg = metrics.get("aggregate", {})
    per_step = metrics.get("per_step", {})
    T = len(per_step.get("RMSE", []))
    max_h = 2 * T

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Evaluation Metrics")
    logger.info("=" * 60)
    for key in ("RMSE", "MAE", "R2", "SSIM"):
        if key in agg:
            logger.info(f"  {key:>6s} : {agg[key]:.6f}")
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Per-Step Metrics (t+2h ~ t+{max_h}h)")
    logger.info("=" * 60)
    logger.info(f"  {'Step':>6s} {'Horizon':>8s} | {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'SSIM':>8s}")
    logger.info("  " + "-" * 56)
    for t in range(T):
        h = 2 * (t + 1)
        horizon = f"t+{h}h"
        logger.info(
            f"  {t+1:>6d} {horizon:>8s} | "
            f"{per_step['RMSE'][t]:8.4f} {per_step['MAE'][t]:8.4f} "
            f"{per_step['R2'][t]:8.4f} {per_step['SSIM'][t]:8.4f}"
        )
    logger.info("=" * 60)
    fh.flush()
    sys.stdout.flush()
    logger.removeHandler(fh)
    logger.removeHandler(sh)
    fh.close()


def _menu_models():
    """菜单候选以注册表为准，列表与已注册模型脱节时立即报错而不是静默漏项。"""
    registered = set(registered_models())
    models = [name for name in CANONICAL_MODELS if name.lower() in registered]
    missing = sorted(registered - {name.lower() for name in models})
    if missing:
        raise RuntimeError(
            f"模型已注册但不在菜单里，请更新 CANONICAL_MODELS: {missing}"
        )
    return models


def _choose(prompt, options):
    """options: {编号: 值}。非法输入重新提问，不静默回退到默认值。"""
    while True:
        raw = input(prompt).strip()
        if raw in options:
            return options[raw]
        print(f"  输入无效，请输入 {' / '.join(sorted(options))} 之一")


def main():
    warnings.filterwarnings('ignore')

    print("=" * 40)
    print("  电离层 TEC 预测系统")
    print("=" * 40)

    models = _menu_models()
    print("\n选择模型:")
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")
    model_name = _choose(
        f"输入模型编号 (1-{len(models)}): ",
        {str(i): name for i, name in enumerate(models, 1)},
    )

    overrides = []
    if model_name == "E_P_D":
        print("\nE_P_D 时序预测器:")
        for i, (number, predictor) in enumerate(EPD_PREDICTORS.items(), 1):
            print(f"  {i}. {predictor}")
        predictor = _choose(
            f"选择预测器 (1-{len(EPD_PREDICTORS)}): ",
            {str(i): name for i, (_, name) in enumerate(EPD_PREDICTORS.items(), 1)},
        )
        overrides.append(f"model.params.predictor_name={predictor}")

    print("\n选择操作:")
    print("  1. 训练 + 训练后推理评估")
    print("  2. 仅训练")
    print("  3. 仅推理评估")
    op = _choose(
        "输入操作编号 (1-3): ",
        {"1": ("train", "predict"), "2": ("train",), "3": ("predict",)},
    )

    config = load_config(model_name=model_name, overrides=overrides)
    output_dir = resolve_output_dir(config)
    slug = model_slug(config["model"]["name"])
    model_dir = str(output_dir / slug)

    training_cfg = config["training"]
    data_cfg = config["data"]
    print(
        f"\n[{config['model']['name']}] epochs={training_cfg['epochs']} "
        f"lr={training_cfg['lr']} batch_size={data_cfg['batch_size']} "
        f"input_length={data_cfg['input_length']} "
        f"output_length={data_cfg['output_length']}"
    )
    print(f"  这些超参数来自 configs/base.yaml + configs/models/{slug}.yaml")

    if "train" in op:
        print(f"\n开始训练 [{model_name}] ...")
        result = run_training(config)
        print("训练完成!")
        _plot_loss(result["history"], model_dir)

    if "predict" in op:
        print(f"\n开始推理评估 [{model_name}] ...")
        result = predict_split(config, split="test")
        _log_evaluation(result["metrics"], model_name, output_dir)

        try:
            _interactive_visualize(result, save_dir=os.path.join(model_dir, "pic"))
        except Exception as e:
            print(f"可视化跳过: {e}")


if __name__ == "__main__":
    main()
