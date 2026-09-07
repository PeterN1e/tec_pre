"""
Baseline comparison for ModelCanon: copy the last 12 input frames
(same-hour day baseline) and report the same metrics used by main.py.

Run from the repo root:
    D:/Anaconda/envs/tec_prediction/python.exe common/baseline_eval.py
"""

import os
import sys

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from common.Data_Preprocessing import inverse_transform_predictions
from common.EvaluationMetrics import log_evaluation, print_evaluation
from common.dataloader1 import TecIonosphereDataset
from config import DatasetConfig, TrainConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cfg_dataset = DatasetConfig()
cfg_train = TrainConfig()


def _get_test_pairs():
    tec_scaler = joblib.load(os.path.join(cfg_train.model_path, "tec_scaler.pkl"))
    aux_scaler = joblib.load(os.path.join(cfg_train.model_path, "aux_scaler.pkl"))

    test_dataset = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=cfg_dataset.start_month_test,
        end_month=cfg_dataset.end_month_test,
        input_day_num=cfg_train.input_day_num,
        is_train=False,
        tec_scaler=tec_scaler,
        aux_scaler=aux_scaler,
    )
    loader = DataLoader(
        test_dataset,
        batch_size=cfg_train.batch_size,
        shuffle=False,
        drop_last=True,
    )
    inputs, targets = [], []
    for batch_in_tec, _, batch_exp_tec, _ in loader:
        inputs.append(batch_in_tec.float())
        targets.append(batch_exp_tec.float())
    return (
        torch.cat(inputs, dim=0).numpy(),
        torch.cat(targets, dim=0).numpy(),
        tec_scaler,
    )


def evaluate_copy_last_day():
    tec_in, tec_gt, tec_scaler = _get_test_pairs()
    pred_raw_norm = tec_in[:, -12:, :, :]            # copy day-3 same-hour frames
    pred_abs = inverse_transform_predictions(pred_raw_norm, tec_scaler)
    act_abs = inverse_transform_predictions(tec_gt, tec_scaler)

    print("=" * 60)
    print("  Baseline: copy last 12 input frames (TEC[t-12])")
    print("=" * 60)
    print_evaluation(pred_abs, act_abs)

    import logging
    logger = logging.getLogger("baseline.copy_last_day")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(
        os.path.join(cfg_train.log_path, "baseline_copy_last_day.log"),
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    log_evaluation(logger, pred_abs, act_abs)

    print("\nLog saved to", os.path.join(cfg_train.log_path, "baseline_copy_last_day.log"))


def evaluate_model_copy_last_day():
    tec_in, tec_gt, tec_scaler = _get_test_pairs()

    import config as _cfg
    _cfg.model_name = "ModelCanon"
    from model_selector import ModelAll

    model = ModelAll().to(cfg_train.device).eval()
    save_dir = "ModelCanon"
    model.load_state_dict(
        torch.load(
            os.path.join(cfg_train.model_path, save_dir, "model_state_dict.pth"),
            map_location=cfg_train.device,
            weights_only=True,
        )
    )

    tec_t = torch.from_numpy(tec_in).float()
    aux_t = torch.randn(tec_t.shape[0], tec_t.shape[1], 6)
    with torch.no_grad():
        pred_raw_norm = model(tec_t.to(cfg_train.device), aux_t.to(cfg_train.device)).cpu().numpy()

    pred_abs = inverse_transform_predictions(pred_raw_norm, tec_scaler)
    act_abs = inverse_transform_predictions(tec_gt, tec_scaler)
    print_evaluation(pred_abs, act_abs)


if __name__ == "__main__":
    evaluate_copy_last_day()
