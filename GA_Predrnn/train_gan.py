"""
GA-Predrnn  GAN cross-training script

Usage (from repo root):
    python -m GA_Predrnn.train_gan

or directly:
    python GA_Predrnn/train_gan.py

What it does:
    1. builds TecIonosphereDataset  (input_day_num=2 -> 24 steps)
    2. builds GAPredrnnPredictor + Discriminator
    3. trains with hinge-GAN cross-training (2 backward passes / batch)
    4. validates with predictor only (RMSE / MAE / R2 / SSIM)
    5. saves best model weights to save/model_dict/ga_predrnn/
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time
import subprocess

# ---- project imports ----
from config import DatasetConfig, TrainConfig
from common.dataloader1 import TecIonosphereDataset
from common.EvaluationMetrics import evaluate_all
from GA_Predrnn.GA_Predrnn import GAPredrnnPredictor
from GA_Predrnn.discriminator import Discriminator, hinge_loss_d, hinge_loss_g


# ================================================================== #
#  hyper-parameters  (override TrainConfig where needed)
# ================================================================== #

INPUT_DAY_NUM   = 2          # paper setting: 2 days -> 24 steps
OUTPUT_DAY_NUM  = 1          # 1 day  -> 12 steps
INPUT_LENGTH    = INPUT_DAY_NUM  * 12   # 24
OUTPUT_LENGTH   = OUTPUT_DAY_NUM * 12   # 12

HIDDEN_DIM      = 64
NUM_LAYERS      = 3
KERNEL_SIZE     = 5
BLOCK_SIZE      = 8
HALO_SIZE       = 2
NUM_HEADS       = 4
DISC_BASE_CH    = 64

LAMBDA_TEC      = 1.0        # L1 weight for TEC
LAMBDA_AUX      = 0.1        # L1 weight for auxiliary data (a >> b)
D_LR            = 1e-4       # discriminator learning rate
G_LR            = 1e-3       # predictor    learning rate
EPOCHS          = 50
BATCH_SIZE      = 4
PATIENCE        = 10         # early stopping patience
CLIP_GRAD       = 1.0


# ================================================================== #
#  logging
# ================================================================== #
cfg_train = TrainConfig()
cfg_dataset = DatasetConfig()

SEP = "=" * 80

MODEL_NAME = "GA_Predrnn"
log_file = cfg_train.log_path / f"{MODEL_NAME}.log"
log_file.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(f"train.{MODEL_NAME}")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(str(log_file), encoding="utf-8")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(fmt)
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)


def _get_gpu_info():
    """Return (mem_used_mb, mem_total_mb, gpu_util_pct) or (0, 0, "N/A") on failure."""
    if not torch.cuda.is_available():
        return 0, 0, "N/A"
    try:
        mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        util = "N/A"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                util = result.stdout.strip() + "%"
        except Exception:
            pass
        return int(mem_reserved), int(mem_total), util
    except Exception:
        return 0, 0, "N/A"


# ================================================================== #
#  training helpers
# ================================================================== #

def train_one_epoch(predictor, discriminator, loader, opt_g, opt_d, device):
    predictor.train()
    discriminator.train()
    running_d, running_g = 0.0, 0.0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=100)
    for tec_in, aux_in, tec_gt, aux_gt in pbar:
        tec_in = tec_in.to(device)            # (B, T_in, H, W)
        aux_in = aux_in.to(device)            # (B, T_in, 6)
        tec_gt = tec_gt.to(device)            # (B, T_out, H, W)
        aux_gt = aux_gt.to(device)            # (B, T_out, 6)

        B, T_out, H, W = tec_gt.shape

        # ---- 1. predictor forward ----
        pred_tec, pred_aux = predictor.train_forward(tec_in, aux_in)
        # pred_tec: (B, T_out, H, W)   pred_aux: (B, T_out, 3)

        # ---- 2. discriminator step ----
        real = tec_gt.unsqueeze(2)                           # (B, T_out, 1, H, W)
        fake = pred_tec.unsqueeze(2).detach()                # detach to avoid generator grad
        real = real.reshape(B * T_out, 1, H, W)
        fake = fake.reshape(B * T_out, 1, H, W)

        score_real = discriminator(real)
        score_fake = discriminator(fake)
        loss_d = hinge_loss_d(score_real, score_fake)

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # ---- 3. predictor step ----
        # re-score fake with updated discriminator
        fake_for_g = pred_tec.unsqueeze(2).reshape(B * T_out, 1, H, W)
        score_fake_g = discriminator(fake_for_g)

        loss_tec = F.l1_loss(pred_tec, tec_gt)
        # aux_gt columns [2,3,4] = dst, ap, f10.7
        loss_aux = F.l1_loss(pred_aux, aux_gt[:, :, [2, 3, 4]])
        loss_g = LAMBDA_TEC * loss_tec + LAMBDA_AUX * loss_aux + hinge_loss_g(score_fake_g)

        opt_g.zero_grad()
        loss_g.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), CLIP_GRAD)
        opt_g.step()

        running_d += loss_d.item()
        running_g += loss_g.item()
        pbar.set_postfix(loss_D=f"{loss_d.item():.4f}", loss_G=f"{loss_g.item():.4f}")

    n = len(loader)
    return running_d / n, running_g / n


@torch.no_grad()
def validate(predictor, loader, device):
    """Validation with predictor only (no discriminator)."""
    predictor.eval()
    all_pred, all_true = [], []
    val_loss = 0.0

    for tec_in, aux_in, tec_gt, aux_gt in loader:
        tec_in = tec_in.to(device)
        aux_in = aux_in.to(device)
        tec_gt = tec_gt.to(device)

        pred_tec = predictor(tec_in, aux_in)          # (B, T_out, H, W)
        val_loss += F.l1_loss(pred_tec, tec_gt).item()

        all_pred.append(pred_tec.cpu().numpy())
        all_true.append(tec_gt.cpu().numpy())

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    metrics = evaluate_all(pred_np, true_np)

    avg_loss = val_loss / max(len(loader), 1)
    return avg_loss, metrics


# ================================================================== #
#  main
# ================================================================== #

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = cfg_train.device

    # ---- data ----
    tec_scaler = MinMaxScaler()
    aux_scaler = MinMaxScaler()

    train_ds = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=cfg_dataset.start_month_train,
        end_month=cfg_dataset.end_month_train,
        input_day_num=INPUT_DAY_NUM,
        is_train=True,
        tec_scaler=tec_scaler,
        aux_scaler=aux_scaler,
    )
    val_ds = TecIonosphereDataset(
        tec_dir=cfg_dataset.tec_dir,
        indices_dir=cfg_dataset.indices_dir,
        start_month=cfg_dataset.start_month_val,
        end_month=cfg_dataset.end_month_val,
        input_day_num=INPUT_DAY_NUM,
        is_train=False,
        tec_scaler=tec_scaler,
        aux_scaler=aux_scaler,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    logger.info(f"train samples: {len(train_ds)},  val samples: {len(val_ds)}")

    # ---- models ----
    predictor = GAPredrnnPredictor(
        input_dim=4,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        input_length=INPUT_LENGTH,
        output_length=OUTPUT_LENGTH,
        block_size=BLOCK_SIZE,
        halo_size=HALO_SIZE,
        num_heads=NUM_HEADS,
    ).to(device)

    discriminator = Discriminator(base_ch=DISC_BASE_CH).to(device)

    logger.info(f"predictor     params: {sum(p.numel() for p in predictor.parameters()):,}")
    logger.info(f"discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ---- log header ----
    total_params = sum(p.numel() for p in predictor.parameters()) + sum(p.numel() for p in discriminator.parameters())
    logger.info(SEP)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Hyperparameters: batch_size={BATCH_SIZE}, epochs_num={EPOCHS}, patience={PATIENCE}, g_lr={G_LR}, d_lr={D_LR}")
    logger.info(f"Dataset: Train: {cfg_dataset.start_month_train}-{cfg_dataset.end_month_train}, Val: {cfg_dataset.start_month_val}-{cfg_dataset.end_month_val}")
    logger.info(f"Parameters: predictor={sum(p.numel() for p in predictor.parameters()):,}, discriminator={sum(p.numel() for p in discriminator.parameters()):,}, total={total_params:,}")
    logger.info(f"Loss Function: HingeGAN (L1_tec + L1_aux + hinge)")
    logger.info(SEP)

    opt_g = optim.Adam(predictor.parameters(),     lr=G_LR)
    opt_d = optim.Adam(discriminator.parameters(), lr=D_LR)
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=0.5, patience=5)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode="min", factor=0.5, patience=5)

    # ---- save dirs ----
    model_dir = cfg_train.model_path / "ga_predrnn"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    best_val = float("inf")
    patience_counter = 0
    history = {"train_d": [], "train_g": [], "val_loss": [], "val_rmse": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"===== Epoch {epoch}/{EPOCHS} =====")

        loss_d, loss_g = train_one_epoch(
            predictor, discriminator, train_loader, opt_g, opt_d, device
        )
        val_loss, val_metrics = validate(predictor, val_loader, device)

        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)

        history["train_d"].append(loss_d)
        history["train_g"].append(loss_g)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_metrics["RMSE"])

        current_lr = opt_g.param_groups[0]["lr"]
        gpu_mem_used, gpu_mem_total, gpu_util = _get_gpu_info()

        logger.info(
            f"Epoch {epoch:3d} | "
            f"D_loss {loss_d:.5f} | G_loss {loss_g:.5f} | "
            f"val_L1 {val_loss:.5f} | "
            f"RMSE {val_metrics['RMSE']:.4f} R2 {val_metrics['R2']:.4f} "
            f"SSIM {val_metrics['SSIM']:.4f} | "
            f"LR {current_lr:.2e} | "
            f"GPU Mem {gpu_mem_used}/{gpu_mem_total} MB | "
            f"GPU Util {gpu_util}"
        )

        # early stopping on val_loss
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(predictor.state_dict(),     model_dir / "predictor.pth")
            torch.save(discriminator.state_dict(), model_dir / "discriminator.pth")
            logger.info(f"  best model saved (val_loss={val_loss:.5f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ---- save scalers ----
    joblib.dump(tec_scaler, model_dir / "tec_scaler.pkl")
    joblib.dump(aux_scaler, model_dir / "aux_scaler.pkl")

    # ---- save plots ----
    pic_dir = cfg_train.pic_path
    pic_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(history["train_d"], label="D loss")
    axes[0].plot(history["train_g"], label="G loss")
    axes[0].plot(history["val_loss"], label="val L1")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("GA-Predrnn training curves")

    axes[1].plot(history["val_rmse"], label="val RMSE", color="tab:red")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Validation RMSE")
    plt.tight_layout()
    plt.savefig(pic_dir / "ga_predrnn_train_loss.png", dpi=150)
    plt.close()
    logger.info(f"plots saved to {pic_dir / 'ga_predrnn_train_loss.png'}")

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Best val loss: {best_val:.5f}")
    logger.info(f"Total time: {h}h {m}m {s}s")
    logger.info(SEP)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
