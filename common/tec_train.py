import torch
from config import TrainConfig, DatasetConfig
cfg_train = TrainConfig()
cfg_dataset = DatasetConfig()

import os
from pathlib import Path
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import time
import subprocess
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from core.checkpoint import save_checkpoint
from common.EvaluationMetrics import StreamingMetrics, evaluate_all
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 输入空间尺寸固定时，让 cuDNN 自动挑选最快卷积算法
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

SEP = "=" * 80


def _get_gpu_info():
    """Return (mem_used_mb, mem_total_mb, gpu_util_pct) or (0, 0, "N/A") on failure."""
    if not torch.cuda.is_available():
        return 0, 0, "N/A"
    try:
        mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
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


class TrainModel:
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 criterion,
                 criterion_name,
                 optimizer,
                 model_save_path,
                 scheduler=None,
                 save_best=True,
                 patience=5,
                 model_name=None,
                 learning_rate=None,
                 batch_size=None,
                 input_length=None,
                 output_length=None,
                 device=None,
                 log_path=None,
                 config=None,
                 use_amp=None,
                 ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.criterion_name = criterion_name
        self.optimizer = optimizer
        self.batch_size = batch_size or cfg_train.batch_size
        self.model_name = model_name or cfg_train.model_name
        self.epochs_num = cfg_train.epochs_num
        self.patience = patience
        self.input_length = input_length or cfg_train.input_length
        self.output_length = output_length or cfg_train.output_length
        self.learning_rate = (
            learning_rate if learning_rate is not None else cfg_train.lr
        )
        self.start_month_train = cfg_dataset.start_month_train
        self.end_month_train = cfg_dataset.end_month_train
        self.start_month_val = cfg_dataset.start_month_val
        self.end_month_val = cfg_dataset.end_month_val
        self.device = device or cfg_train.device
        self.scheduler = scheduler
        self.save_best = save_best
        self.model_save_path = model_save_path
        self.config = config
        self.best_test_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.use_amp = (
            getattr(cfg_train, "use_amp", False)
            if use_amp is None
            else bool(use_amp)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ---- per-model logger ----
        log_file = Path(log_path or cfg_train.log_path) / f"{self.model_name}.log"
        self.logger = logging.getLogger(f"train.{self.model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def train(self, num_epochs):
        train_losses = []
        test_losses = []

        param_count = sum(p.numel() for p in self.model.parameters())

        self.logger.info(SEP)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Hyperparameters: batch_size={self.batch_size}, epochs_num={num_epochs}, patience={self.patience}, lr={self.learning_rate}")
        self.logger.info(f"Dataset: Train: {self.start_month_train}-{self.end_month_train}, Val: {self.start_month_val}-{self.end_month_val}")
        self.logger.info(f"Parameters: {param_count:,}")
        self.logger.info(f"Loss Function: {self.criterion_name}")
        self.logger.info(SEP)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            train_loss = 0.0
            pbar = tqdm(self.train_loader,
                        total=len(self.train_loader),
                        ncols=100,
                        desc=f"Epoch {epoch}/{num_epochs}",
                        leave=False)
            for batch_in_tec, batch_in_aux, batch_exp_tec, batch_exp_aux in pbar:
                batch_in_tec = batch_in_tec.float().to(self.device)
                batch_in_aux = batch_in_aux.float().to(self.device)
                batch_exp_tec = batch_exp_tec.float().to(self.device)
                batch_exp_aux = batch_exp_aux.float().to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    if getattr(self.criterion, "_needs_context", False):
                        loss = self.criterion(batch_in_tec, batch_in_aux, batch_exp_tec)
                    else:
                        output = self.model(batch_in_tec, batch_in_aux)
                        loss = self.criterion(output, batch_exp_tec)
                self.optimizer.zero_grad()
                if self.use_amp:   #使用自动混合精度训练
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                train_loss += loss.item()
                avg_loss = train_loss / (pbar.n + 1)
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}",
                                  "avg": f"{avg_loss:.4f}"})

            # ---- validation ----
            self.model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_in_tec, batch_in_aux, batch_exp_tec, batch_exp_aux in self.test_loader:
                    batch_in_tec = batch_in_tec.float().to(self.device)
                    batch_in_aux = batch_in_aux.float().to(self.device)
                    batch_exp_tec = batch_exp_tec.float().to(self.device)
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        if getattr(self.criterion, "_needs_context", False):
                            test_loss += self.criterion(batch_in_tec, batch_in_aux, batch_exp_tec).item()
                        else:
                            outputs = self.model(batch_in_tec, batch_in_aux)
                            test_loss += self.criterion(outputs, batch_exp_tec).item()

            avg_train_loss = train_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            # ---- scheduler ----
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_test_loss)
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            gpu_mem_used, gpu_mem_total, gpu_util = _get_gpu_info()

            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Train {avg_train_loss:.5f} | "
                f"Val   {avg_test_loss:.5f} | "
                f"LR {current_lr:.2e} | "
                f"GPU Mem {gpu_mem_used}/{gpu_mem_total} MB | "
                f"GPU Util {gpu_util}"
            )
            pbar.close()

            if avg_test_loss < self.best_test_loss:
                self.best_test_loss = avg_test_loss
                self.counter = 0
                if self.save_best:
                    save_checkpoint(
                        self.model_save_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        best_metric=avg_test_loss,
                        config=self.config,
                    )
                    self.logger.info(f"Best model saved at epoch {epoch} with val loss {avg_test_loss:.5f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.early_stop = True
                    break

        elapsed = time.time() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        self.logger.info(f"Best val loss: {self.best_test_loss:.5f}")
        self.logger.info(f"Total time: {h}h {m}m {s}s")
        self.logger.info(SEP)

        return train_losses, test_losses


# ================================================================== #
#  Reusable GAN training framework
# ================================================================== #

def _default_d_loss(score_real, score_fake):
    """Hinge discriminator loss."""
    return F.relu(1.0 - score_real).mean() + F.relu(1.0 + score_fake).mean()


def _default_g_adv_loss(score_fake):
    """Hinge generator adversarial loss."""
    return -score_fake.mean()


def _resolve_train_forward(generator):
    fn = getattr(generator, "train_forward", None)
    if fn is None and hasattr(generator, "model"):
        fn = getattr(generator.model, "train_forward", None)
    return fn


def _generator_outputs(generator, tec, aux):
    """Return (pred_tec, pred_aux); pred_aux is None when unavailable."""
    fn = _resolve_train_forward(generator)
    if fn is not None:
        out = fn(tec, aux)
        if isinstance(out, tuple):
            pred_tec = out[0]
            pred_aux = out[1] if len(out) > 1 else None
            return pred_tec, pred_aux
        return out, None
    return generator(tec, aux), None


@torch.no_grad()
def _gan_validate(generator, loader, device):
    """流式验证：只保留充分统计量，不把整个验证集的预测堆进内存。

    旧实现把 pred/target 全部 append 再 concatenate，验证集两个数组就有数 GB，
    再交给 ``evaluate_all`` 做整表 SSIM 卷积。改为逐 batch 累加指标，峰值内存
    与验证集大小无关。
    """
    generator.eval()
    accumulator = StreamingMetrics()
    val_loss = 0.0
    for tec_in, aux_in, tec_gt, _ in loader:
        tec_in = tec_in.float().to(device)
        aux_in = aux_in.float().to(device)
        tec_gt = tec_gt.float().to(device)
        pred_tec = generator(tec_in, aux_in)
        val_loss += F.l1_loss(pred_tec, tec_gt).item()
        accumulator.update(
            pred_tec.detach().cpu().numpy(),
            tec_gt.detach().cpu().numpy(),
        )

    metrics = accumulator.aggregate()
    # 与既有日志字段保持一致：训练循环读取 RMSE / R2 / SSIM。
    metrics.setdefault("SSIM", float("nan"))
    metrics.setdefault("R2", float("nan"))
    metrics.setdefault("RMSE", float("nan"))
    avg_loss = val_loss / max(len(loader), 1)
    return avg_loss, metrics


def train_gan(
    generator,
    discriminator,
    train_loader,
    val_loader,
    g_optimizer,
    d_optimizer,
    model_name,
    model_save_path,
    device,
    epochs,
    patience,
    lambda_tec=1.0,
    lambda_aux=0.1,
    adv_weight=1.0,
    clip_grad=1.0,
    use_amp=False,
    amp_dtype="bf16",
    scheduler_g=None,
    scheduler_d=None,
    g_steps=1,
    d_steps=1,
    d_loss_fn=None,
    g_adv_loss_fn=None,
    log_path=None,
    config=None,
    aux_indices=(2, 3, 4),
):
    """Reusable GAN trainer: alternating D/G steps with generator-only validation.

    Model contract:
        generator.forward(tec, aux) -> pred_tec
        generator.train_forward(tec, aux) -> (pred_tec, pred_aux)  [optional]
        discriminator(frame) -> score, frame shape (B*T, 1, H, W)
    """
    d_loss_fn = d_loss_fn or _default_d_loss
    g_adv_loss_fn = g_adv_loss_fn or _default_g_adv_loss

    log_dir = Path(log_path or cfg_train.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"gan.{model_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, f"{model_name}.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    disc_save_path = os.path.join(os.path.dirname(model_save_path), "discriminator_state_dict.pth")
    best_val_loss = float("inf")
    counter = 0
    history = {"train_d": [], "train_g": [], "val_loss": [], "val_rmse": [], "val_r2": [], "val_ssim": []}

    total_params = sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters())
    logger.info(SEP)
    logger.info(f"Model: {model_name} (GAN)")
    logger.info(f"Hyperparameters: epochs={epochs}, patience={patience}, g_steps={g_steps}, d_steps={d_steps}")
    logger.info(f"Loss weights: lambda_tec={lambda_tec}, lambda_aux={lambda_aux}, adv_weight={adv_weight}")
    logger.info(f"Parameters: generator={sum(p.numel() for p in generator.parameters()):,}, "
                f"discriminator={sum(p.numel() for p in discriminator.parameters()):,}, total={total_params:,}")
    logger.info(SEP)

    amp_dtype = (
        torch.bfloat16
        if str(amp_dtype).lower() in ("bf16", "bfloat16")
        else torch.float16
    )
    autocast_enabled = bool(use_amp) and device.type == "cuda"
    # bf16 keeps enough exponent range to skip loss scaling; fp16 still needs it.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=autocast_enabled and amp_dtype == torch.float16,
    )
    logger.info(
        f"AMP: enabled={autocast_enabled} "
        f"dtype={amp_dtype if autocast_enabled else 'fp32'} "
        f"loss_scaling={scaler.is_enabled()}"
    )
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        running_d = running_g = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), ncols=100, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch_in_tec, batch_in_aux, batch_exp_tec, batch_exp_aux in pbar:
            batch_in_tec = batch_in_tec.float().to(device)
            batch_in_aux = batch_in_aux.float().to(device)
            batch_exp_tec = batch_exp_tec.float().to(device)
            batch_exp_aux = batch_exp_aux.float().to(device)
            B, T_out, H, W = batch_exp_tec.shape
            real = batch_exp_tec.reshape(B * T_out, 1, H, W)

            # One generator forward per batch. The discriminator consumes a
            # detached view of these predictions, so the project no longer
            # needs the extra no_grad generator pass it used to run here. The
            # graph stays alive for the generator step below.
            with torch.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                pred_tec, pred_aux = _generator_outputs(
                    generator, batch_in_tec, batch_in_aux
                )
            fake_for_d = pred_tec.detach().reshape(B * T_out, 1, H, W)
            fake_for_g = pred_tec.reshape(B * T_out, 1, H, W)

            # ---- Discriminator steps ----
            for _ in range(d_steps):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                    score_real = discriminator(real)
                    score_fake = discriminator(fake_for_d)
                # losses are evaluated in fp32 regardless of the autocast dtype
                loss_d = d_loss_fn(score_real.float(), score_fake.float())
                d_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(d_optimizer)
                scaler.update()

            # ---- Generator steps ----
            for step_index in range(g_steps):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=autocast_enabled):
                    score_fake = discriminator(fake_for_g)
                loss_tec = F.l1_loss(pred_tec.float(), batch_exp_tec)
                loss_adv = g_adv_loss_fn(score_fake.float())
                loss_g = lambda_tec * loss_tec + adv_weight * loss_adv
                if pred_aux is not None:
                    loss_aux = F.l1_loss(
                        pred_aux.float(),
                        batch_exp_aux[:, :, list(aux_indices)],
                    )
                    loss_g = loss_g + lambda_aux * loss_aux
                g_optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss_g).backward(
                    retain_graph=step_index < g_steps - 1
                )
                scaler.unscale_(g_optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad)
                scaler.step(g_optimizer)
                scaler.update()

            running_d += loss_d.item()
            running_g += loss_g.item()
            pbar.set_postfix(loss_D=f"{loss_d.item():.4f}", loss_G=f"{loss_g.item():.4f}")

        avg_d = running_d / max(len(train_loader), 1)
        avg_g = running_g / max(len(train_loader), 1)
        val_loss, val_metrics = _gan_validate(generator, val_loader, device)

        for sched in (scheduler_g, scheduler_d):
            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_loss)
                else:
                    sched.step()

        history["train_d"].append(avg_d)
        history["train_g"].append(avg_g)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_metrics["RMSE"])
        history["val_r2"].append(val_metrics["R2"])
        history["val_ssim"].append(val_metrics["SSIM"])

        gpu_mem_used, gpu_mem_total, gpu_util = _get_gpu_info()
        logger.info(
            f"Epoch {epoch:3d} | D {avg_d:.5f} | G {avg_g:.5f} | "
            f"Val L1 {val_loss:.5f} | RMSE {val_metrics['RMSE']:.4f} "
            f"R2 {val_metrics['R2']:.4f} SSIM {val_metrics['SSIM']:.4f} | "
            f"GPU Mem {gpu_mem_used}/{gpu_mem_total} MB | GPU Util {gpu_util}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_checkpoint(
                model_save_path,
                model=generator,
                optimizer=g_optimizer,
                scheduler=scheduler_g,
                scaler=scaler,
                epoch=epoch,
                best_metric=val_loss,
                config=config,
                extra={"discriminator_state_dict": discriminator.state_dict()},
            )
            torch.save(discriminator.state_dict(), disc_save_path)
            logger.info(f"Best GAN model saved at epoch {epoch} with val loss {val_loss:.5f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Best val loss: {best_val_loss:.5f}")
    logger.info(f"Total time: {h}h {m}m {s}s")
    logger.info(SEP)
    return history
