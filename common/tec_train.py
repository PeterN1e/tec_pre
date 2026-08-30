import torch
from config import TrainConfig, DatasetConfig
cfg_train = TrainConfig
cfg_dataset = DatasetConfig

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
import time
import subprocess
from tqdm import tqdm
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
                 ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.criterion_name = criterion_name
        self.optimizer = optimizer
        self.batch_size = cfg_train.batch_size
        self.model_name = cfg_train.model_name
        self.epochs_num = cfg_train.epochs_num
        self.patience = patience
        self.input_length = cfg_train.input_length
        self.output_length = cfg_train.output_length
        self.start_month_train = cfg_dataset.start_month_train
        self.end_month_train = cfg_dataset.end_month_train
        self.start_month_val = cfg_dataset.start_month_val
        self.end_month_val = cfg_dataset.end_month_val
        self.device = cfg_train.device
        self.scheduler = scheduler
        self.save_best = save_best
        self.model_save_path = model_save_path
        self.best_test_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.use_amp = getattr(cfg_train, "use_amp", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ---- per-model logger ----
        log_file = cfg_train.log_path / f"{self.model_name}.log"
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
        self.logger.info(f"Hyperparameters: batch_size={self.batch_size}, epochs_num={num_epochs}, patience={self.patience}, lr={cfg_train.lr}")
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

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(batch_in_tec, batch_in_aux)
                    loss = self.criterion(output, batch_exp_tec)
                self.optimizer.zero_grad()
                if self.use_amp:
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
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
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
                    torch.save(self.model.state_dict(), self.model_save_path)
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
