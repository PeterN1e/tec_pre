from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from common.loss_function import DeltaCriterion
from common.tec_train import TrainModel, train_gan
from core.config import (
    load_config,
    model_slug,
    resolve_data_paths,
    resolve_output_dir,
    save_config,
)
from core.registry import build_model
from data.tec_dataset import TecIonosphereDataset


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_range(config: Dict[str, Any], split: str) -> Tuple[int, int]:
    value = config["data"]["splits"][split]
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Split '{split}' must contain two month values")
    return int(value[0]), int(value[1])


def build_datasets(
    config: Dict[str, Any],
) -> Tuple[
    TecIonosphereDataset,
    TecIonosphereDataset,
    Optional[TecIonosphereDataset],
    StandardScaler,
    StandardScaler,
]:
    paths = resolve_data_paths(config)
    data_cfg = config["data"]
    aux_columns = data_cfg["aux_columns"]
    tec_scaler = StandardScaler()
    aux_scaler = StandardScaler()
    splits = data_cfg["splits"]

    def make_dataset(split: str, is_train: bool):
        start_month, end_month = _split_range(config, split)
        return TecIonosphereDataset(
            tec_dir=paths["tec_dir"],
            indices_dir=paths["indices_dir"],
            start_month=start_month,
            end_month=end_month,
            input_length=int(data_cfg["input_length"]),
            output_length=int(data_cfg["output_length"]),
            is_train=is_train,
            tec_scaler=tec_scaler,
            aux_scaler=aux_scaler,
            aux_columns=aux_columns,
        )

    train_dataset = make_dataset("train", is_train=True)
    val_dataset = make_dataset("val", is_train=False)
    test_dataset = (
        make_dataset("test", is_train=False) if "test" in splits else None
    )
    return train_dataset, val_dataset, test_dataset, tec_scaler, aux_scaler


def build_loaders(
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Any, Any]:
    train_ds, val_ds, test_ds, tec_scaler, aux_scaler = build_datasets(config)
    data_cfg = config["data"]
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
    loader_kwargs = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": pin_memory,
        "persistent_workers": bool(data_cfg.get("num_workers", 0)),
    }
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_ds,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader, tec_scaler, aux_scaler


def _build_criterion(model: nn.Module, config: Dict[str, Any]):
    model_name = config["model"]["name"]
    if model_name == "ModelCanon":
        lw = config.get("training", {}).get("loss_weights", {})
        return DeltaCriterion(
            model.model,
            delta_weight=float(lw.get("delta", 1.0)),
            recon_weight=float(lw.get("recon", 0.3)),
            fourier_weight=float(lw.get("fourier", 0.2)),
            ssim_weight=float(lw.get("ssim", 0.1)),
            temporal_weight=float(lw.get("temporal", 0.05)),
        ), "DeltaCriterion"
    loss_name = config.get("training", {}).get("loss", "l1").lower()
    losses = {
        "l1": nn.L1Loss,
        "mse": nn.MSELoss,
        "smooth_l1": nn.SmoothL1Loss,
    }
    if loss_name not in losses:
        raise ValueError(f"Unsupported loss: {loss_name}")
    return losses[loss_name](), losses[loss_name].__name__


def _build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
    learning_rate: Optional[float] = None,
):
    training_cfg = config["training"]
    lr = learning_rate
    if lr is None:
        lr = float(training_cfg.get("lr", 1e-3))
    return optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )


def _build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    scheduler_cfg = config.get("training", {}).get("scheduler") or {}
    if scheduler_cfg.get("name") != "reduce_on_plateau":
        return None
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
    )


def _build_discriminator(config: Dict[str, Any], device: torch.device):
    model_name = config["model"]["name"]
    if model_name != "GA_Predrnn":
        raise ValueError(f"No discriminator registered for {model_name}")
    from GA_Predrnn.discriminator import Discriminator

    gan_cfg = config.get("training", {}).get("gan", {})
    return Discriminator(base_ch=int(gan_cfg.get("disc_base_ch", 64))).to(device)


def run_training(
    config: Dict[str, Any],
    resume_from: Optional[str | Path] = None,
) -> Dict[str, Any]:
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    train_loader, val_loader, _, tec_scaler, aux_scaler = build_loaders(
        config,
        device,
    )

    model = build_model(config).to(device)
    model_name = config["model"]["name"]
    model_dir = resolve_output_dir(config) / model_slug(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = model_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, model_dir / "config.yaml")
    joblib.dump(tec_scaler, model_dir / "tec_scaler.pkl")
    joblib.dump(aux_scaler, model_dir / "aux_scaler.pkl")

    checkpoint_path = model_dir / "model_state_dict.pth"
    training_cfg = config["training"]
    use_amp = bool(training_cfg.get("use_amp", False)) and device.type == "cuda"

    if bool(training_cfg.get("gan", {}).get("enabled", False)):
        gan_cfg = training_cfg["gan"]
        discriminator = _build_discriminator(config, device)
        g_optimizer = _build_optimizer(
            model,
            config,
            learning_rate=float(gan_cfg.get("g_lr", training_cfg.get("lr", 1e-3))),
        )
        d_optimizer = _build_optimizer(
            discriminator,
            config,
            learning_rate=float(gan_cfg.get("d_lr", 1e-4)),
        )
        scheduler_g = _build_scheduler(g_optimizer, config)
        scheduler_d = _build_scheduler(d_optimizer, config)
        history = train_gan(
            generator=model,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=val_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            model_name=model_name,
            model_save_path=checkpoint_path,
            device=device,
            epochs=int(training_cfg["epochs"]),
            patience=int(training_cfg["patience"]),
            lambda_tec=float(gan_cfg.get("lambda_tec", 1.0)),
            lambda_aux=float(gan_cfg.get("lambda_aux", 0.1)),
            adv_weight=float(gan_cfg.get("adv_weight", 1.0)),
            clip_grad=float(training_cfg.get("grad_clip", 1.0)),
            use_amp=use_amp,
            amp_dtype=str(training_cfg.get("amp_dtype", "bf16")),
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
            log_path=log_dir,
            config=config,
            aux_indices=tuple(gan_cfg.get("aux_indices", (2, 3, 4))),
        )
        metrics = history
    else:
        criterion, criterion_name = _build_criterion(model, config)
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config)
        trainer = TrainModel(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            criterion=criterion,
            criterion_name=criterion_name,
            optimizer=optimizer,
            scheduler=scheduler,
            model_save_path=checkpoint_path,
            save_best=bool(training_cfg.get("save_best", True)),
            patience=int(training_cfg["patience"]),
            model_name=model_name,
            learning_rate=float(training_cfg.get("lr", 1e-3)),
            batch_size=int(config["data"]["batch_size"]),
            input_length=int(config["data"]["input_length"]),
            output_length=int(config["data"]["output_length"]),
            device=device,
            log_path=log_dir,
            config=config,
            use_amp=use_amp,
        )
        if resume_from is not None:
            from core.checkpoint import load_checkpoint, resume_optimizer_state

            checkpoint = load_checkpoint(resume_from, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            resume_optimizer_state(checkpoint, optimizer)
        train_losses, val_losses = trainer.train(int(training_cfg["epochs"]))
        metrics = {"train_loss": train_losses, "val_loss": val_losses}

    return {
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "history": metrics,
        "tec_scaler": tec_scaler,
        "aux_scaler": aux_scaler,
    }


def load_training_config(
    model_name: str,
    overrides: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return load_config(model_name=model_name, overrides=overrides)
