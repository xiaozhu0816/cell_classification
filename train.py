from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from datasets import build_datasets
from models import build_model
from utils import AverageMeter, binary_metrics, build_transforms, get_logger, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train infected vs uninfected classifier")
    parser.add_argument("--config", type=str, default="configs/resnet50_baseline.yaml", help="Path to YAML config")
    return parser.parse_args()


def create_dataloaders(train_ds, val_ds, test_ds, data_cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=data_cfg.get("drop_last", True), **loader_kwargs)
    eval_loader = lambda ds: DataLoader(ds, shuffle=False, **loader_kwargs)  # noqa: E731
    return train_loader, eval_loader(val_ds), eval_loader(test_ds)


def log_dataset_overview(logger, dataset, split_name: str, max_examples: int = 3) -> None:
    samples = getattr(dataset, "samples", dataset)
    counter = Counter(sample.condition for sample in samples)
    label_counter = Counter(sample.label for sample in samples)
    breakdown = ", ".join(f"{cond}:{count}" for cond, count in sorted(counter.items()))
    label_breakdown = ", ".join(f"label{label}:{count}" for label, count in sorted(label_counter.items()))
    logger.info(
        f"{split_name} split -> total:{len(dataset)} | conditions[{breakdown}] | labels[{label_breakdown}]"
    )
    sample_paths = [f"{sample.path}#t{sample.frame_index}" for sample in samples[:max_examples]]
    if sample_paths:
        logger.info(f"{split_name} sample frames: {sample_paths}")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    logger,
    use_amp: bool,
    grad_clip: float | None = None,
) -> float:
    model.train()
    meter = AverageMeter("train_loss")
    amp_device = device.type
    autocast_enabled = use_amp and amp_device == "cuda"
    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(amp_device, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), n=images.size(0))
    logger.info(f"train loss: {meter.avg:.4f}")
    return meter.avg


def evaluate(model, loader, criterion, device, logger, split_name: str) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter(f"{split_name}_loss")
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=split_name, leave=False):
            images = images.to(device, non_blocking=True)
            labels_tensor = labels.float().unsqueeze(1).to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_tensor)
            losses.update(loss.item(), n=images.size(0))
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels_tensor.cpu().numpy())
    all_logits = np.concatenate(logits_list, axis=0).squeeze(-1)
    all_labels = np.concatenate(labels_list, axis=0).squeeze(-1)
    metrics_dict = binary_metrics(all_logits, all_labels)
    metrics_dict["loss"] = losses.avg
    summary = " | ".join(f"{k}:{v:.4f}" for k, v in metrics_dict.items())
    logger.info(f"{split_name}: {summary}")
    return metrics_dict


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = device.type
    experiment_name = cfg.get("experiment_name", "cell_classification")
    log_dir = Path(cfg.get("logging", {}).get("output_dir", "outputs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(name=experiment_name, log_dir=log_dir)

    data_cfg = cfg.get("data", {})
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)
    train_ds, val_ds, test_ds = build_datasets(data_cfg, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, data_cfg)
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        log_dataset_overview(logger, dataset, split_name)

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg).to(device)

    optimizer_cfg = cfg.get("optimizer", {"lr": 1e-4})
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)

    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", 10),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )

    training_cfg = cfg.get("training", {})
    criterion = nn.BCEWithLogitsLoss()
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = amp.GradScaler(amp_device, enabled=use_amp and amp_device == "cuda")
    epochs = training_cfg.get("epochs", 10)
    best_auc = 0.0
    checkpoint_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            logger,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, logger, split_name="val")
        if scheduler:
            scheduler.step()
        auc = val_metrics.get("auc", 0.0)
        is_best = not math.isnan(auc) and auc > best_auc
        if is_best:
            best_auc = auc
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, checkpoint_dir / "best.pt")
            logger.info(f"New best model with AUC {best_auc:.4f}")

    logger.info("Evaluating on test split")
    evaluate(model, test_loader, criterion, device, logger, split_name="test")


if __name__ == "__main__":
    main()
