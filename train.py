from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from datasets import build_datasets
from models import build_model
from utils import AverageMeter, binary_metrics, build_transforms, get_logger, load_config, set_seed
from rich import print

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
    tiff_sets = defaultdict(set)
    frame_counts = defaultdict(int)
    for sample in samples:
        path_str = str(sample.path)
        tiff_sets[sample.condition].add(path_str)
        frame_counts[path_str] += 1
    tiff_breakdown = ", ".join(f"{cond}:{len(paths)}" for cond, paths in sorted(tiff_sets.items()))
    avg_frames = []
    for cond, paths in sorted(tiff_sets.items()):
        if paths:
            avg = sum(frame_counts[p] for p in paths) / len(paths)
            avg_frames.append(f"{cond}:{avg:.1f}")
    total_tiffs = sum(len(paths) for paths in tiff_sets.values())
    logger.info(
        f"{split_name} TIFFs -> total:{total_tiffs} | per-condition[{tiff_breakdown}] | avg-frames/TIFF[{', '.join(avg_frames)}]"
    )


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
    progress_desc: str = "train",
) -> float:
    model.train()
    meter = AverageMeter("train_loss")
    amp_device = device.type
    autocast_enabled = use_amp and amp_device == "cuda"
    for images, labels, _ in tqdm(loader, desc=progress_desc, leave=False):
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


def evaluate(
    model,
    loader,
    criterion,
    device,
    logger,
    split_name: str,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter(f"{split_name}_loss")
    logits_list, labels_list = [], []
    desc = progress_desc or split_name
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels_tensor = labels.float().unsqueeze(1).to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels_tensor)
            losses.update(loss.item(), n=images.size(0))
            logits_list.append(logits.detach().cpu().numpy())
            labels_list.append(labels_tensor.cpu().numpy())
    if not logits_list:
        logger.warning(f"{split_name}: no samples available for evaluation")
        return {"loss": losses.avg}
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
    training_cfg = cfg.get("training", {})
    k_folds = max(1, int(training_cfg.get("k_folds", 1)))
    run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")

    log_root = Path(cfg.get("logging", {}).get("output_dir", "outputs")) / experiment_name / run_id
    ckpt_root = Path(training_cfg.get("checkpoint_dir", "checkpoints")) / experiment_name / run_id
    log_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)

    log_global_dataset_info(data_cfg)

    fold_summaries: List[Dict] = []
    for fold_idx in range(k_folds):
        fold_tag = f"fold_{fold_idx + 1:02d}of{k_folds:02d}"
        fold_log_dir = log_root / fold_tag
        fold_ckpt_dir = ckpt_root / fold_tag
        fold_log_dir.mkdir(parents=True, exist_ok=True)
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger = get_logger(name=f"{experiment_name}_{fold_tag}", log_dir=fold_log_dir)
        summary = run_fold(
            cfg,
            data_cfg,
            transforms,
            training_cfg,
            fold_idx,
            k_folds,
            device,
            amp_device,
            logger,
            fold_log_dir,
            fold_ckpt_dir,
        )
        fold_summaries.append(summary)

    if k_folds > 1:
        summarize_cross_validation(fold_summaries)


def run_fold(
    cfg: Dict,
    data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    training_cfg: Dict,
    fold_idx: int,
    k_folds: int,
    device,
    amp_device: str,
    logger,
    log_dir: Path,
    checkpoint_dir: Path,
) -> Dict:
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg,
        transforms,
        fold_index=fold_idx if k_folds > 1 else None,
        num_folds=k_folds,
    )
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

    criterion = nn.BCEWithLogitsLoss()
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = amp.GradScaler(amp_device, enabled=use_amp and amp_device == "cuda")
    epochs = training_cfg.get("epochs", 10)
    best_auc = 0.0
    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        logger.info(f"Fold {fold_idx + 1}/{k_folds} - Epoch {epoch}/{epochs}")
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
            progress_desc=f"train_f{fold_idx + 1}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            logger,
            split_name="val",
            progress_desc=f"val_f{fold_idx + 1}",
        )
        if scheduler:
            scheduler.step()
        auc = val_metrics.get("auc", 0.0)
        is_best = not math.isnan(auc) and auc > best_auc
        if is_best:
            best_auc = auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "fold": fold_idx,
                },
                best_checkpoint,
            )
            logger.info(f"New best model with AUC {best_auc:.4f}")

    logger.info("Evaluating on test split")
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        logger,
        split_name="test",
        progress_desc=f"test_f{fold_idx + 1}",
    )

    return {
        "fold": fold_idx,
        "val": val_metrics,
        "test": test_metrics,
        "best_auc": best_auc,
        "checkpoint": str(best_checkpoint),
        "log_dir": str(log_dir),
    }


def log_global_dataset_info(data_cfg: Dict) -> None:
    infected_dir = Path(data_cfg["infected_dir"])
    uninfected_dir = Path(data_cfg["uninfected_dir"])
    infected_files = sorted(infected_dir.glob("*.tif*"))
    uninfected_files = sorted(uninfected_dir.glob("*.tif*"))
    total_infected = len(infected_files)
    total_uninfected = len(uninfected_files)
    total_files = total_infected + total_uninfected

    frames_cfg = data_cfg.get("frames", {})
    frames_per_hour = frames_cfg.get("frames_per_hour", 2.0)
    infected_window = frames_cfg.get("infected_window_hours", (16, 30))
    infected_stride = frames_cfg.get("infected_stride", 1)
    uninfected_stride = frames_cfg.get("uninfected_stride", 1)
    uninfected_use_all = frames_cfg.get("uninfected_use_all", True)

    print("\n[bold cyan]Dataset overview[/bold cyan]")
    print(
        f"Total TIFFs: {total_files} (infected={total_infected}, uninfected={total_uninfected}) | "
        f"frames_per_hour={frames_per_hour}"
    )
    print(
        f"Infected frame window: {infected_window[0]}h–{infected_window[1]}h | stride={infected_stride}"
    )
    if uninfected_use_all:
        print(f"Uninfected frames: use all (stride={uninfected_stride})")
    else:
        print(f"Uninfected frames: first frame only (stride ignored)")


def summarize_cross_validation(fold_summaries: List[Dict]) -> None:
    print("\nCross-validation summary")
    for summary in fold_summaries:
        fold_id = summary["fold"] + 1
        val_auc = summary["val"].get("auc") if summary.get("val") else float("nan")
        test_auc = summary.get("test", {}).get("auc", float("nan"))
        print(f"Fold {fold_id}: val_auc={val_auc:.4f} | test_auc={test_auc:.4f} | ckpt={summary['checkpoint']}")

    for split in ("val", "test"):
        metric_keys = set()
        for summary in fold_summaries:
            metric_keys.update(summary.get(split, {}).keys())
        metric_keys.discard("loss")  # optional: keep accuracy-style metrics separate
        if not metric_keys:
            continue
        print(f"\n{split.upper()} metrics (mean across folds):")
        for key in sorted(metric_keys):
            values = [summary.get(split, {}).get(key) for summary in fold_summaries if key in summary.get(split, {})]
            values = [v for v in values if v is not None and not math.isnan(v)]
            if values:
                mean_value = sum(values) / len(values)
                print(f"  {key}: {mean_value:.4f}")


if __name__ == "__main__":
    main()
