from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import json

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import numpy as np

from datasets import build_datasets, resolve_frame_policies, format_policy_summary
from models import build_model
from utils import AverageMeter, binary_metrics, build_transforms, get_logger, load_config, set_seed
from rich import print

def meta_batch_to_list(meta_batch: Any) -> List[Dict[str, Any]]:
    if isinstance(meta_batch, list):
        return meta_batch
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        if not keys:
            return []
        length = len(meta_batch[keys[0]])
        meta_list: List[Dict[str, Any]] = []
        for i in range(length):
            entry: Dict[str, Any] = {}
            for key in keys:
                value = meta_batch[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                entry[key] = value
            meta_list.append(entry)
        return meta_list
    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")


def build_class_balanced_sampler(dataset) -> WeightedRandomSampler:
    samples = getattr(dataset, "samples", dataset)
    label_counts = Counter(sample.label for sample in samples)
    total_classes = len(label_counts)
    if total_classes <= 1:
        weights = [1.0 for _ in samples]
    else:
        class_weights = {label: 1.0 / max(count, 1) for label, count in label_counts.items()}
        weights = [class_weights[sample.label] for sample in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train infected vs uninfected classifier")
    parser.add_argument("--config", type=str, default="configs/resnet50_baseline.yaml", help="Path to YAML config")
    return parser.parse_args()


def get_task_config(cfg: Dict) -> Dict:
    task_cfg = cfg.get("task") or {}
    task_cfg = {**task_cfg}
    task_cfg.setdefault("type", "classification")
    if task_cfg["type"] == "regression":
        reg_cfg = task_cfg.get("regression", {})
        reg_cfg = {**reg_cfg}
        reg_cfg.setdefault("target", "time_since_onset")
        reg_cfg.setdefault("infection_onset_hour", 2.0)
        reg_cfg.setdefault("uninfected_target", 0.0)
        reg_cfg.setdefault("clamp_range", [0.0, 48.0])
        task_cfg["regression"] = reg_cfg
    return task_cfg


def get_analysis_config(cfg: Dict) -> Dict:
    analysis_cfg = cfg.get("analysis") or {}
    analysis_cfg = {**analysis_cfg}
    analysis_cfg.setdefault("thresholds", [0.1, 0.2, 0.3, 0.4, 0.5])
    analysis_cfg.setdefault("time_bins", [])
    return analysis_cfg


def build_target_tensor(labels: torch.Tensor, meta_list: List[Dict[str, Any]], task_cfg: Dict, device) -> torch.Tensor:
    task_type = task_cfg.get("type", "classification")
    if task_type == "regression":
        reg_cfg = task_cfg.get("regression", {})
        onset = float(reg_cfg.get("infection_onset_hour", 2.0))
        target_mode = reg_cfg.get("target", "time_since_onset")
        uninfected_value = float(reg_cfg.get("uninfected_target", 0.0))
        clamp_min, clamp_max = reg_cfg.get("clamp_range", [0.0, 48.0])
        targets: List[float] = []
        for label, meta in zip(labels.tolist(), meta_list):
            hours = float(meta.get("hours_since_start", 0.0))
            if int(label) == 1:
                if target_mode == "time_to_onset":
                    value = max(onset - hours, 0.0)
                else:  # time_since_onset
                    value = max(hours - onset, 0.0)
            else:
                value = uninfected_value
            if clamp_min is not None:
                value = max(value, clamp_min)
            if clamp_max is not None:
                value = min(value, clamp_max)
            targets.append(value)
        target_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    else:
        target_tensor = labels.float().unsqueeze(1)
    return target_tensor.to(device, non_blocking=True)


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    diff = preds - targets
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return {"mae": mae, "rmse": rmse}


def _threshold_prf(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tp = float(np.logical_and(preds == 1, labels == 1).sum())
    fp = float(np.logical_and(preds == 1, labels == 0).sum())
    fn = float(np.logical_and(preds == 0, labels == 1).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def log_threshold_sweep(logger, split_name: str, probs: np.ndarray, labels: np.ndarray, thresholds: List[float]) -> None:
    rows = []
    for threshold in thresholds:
        metrics = _threshold_prf(probs, labels, threshold)
        rows.append(
            f"thr={metrics['threshold']:.2f} | P={metrics['precision']:.2f} | R={metrics['recall']:.2f} | F1={metrics['f1']:.2f} | TP={metrics['tp']:.0f}"
        )
    if rows:
        logger.info(f"{split_name} threshold sweep:\n  " + "\n  ".join(rows))


def log_time_bin_metrics(
    logger,
    split_name: str,
    time_bins: List[List[float]],
    logits: np.ndarray,
    labels: np.ndarray,
    hours: List[float],
) -> None:
    if not time_bins:
        return
    hours_arr = np.asarray(hours)
    labels_arr = np.asarray(labels)
    for start, end in time_bins:
        mask = (hours_arr >= start) & (hours_arr < end)
        count = int(mask.sum())
        if count == 0:
            logger.info(
                f"{split_name} time bin [{start:.1f}h, {end:.1f}h): samples=0 | metrics=NA (no data)"
            )
            continue
        metrics = binary_metrics(logits[mask], labels_arr[mask])
        metrics_str = " | ".join(f"{k}:{v:.3f}" for k, v in metrics.items())
        logger.info(
            f"{split_name} time bin [{start:.1f}h, {end:.1f}h): samples={count} | {metrics_str}"
        )


def create_dataloaders(train_ds, val_ds, test_ds, data_cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = data_cfg.get("batch_size", 4)
    num_workers = data_cfg.get("num_workers", 4)
    
    # Evaluation can use larger batch size since no gradients are computed
    eval_batch_size_multiplier = data_cfg.get("eval_batch_size_multiplier", 2)
    eval_batch_size = batch_size * eval_batch_size_multiplier
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    
    train_sampler = None
    if data_cfg.get("balance_sampler"):
        train_sampler = build_class_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=data_cfg.get("drop_last", True),
        **loader_kwargs,
    )
    eval_loader = lambda ds: DataLoader(ds, shuffle=False, **eval_loader_kwargs)  # noqa: E731
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


def log_dataset_stack_membership(
    logger,
    dataset,
    split_name: str,
    save_dir: Optional[Path] = None,
    max_preview: int = 20,
) -> None:
    samples = getattr(dataset, "samples", dataset)
    stack_paths = sorted({str(sample.path) for sample in samples})
    if not stack_paths:
        logger.info(f"{split_name} stack list: <empty>")
        return
    preview_names = [Path(path).name for path in stack_paths[:max_preview]]
    preview = ", ".join(preview_names)
    if len(stack_paths) > max_preview:
        preview += ", ..."
    logger.info(
        "%s stack membership (%d unique TIFFs): %s",
        split_name,
        len(stack_paths),
        preview,
    )
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{split_name}_stacks.txt"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(stack_paths))
        logger.info("Saved %s stack list to %s", split_name, out_path)


def log_frame_policies(logger, data_cfg: Dict) -> None:
    policies = resolve_frame_policies(data_cfg.get("frames"))
    for split_name in ("train", "val", "test"):
        policy = policies[split_name]
        logger.info("frame_policy[%s]: %s", split_name, format_policy_summary(policy))


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
    task_cfg: Optional[Dict] = None,
) -> float:
    model.train()
    meter = AverageMeter("train_loss")
    amp_device = device.type
    autocast_enabled = use_amp and amp_device == "cuda"
    task_cfg = task_cfg or {"type": "classification"}
    for images, labels, meta in tqdm(loader, desc=progress_desc, leave=False):
        images = images.to(device, non_blocking=True)
        meta_list = meta_batch_to_list(meta)
        targets = build_target_tensor(labels, meta_list, task_cfg, device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(amp_device, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
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
    task_cfg: Optional[Dict] = None,
    analysis_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    losses = AverageMeter(f"{split_name}_loss")
    logits_list, target_list = [], []
    desc = progress_desc or split_name
    task_cfg = task_cfg or {"type": "classification"}
    analysis_cfg = analysis_cfg or {}
    task_type = task_cfg.get("type", "classification")
    hours_list: List[float] = []
    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            target_tensor = build_target_tensor(labels, meta_list, task_cfg, device)
            logits = model(images)
            loss = criterion(logits, target_tensor)
            losses.update(loss.item(), n=images.size(0))
            logits_list.append(logits.detach().cpu().numpy())
            target_list.append(target_tensor.cpu().numpy())
            for entry in meta_list:
                hours_list.append(float(entry.get("hours_since_start", 0.0)))
    if not logits_list:
        logger.warning(f"{split_name}: no samples available for evaluation")
        return {"loss": losses.avg}
    all_logits = np.concatenate(logits_list, axis=0).squeeze(-1)
    all_targets = np.concatenate(target_list, axis=0).squeeze(-1)
    if task_type == "regression":
        metrics_dict = compute_regression_metrics(all_logits, all_targets)
    else:
        metrics_dict = binary_metrics(all_logits, all_targets)
        probs = 1.0 / (1.0 + np.exp(-all_logits))
        thresholds = analysis_cfg.get("thresholds", [])
        if thresholds:
            log_threshold_sweep(logger, split_name, probs, all_targets, thresholds)
        time_bins = analysis_cfg.get("time_bins", [])
        if time_bins:
            log_time_bin_metrics(logger, split_name, time_bins, all_logits, all_targets, hours_list)
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
    task_cfg = get_task_config(cfg)
    analysis_cfg = get_analysis_config(cfg)

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
            task_cfg,
            analysis_cfg,
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
        summary_path = log_root / "cv_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(fold_summaries, f, indent=2)
        print(f"Saved cross-validation summary to {summary_path}")


def run_fold(
    cfg: Dict,
    data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    training_cfg: Dict,
    task_cfg: Dict,
    analysis_cfg: Dict,
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
    log_frame_policies(logger, data_cfg)
    train_loader, val_loader, test_loader = create_dataloaders(train_ds, val_ds, test_ds, data_cfg)
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        log_dataset_overview(logger, dataset, split_name)
        log_dataset_stack_membership(logger, dataset, split_name, save_dir=log_dir)

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

    task_type = task_cfg.get("type", "classification")
    if task_type == "regression":
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = amp.GradScaler(amp_device, enabled=use_amp and amp_device == "cuda")
    epochs = training_cfg.get("epochs", 10)
    primary_metric = "auc" if task_type == "classification" else "rmse"
    greater_is_better = task_type == "classification"
    best_score = -math.inf if greater_is_better else math.inf
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
            task_cfg=task_cfg,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            logger,
            split_name="val",
            progress_desc=f"val_f{fold_idx + 1}",
            task_cfg=task_cfg,
            analysis_cfg=analysis_cfg,
        )
        if scheduler:
            scheduler.step()
        metric_value = val_metrics.get(primary_metric)
        is_valid = metric_value is not None and not math.isnan(metric_value)
        if is_valid:
            if (greater_is_better and metric_value > best_score) or (not greater_is_better and metric_value < best_score):
                best_score = metric_value
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
                logger.info(
                    f"New best model with {primary_metric} {best_score:.4f}"
                )

    logger.info("Evaluating on test split")
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        logger,
        split_name="test",
        progress_desc=f"test_f{fold_idx + 1}",
        task_cfg=task_cfg,
        analysis_cfg=analysis_cfg,
    )

    return {
        "fold": fold_idx,
        "val": val_metrics,
        "test": test_metrics,
        "best_metric": best_score,
        "primary_metric": primary_metric,
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

    policies = resolve_frame_policies(data_cfg.get("frames"))
    train_policy = policies["train"]
    frames_per_hour = train_policy.frames_per_hour
    infected_window = train_policy.infected_window_hours
    infected_stride = train_policy.infected_stride
    uninfected_stride = train_policy.uninfected_stride
    uninfected_use_all = train_policy.uninfected_use_all

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
    print("Frame extraction policies by split:")
    for split_name, policy in policies.items():
        print(f"  {split_name}: {format_policy_summary(policy)}")


def summarize_cross_validation(fold_summaries: List[Dict]) -> None:
    print("\nCross-validation summary")
    for summary in fold_summaries:
        fold_id = summary["fold"] + 1
        val_metrics = summary.get("val", {})
        test_metrics = summary.get("test", {})
        primary_metric = summary.get("primary_metric", "auc")
        val_value = val_metrics.get(primary_metric, float("nan"))
        test_value = test_metrics.get(primary_metric, float("nan"))
        print(
            f"Fold {fold_id}: val_{primary_metric}={val_value:.4f} | "
            f"test_{primary_metric}={test_value:.4f} | ckpt={summary['checkpoint']}"
        )

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
