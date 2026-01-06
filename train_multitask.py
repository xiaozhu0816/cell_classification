"""
Multi-Task Training Script for Cell Classification + Time Regression

Goal: Train a model that simultaneously:
  1. Classifies cells as infected or uninfected
  2. Predicts time:
     - For infected cells: time since infection onset
     - For uninfected cells: elapsed time from experiment start

This preserves temporal information for both infected and uninfected samples,
allowing the model to learn comprehensive temporal patterns.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

from datasets import build_datasets, resolve_frame_policies, format_policy_summary
from models import build_multitask_model
from utils import AverageMeter, binary_metrics, build_transforms, get_logger, load_config, set_seed
from rich import print
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


def meta_batch_to_list(meta_batch: Any) -> List[Dict[str, Any]]:
    """Convert batch metadata to list of dictionaries."""
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


def build_multitask_targets(
    labels: torch.Tensor,
    meta_list: List[Dict[str, Any]],
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build targets for multi-task learning.
    
    Args:
        labels: Binary labels (0=uninfected, 1=infected)
        meta_list: List of metadata dicts with 'hours_since_start'
        infection_onset_hour: Hour when infection occurs (e.g., 2.0)
        clamp_range: (min, max) for clamping time values
        device: Target device
        
    Returns:
        cls_targets: Class indices [batch_size] for CrossEntropyLoss
        time_targets: Regression targets [batch_size, 1]
            - For infected: time since infection onset
            - For uninfected: elapsed time from experiment start
    """
    clamp_min, clamp_max = clamp_range
    
    # Classification targets: class indices for CrossEntropyLoss
    cls_targets = labels.long().to(device, non_blocking=True)
    
    # Regression targets: different time references
    time_list: List[float] = []
    for label, meta in zip(labels.tolist(), meta_list):
        hours = float(meta.get("hours_since_start", 0.0))
        
        if int(label) == 1:  # Infected
            # Time since infection onset (how long infected)
            time_value = max(hours - infection_onset_hour, 0.0)
        else:  # Uninfected
            # Elapsed time from experiment start (preserves temporal info!)
            time_value = hours
        
        # Apply clamping
        if clamp_min is not None:
            time_value = max(time_value, clamp_min)
        if clamp_max is not None:
            time_value = min(time_value, clamp_max)
        
        time_list.append(time_value)
    
    time_targets = torch.tensor(time_list, dtype=torch.float32).unsqueeze(1)
    return cls_targets, time_targets.to(device, non_blocking=True)


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics (MAE, RMSE, etc.)."""
    diff = preds - targets
    mae = np.abs(diff).mean()
    mse = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    return {"mae": mae, "rmse": rmse, "mse": mse}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    logger,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    use_amp: bool = True,
    grad_clip: Optional[float] = None,
    progress_desc: str = "train",
) -> Dict[str, float]:
    """Train for one epoch with multi-task learning."""
    model.train()
    
    total_loss_meter = AverageMeter("total_loss")
    cls_loss_meter = AverageMeter("cls_loss")
    reg_loss_meter = AverageMeter("reg_loss")
    
    amp_device = device.type
    autocast_enabled = use_amp and amp_device == "cuda"
    
    for images, labels, meta in tqdm(loader, desc=progress_desc, leave=False):
        images = images.to(device, non_blocking=True)
        meta_list = meta_batch_to_list(meta)
        
        # Build targets
        cls_targets, time_targets = build_multitask_targets(
            labels, meta_list, infection_onset_hour, clamp_range, device
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp.autocast(amp_device, enabled=autocast_enabled):
            # Forward pass: model returns (cls_logits, time_pred)
            cls_logits, time_pred = model(images)
            
            # Compute separate losses
            cls_loss = cls_criterion(cls_logits, cls_targets)
            reg_loss = reg_criterion(time_pred, time_targets)
            
            # Combined loss
            total_loss = cls_weight * cls_loss + reg_weight * reg_loss
        
        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update meters
        batch_size = images.size(0)
        total_loss_meter.update(total_loss.item(), n=batch_size)
        cls_loss_meter.update(cls_loss.item(), n=batch_size)
        reg_loss_meter.update(reg_loss.item(), n=batch_size)
    
    # Log results
    logger.info(
        f"{progress_desc} - "
        f"total_loss: {total_loss_meter.avg:.4f} | "
        f"cls_loss: {cls_loss_meter.avg:.4f} | "
        f"reg_loss: {reg_loss_meter.avg:.4f}"
    )
    
    return {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    device: torch.device,
    logger,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    split_name: str = "val",
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate multi-task model."""
    model.eval()
    
    total_loss_meter = AverageMeter(f"{split_name}_total_loss")
    cls_loss_meter = AverageMeter(f"{split_name}_cls_loss")
    reg_loss_meter = AverageMeter(f"{split_name}_reg_loss")
    
    # Collect predictions and targets
    cls_logits_list, cls_targets_list = [], []
    time_preds_list, time_targets_list = [], []
    
    desc = progress_desc or split_name
    
    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            
            # Build targets
            cls_targets, time_targets = build_multitask_targets(
                labels, meta_list, infection_onset_hour, clamp_range, device
            )
            
            # Forward pass
            cls_logits, time_pred = model(images)
            
            # Compute losses
            cls_loss = cls_criterion(cls_logits, cls_targets)
            reg_loss = reg_criterion(time_pred, time_targets)
            total_loss = cls_weight * cls_loss + reg_weight * reg_loss
            
            # Update meters
            batch_size = images.size(0)
            total_loss_meter.update(total_loss.item(), n=batch_size)
            cls_loss_meter.update(cls_loss.item(), n=batch_size)
            reg_loss_meter.update(reg_loss.item(), n=batch_size)
            
            # Collect for metrics
            cls_logits_list.append(cls_logits.detach().cpu().numpy())
            cls_targets_list.append(cls_targets.cpu().numpy())
            time_preds_list.append(time_pred.detach().cpu().numpy())
            time_targets_list.append(time_targets.cpu().numpy())
    
    if not cls_logits_list:
        logger.warning(f"{split_name}: no samples available")
        return {"total_loss": total_loss_meter.avg}
    
    # Compute classification metrics
    all_cls_logits = np.concatenate(cls_logits_list, axis=0)  # [N, 2]
    all_cls_targets = np.concatenate(cls_targets_list, axis=0)  # [N]
    
    # Convert 2-class logits to probabilities for positive class
    # Apply softmax and take the probability of class 1
    from scipy.special import softmax
    all_cls_probs = softmax(all_cls_logits, axis=1)[:, 1]  # [N] - prob of class 1
    
    # Convert probabilities to "logits" format for binary_metrics function
    # binary_metrics expects logits and applies sigmoid, so we need to reverse that
    # probs = sigmoid(logits) => logits = log(probs / (1 - probs))
    epsilon = 1e-7
    all_cls_probs = np.clip(all_cls_probs, epsilon, 1 - epsilon)
    all_cls_logits_binary = np.log(all_cls_probs / (1 - all_cls_probs))
    
    cls_metrics = binary_metrics(all_cls_logits_binary, all_cls_targets)
    
    # Compute regression metrics
    all_time_preds = np.concatenate(time_preds_list, axis=0).squeeze(-1)
    all_time_targets = np.concatenate(time_targets_list, axis=0).squeeze(-1)
    reg_metrics = compute_regression_metrics(all_time_preds, all_time_targets)
    
    # Combine metrics
    metrics = {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
    }
    
    # Add classification metrics with prefix
    for k, v in cls_metrics.items():
        metrics[f"cls_{k}"] = v
    
    # Add regression metrics with prefix
    for k, v in reg_metrics.items():
        metrics[f"reg_{k}"] = v
    
    # Compute combined metric for model selection
    # Balances classification F1 and regression MAE
    cls_f1 = metrics.get("cls_f1", 0.0)
    reg_mae = metrics.get("reg_mae", clamp_range[1])
    max_time = clamp_range[1]
    
    # Normalize regression: 0 MAE → 1.0, max_time MAE → 0.0
    reg_score = max(0.0, 1.0 - (reg_mae / max_time))
    
    # Weighted combination: 60% classification, 40% regression
    combined_metric = 0.6 * cls_f1 + 0.4 * reg_score
    metrics["combined"] = combined_metric
    
    # Log summary
    summary = " | ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
    logger.info(f"{split_name}: {summary}")
    
    # Return metrics along with predictions for visualization
    return metrics, {
        "time_preds": all_time_preds,
        "time_targets": all_time_targets,
        "cls_preds": all_cls_probs,
        "cls_targets": all_cls_targets,
    }


def evaluate_temporal_generalization(
    predictions: Dict[str, np.ndarray],
    metadata_list: List[Dict[str, Any]],
    window_size: float = 6.0,
    stride: float = 3.0,
    start_hour: float = 0.0,
    end_hour: float = 48.0,
    time_field: str = "hours_since_start",
    logger = None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """
    Compute classification metrics across sliding time windows.
    
    Args:
        predictions: Dict with 'cls_preds', 'cls_targets' from evaluate()
        metadata_list: List of metadata dicts with time information
        window_size: Size of each window in hours
        stride: Step between windows in hours
        start_hour: Start of first window
        end_hour: End of last window
        time_field: Field name for time in metadata
        logger: Logger for info messages
        
    Returns:
        window_centers: List of window center times
        metrics_by_window: Dict mapping metric_name -> list of values per window
    """
    cls_probs = predictions["cls_preds"]
    cls_targets = predictions["cls_targets"]
    
    # Extract times from metadata
    times = np.array([float(m.get(time_field, m.get("hours_since_start", 0.0))) 
                      for m in metadata_list])
    
    # Generate windows
    windows = []
    current = start_hour
    while current + window_size <= end_hour:
        start = current
        end = current + window_size
        center = current + window_size / 2.0
        windows.append((start, end, center))
        current += stride
    
    if logger:
        logger.info(f"\nTemporal Generalization Analysis:")
        logger.info(f"  Window size: {window_size}h, Stride: {stride}h")
        logger.info(f"  Number of windows: {len(windows)}")
        logger.info(f"  Time range: [{times.min():.1f}, {times.max():.1f}]h")
    
    # Compute metrics for each window
    window_centers = []
    metrics_by_window = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    
    for start, end, center in windows:
        # Find samples in this window
        mask = (times >= start) & (times <= end)
        n_samples = mask.sum()
        
        if n_samples == 0:
            if logger:
                logger.info(f"  Window [{start:.1f}, {end:.1f}]h: No samples, skipping")
            continue
        
        window_probs = cls_probs[mask]
        window_labels = cls_targets[mask]
        window_pred_binary = (window_probs >= 0.5).astype(int)
        
        # Check if we have both classes
        unique_labels = np.unique(window_labels)
        n_infected = np.sum(window_labels == 1)
        n_uninfected = np.sum(window_labels == 0)
        
        # Compute metrics
        try:
            # AUC (need both classes)
            if len(unique_labels) > 1:
                auc = roc_auc_score(window_labels, window_probs)
            else:
                auc = None
            
            # Other metrics
            acc = accuracy_score(window_labels, window_pred_binary)
            prec, rec, f1, _ = precision_recall_fscore_support(
                window_labels, window_pred_binary, average="binary", zero_division=0
            )
            
            window_centers.append(center)
            metrics_by_window["auc"].append(auc)
            metrics_by_window["accuracy"].append(acc)
            metrics_by_window["f1"].append(f1)
            metrics_by_window["precision"].append(prec)
            metrics_by_window["recall"].append(rec)
            
            # Log
            if logger:
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                logger.info(
                    f"  Window [{start:.1f}, {end:.1f}]h: n={n_samples} "
                    f"(inf={n_infected}, uninf={n_uninfected}), "
                    f"AUC={auc_str}, Acc={acc:.4f}, F1={f1:.4f}"
                )
        
        except Exception as e:
            if logger:
                logger.warning(f"  Window [{start:.1f}, {end:.1f}]h: Error computing metrics: {e}")
    
    return window_centers, metrics_by_window


def plot_temporal_generalization(
    window_centers: List[float],
    metrics_by_window: Dict[str, List[float]],
    window_size: float,
    output_path: Path,
    model_name: str = "Multitask Model",
) -> None:
    """Create temporal generalization plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    markers = ['o', 's', '^', 'D', 'v']
    
    metric_labels = {
        "auc": "AUC",
        "accuracy": "Accuracy",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    
    for idx, (metric, values) in enumerate(metrics_by_window.items()):
        # Filter out None values
        valid_points = [(c, v) for c, v in zip(window_centers, values) if v is not None]
        
        if not valid_points:
            continue
        
        centers, vals = zip(*valid_points)
        
        ax.plot(
            centers,
            vals,
            f"-{markers[idx]}",
            linewidth=2,
            markersize=8,
            color=colors[idx],
            label=metric_labels.get(metric, metric),
            alpha=0.85,
        )
    
    ax.set_xlabel("Window Center (hours)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=13, fontweight='bold')
    ax.set_title(
        f"{model_name}: Temporal Generalization (Classification Performance)\n"
        f"Sliding Window Size: {window_size:.1f}h",
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.set_ylim([0, 1.05])
    
    # Secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-task training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    
    # Set random seed
    set_seed(cfg.get("seed", 42))
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = device.type
    
    # Experiment setup
    experiment_name = cfg.get("experiment_name", "multitask_cell_classification")
    run_id = cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = Path(cfg.get("output_dir", "outputs")) / experiment_name / run_id
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_base / "train.log"
    logger = get_logger("multitask_train", log_file)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2)}")
    
    # Multi-task configuration
    mt_cfg = cfg.get("multitask", {})
    infection_onset_hour = float(mt_cfg.get("infection_onset_hour", 2.0))
    clamp_range = tuple(mt_cfg.get("clamp_range", [0.0, 48.0]))
    cls_weight = float(mt_cfg.get("classification_weight", 1.0))
    reg_weight = float(mt_cfg.get("regression_weight", 1.0))
    
    logger.info(f"Multi-task config:")
    logger.info(f"  - Infection onset: {infection_onset_hour}h")
    logger.info(f"  - Clamp range: {clamp_range}")
    logger.info(f"  - Loss weights: cls={cls_weight}, reg={reg_weight}")
    
    # Build datasets
    data_cfg = cfg.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    logger.info("Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=None,  # Not using CV for now
        num_folds=1,
    )
    
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")
    
    # Build dataloaders
    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 32)
    num_workers = training_cfg.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Build model
    model_cfg = cfg.get("model", {})
    model = build_multitask_model(model_cfg).to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build optimizer
    optimizer_cfg = cfg.get("optimizer", {"lr": 1e-4})
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
    logger.info(f"Optimizer: AdamW with lr={optimizer_cfg.get('lr', 1e-4)}")
    
    # Build scheduler
    scheduler_cfg = cfg.get("scheduler")
    scheduler = None
    if scheduler_cfg:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", 10),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
        logger.info(f"Scheduler: CosineAnnealingLR")
    
    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    
    # Training setup
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = amp.GradScaler(amp_device, enabled=use_amp and amp_device == "cuda")
    epochs = training_cfg.get("epochs", 10)
    
    # Checkpointing
    checkpoint_dir = output_base / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    
    # Primary metric for model selection
    # Use combined metric to balance classification and regression performance
    # This prevents saturation issues when AUC reaches 1.0
    primary_metric = "combined"
    best_score = -math.inf
    
    logger.info(f"Training for {epochs} epochs...")
    logger.info(f"Primary metric: {primary_metric} (0.6*F1 + 0.4*(1-MAE/{clamp_range[1]}))")
    logger.info(f"  - Balances classification F1 and regression MAE")
    logger.info(f"  - Prevents AUC saturation at 1.0")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"=" * 80)
        logger.info(f"Epoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            logger=logger,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            use_amp=use_amp,
            grad_clip=grad_clip,
            progress_desc=f"train_e{epoch}",
        )
        
        # Validate
        val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            device=device,
            logger=logger,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            split_name="val",
            progress_desc=f"val_e{epoch}",
        )
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        # Save best model
        metric_value = val_metrics.get(primary_metric)
        if metric_value is not None and not math.isnan(metric_value):
            if metric_value > best_score:
                best_score = metric_value
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "config": cfg,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_checkpoint,
                )
                logger.info(f"✓ New best model! {primary_metric}={best_score:.4f}")
    
    # Final evaluation on test set
    logger.info("=" * 80)
    logger.info("Final evaluation on test set")
    test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        cls_criterion=cls_criterion,
        reg_criterion=reg_criterion,
        device=device,
        logger=logger,
        infection_onset_hour=infection_onset_hour,
        clamp_range=clamp_range,
        cls_weight=cls_weight,
        reg_weight=reg_weight,
        split_name="test",
        progress_desc="test_final",
    )
    
    # Save predictions for visualization
    predictions_file = output_base / "test_predictions.npz"
    np.savez(
        predictions_file,
        time_preds=test_predictions["time_preds"],
        time_targets=test_predictions["time_targets"],
        cls_preds=test_predictions["cls_preds"],
        cls_targets=test_predictions["cls_targets"],
    )
    logger.info(f"Test predictions saved to {predictions_file}")
    
    # Temporal generalization analysis
    logger.info("\n" + "=" * 80)
    logger.info("Temporal Generalization Analysis")
    logger.info("=" * 80)
    
    try:
        # Collect metadata from test dataset
        test_metadata = []
        for i in range(len(test_ds)):
            meta = test_ds.get_metadata(i)
            test_metadata.append(meta)
        
        # Run temporal analysis with sliding windows
        window_size = 6.0  # 6 hour windows
        stride = 3.0       # 3 hour stride
        
        window_centers, metrics_by_window = evaluate_temporal_generalization(
            predictions=test_predictions,
            metadata_list=test_metadata,
            window_size=window_size,
            stride=stride,
            start_hour=0.0,
            end_hour=48.0,
            logger=logger,
        )
        
        # Plot temporal generalization
        temporal_plot_path = output_base / "temporal_generalization.png"
        plot_temporal_generalization(
            window_centers=window_centers,
            metrics_by_window=metrics_by_window,
            window_size=window_size,
            output_path=temporal_plot_path,
            model_name=experiment_name,
        )
        logger.info(f"✓ Temporal generalization plot saved to {temporal_plot_path}")
        
        # Save temporal metrics
        temporal_metrics = {
            "window_size_hours": window_size,
            "stride_hours": stride,
            "window_centers": window_centers,
            "metrics_by_window": metrics_by_window,
        }
        temporal_metrics_file = output_base / "temporal_metrics.json"
        with open(temporal_metrics_file, "w") as f:
            json.dump(temporal_metrics, f, indent=2)
        logger.info(f"✓ Temporal metrics saved to {temporal_metrics_file}")
        
    except Exception as e:
        logger.warning(f"Failed to run temporal generalization analysis: {e}")
        logger.warning("Skipping temporal analysis, but training results are still saved.")
    
    # Save final results
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy/torch types to JSON-serializable Python types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "config": cfg,
        "best_val_metric": float(best_score) if best_score is not None else None,
        "test_metrics": convert_to_serializable(test_metrics),
    }
    
    results_file = output_base / "results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results.json: {e}")
        # Try saving minimal version
        try:
            minimal_results = {
                "experiment_name": experiment_name,
                "run_id": run_id,
                "best_val_metric": float(best_score) if best_score is not None else None,
            }
            with open(results_file, "w") as f:
                json.dump(minimal_results, f, indent=2)
            logger.warning(f"Saved minimal results.json (without full metrics)")
        except:
            logger.error(f"Could not save results.json at all!")
    logger.info("Training complete!")
    
    # Automatically generate visualizations and summary
    logger.info("=" * 80)
    logger.info("Generating analysis plots and summary...")
    try:
        import subprocess
        import sys
        
        analysis_cmd = [
            sys.executable,  # Use same Python interpreter
            "analyze_multitask_results.py",
            "--result-dir", str(output_base),
        ]
        
        result = subprocess.run(
            analysis_cmd, 
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent  # Run from script directory
        )
        
        logger.info("✓ Analysis complete! Check output directory for plots and summary.")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate analysis (exit code {e.returncode}):")
        if e.stdout:
            logger.warning(f"  stdout: {e.stdout[:200]}")
        if e.stderr:
            logger.warning(f"  stderr: {e.stderr[:200]}")
        logger.warning(f"You can manually run: python analyze_multitask_results.py --result-dir {output_base}")
        
    except Exception as e:
        logger.warning(f"Failed to generate analysis: {e}")
        logger.warning(f"You can manually run: python analyze_multitask_results.py --result-dir {output_base}")


if __name__ == "__main__":
    main()
