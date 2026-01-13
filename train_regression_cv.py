"""
5-Fold Cross-Validation Training for Regression-Only Model

This script trains a SINGLE-TASK regression model (time prediction only).
It supports two modes:
  1. Infected-only: Train on infected samples, predict time since infection onset
  2. Uninfected-only: Train on uninfected samples, predict elapsed time from experiment start

This allows fair comparison with the multitask model to measure the benefit
of joint classification+regression vs. pure regression.

Usage:
    # Infected-only regression
    python train_regression_cv.py --config configs/regression_infected.yaml --num-folds 5
    
    # Uninfected-only regression
    python train_regression_cv.py --config configs/regression_uninfected.yaml --num-folds 5
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import build_datasets, format_policy_summary, resolve_frame_policies
from models import build_regression_model
from utils import AverageMeter, build_transforms, get_logger, load_config, set_seed


def meta_batch_to_list(meta_batch) -> List[Dict[str, Any]]:
    """Convert batched metadata to list of dicts."""
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


def build_regression_targets(
    labels: torch.Tensor,
    meta_list: List[Dict[str, Any]],
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    device: torch.device,
    target_class: Optional[int] = None,  # None=all, 0=uninfected, 1=infected
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build regression targets for single-task regression.
    
    Args:
        labels: Binary labels (0=uninfected, 1=infected)
        meta_list: List of metadata dicts with 'hours_since_start'
        infection_onset_hour: Hour when infection occurs
        clamp_range: (min, max) for clamping time values
        device: Target device
        target_class: If specified, only samples of this class are kept
        
    Returns:
        labels_out: Labels [N] (for filtering/reporting only)
        time_targets: Regression targets [N, 1]
    """
    clamp_min, clamp_max = clamp_range
    
    time_list: List[float] = []
    label_list: List[int] = []
    
    for label, meta in zip(labels.tolist(), meta_list):
        hours = float(meta.get("hours_since_start", 0.0))
        label_int = int(label)
        
        # Skip if we're filtering by class
        if target_class is not None and label_int != target_class:
            continue
        
        # Compute regression target based on class
        if label_int == 1:  # Infected
            # Time since infection onset
            time_value = max(hours - infection_onset_hour, 0.0)
        else:  # Uninfected
            # Elapsed time from experiment start
            time_value = hours
        
        # Apply clamping
        if clamp_min is not None:
            time_value = max(time_value, clamp_min)
        if clamp_max is not None:
            time_value = min(time_value, clamp_max)
        
        time_list.append(time_value)
        label_list.append(label_int)
    
    # Convert to tensors
    labels_out = torch.tensor(label_list, dtype=torch.long).to(device, non_blocking=True)
    time_targets = torch.tensor(time_list, dtype=torch.float32).unsqueeze(1).to(device, non_blocking=True)
    
    return labels_out, time_targets


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    diff = preds - targets
    mae = np.abs(diff).mean()
    mse = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    return {"mae": mae, "rmse": rmse, "mse": mse}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    target_class: Optional[int],
    use_amp: bool = True,
    grad_clip: Optional[float] = None,
    progress_desc: str = "train",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter("loss")
    
    for images, labels, meta in tqdm(loader, desc=progress_desc, leave=False):
        images = images.to(device, non_blocking=True)
        meta_list = meta_batch_to_list(meta)
        
        # Build targets (filters by target_class)
        _, time_targets = build_regression_targets(
            labels, meta_list, infection_onset_hour, clamp_range, device, target_class
        )
        
        # Skip batch if no valid samples
        if time_targets.size(0) == 0:
            continue
        
        # Filter images to match
        if target_class is not None:
            mask = (labels == target_class).to(device)
            images = images[mask]
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                time_pred = model(images)
                loss = criterion(time_pred, time_targets)
        else:
            time_pred = model(images)
            loss = criterion(time_pred, time_targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        loss_meter.update(loss.item(), n=images.size(0))
    
    return {"loss": loss_meter.avg}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    target_class: Optional[int],
    split_name: str = "val",
    progress_desc: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate regression model."""
    model.eval()
    loss_meter = AverageMeter(f"{split_name}_loss")
    
    time_preds_list, time_targets_list = [], []
    labels_list = []
    
    desc = progress_desc or split_name
    
    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            
            # Build targets
            labels_filtered, time_targets = build_regression_targets(
                labels, meta_list, infection_onset_hour, clamp_range, device, target_class
            )
            
            if time_targets.size(0) == 0:
                continue
            
            # Filter images
            if target_class is not None:
                mask = (labels == target_class).to(device)
                images = images[mask]
            
            # Forward pass
            time_pred = model(images)
            loss = criterion(time_pred, time_targets)
            
            loss_meter.update(loss.item(), n=images.size(0))
            
            # Collect predictions
            time_preds_list.append(time_pred.detach().cpu().numpy())
            time_targets_list.append(time_targets.cpu().numpy())
            labels_list.append(labels_filtered.cpu().numpy())
    
    if not time_preds_list:
        return {"loss": loss_meter.avg}, {}
    
    # Compute regression metrics
    all_time_preds = np.concatenate(time_preds_list, axis=0).squeeze(-1)
    all_time_targets = np.concatenate(time_targets_list, axis=0).squeeze(-1)
    all_labels = np.concatenate(labels_list, axis=0)
    
    reg_metrics = compute_regression_metrics(all_time_preds, all_time_targets)
    
    metrics = {"loss": loss_meter.avg}
    for k, v in reg_metrics.items():
        metrics[f"reg_{k}"] = v
    
    predictions = {
        "time_preds": all_time_preds,
        "time_targets": all_time_targets,
        "labels": all_labels,  # Actual class labels (for reporting)
    }
    
    return metrics, predictions


def train_single_fold(
    fold_idx: int,
    num_folds: int,
    cfg: Dict,
    output_base: Path,
    logger,
) -> Dict[str, Any]:
    """Train a single fold."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
    logger.info(f"{'='*80}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Regression config
    reg_cfg = cfg.get("regression", {})
    infection_onset_hour = float(reg_cfg.get("infection_onset_hour", 1.0))
    clamp_range = tuple(reg_cfg.get("clamp_range", [0.0, 48.0]))
    target_class = reg_cfg.get("target_class")  # None, 0 (uninfected), or 1 (infected)
    
    if target_class is not None:
        class_name = "infected" if target_class == 1 else "uninfected"
        logger.info(f"Training on {class_name} samples only (target_class={target_class})")
    else:
        logger.info("Training on all samples (both infected and uninfected)")
    
    logger.info(f"Infection onset: {infection_onset_hour}h")
    logger.info(f"Clamp range: {clamp_range}")
    
    # Build datasets for this fold
    data_cfg = cfg.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    logger.info(f"Building datasets for fold {fold_idx + 1}...")
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=fold_idx,
        num_folds=num_folds,
    )
    
    logger.info(f"Fold {fold_idx + 1} - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Data loaders
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Build model
    model_cfg = cfg.get("model", {})
    model = build_regression_model(model_cfg)
    model = model.to(device)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    
    # Scheduler
    scheduler_cfg = cfg.get("scheduler", {})
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 30)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_cfg.get("t_max", epochs),
        eta_min=scheduler_cfg.get("eta_min", 1e-6),
    )
    
    # Loss function
    criterion = nn.SmoothL1Loss()
    
    # Training setup
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    
    # Checkpointing
    fold_checkpoint_dir = output_base / f"fold_{fold_idx + 1}" / "checkpoints"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = fold_checkpoint_dir / "best.pt"
    
    primary_metric = "reg_mae"
    best_score = math.inf  # Lower is better for MAE
    
    logger.info(f"Training fold {fold_idx + 1} for {epochs} epochs...")
    logger.info(f"Primary metric: {primary_metric} (lower is better)")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"Fold {fold_idx + 1}, Epoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            target_class=target_class,
            use_amp=use_amp,
            grad_clip=grad_clip,
            progress_desc=f"F{fold_idx+1}_E{epoch}_train",
        )
        
        logger.info(f"  Train loss: {train_metrics['loss']:.4f}")
        
        # Validate
        val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            target_class=target_class,
            split_name="val",
            progress_desc=f"F{fold_idx+1}_E{epoch}_val",
        )
        
        val_summary = " | ".join(f"{k}:{v:.4f}" for k, v in val_metrics.items())
        logger.info(f"  Val: {val_summary}")
        
        scheduler.step()
        
        # Save best model
        metric_value = val_metrics.get(primary_metric)
        if metric_value is not None and not math.isnan(metric_value):
            if metric_value < best_score:
                best_score = metric_value
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": cfg,
                        "fold_index": fold_idx,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_checkpoint,
                )
                logger.info(f"✓ Fold {fold_idx + 1}: New best model! {primary_metric}={best_score:.4f}")
    
    # Load best model for final test
    logger.info(f"Loading best model for fold {fold_idx + 1}...")
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    # Final test evaluation
    logger.info(f"Fold {fold_idx + 1}: Final evaluation on test set")
    test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        infection_onset_hour=infection_onset_hour,
        clamp_range=clamp_range,
        target_class=target_class,
        split_name="test",
        progress_desc=f"F{fold_idx+1}_test",
    )
    
    # Log test results
    summary = " | ".join(f"{k}:{v:.4f}" for k, v in test_metrics.items())
    logger.info(f"Fold {fold_idx + 1} test results: {summary}")
    
    # Save per-sample predictions for downstream analysis
    fold_dir = output_base / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = fold_dir / "test_predictions.npz"
    
    try:
        # Save in format compatible with multitask analysis scripts
        # For regression-only, we don't have classification predictions, so we use dummy values
        labels = test_predictions.get("labels")
        
        np.savez(
            predictions_file,
            time_preds=test_predictions.get("time_preds"),
            time_targets=test_predictions.get("time_targets"),
            cls_targets=labels,  # Actual labels (for stratified analysis)
            cls_preds=np.full_like(labels, 0.5, dtype=np.float32),  # Dummy (not used in regression analysis)
        )
        logger.info(f"✓ Saved test predictions to {predictions_file}")
    except Exception as e:
        logger.warning(f"Failed to save {predictions_file}: {e}")
    
    # Save metadata
    metadata_file = fold_dir / "test_metadata.jsonl"
    try:
        if hasattr(test_ds, "get_metadata"):
            with open(metadata_file, "w", encoding="utf-8") as f:
                for i in range(len(test_ds)):
                    meta = test_ds.get_metadata(i)
                    if not isinstance(meta, dict):
                        meta = {"meta": meta}
                    f.write(json.dumps(meta) + "\n")
            logger.info(f"✓ Saved test metadata to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save {metadata_file}: {e}")
    
    # Save fold results
    fold_results = {
        "fold_index": fold_idx,
        "best_val_metric": float(best_score),
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in test_metrics.items()},
    }
    
    fold_results_file = fold_dir / "results.json"
    with open(fold_results_file, "w") as f:
        json.dump(fold_results, f, indent=2)
    
    logger.info(f"✓ Fold {fold_idx + 1} complete!")
    
    return fold_results


def aggregate_cv_results(fold_results: List[Dict], output_base: Path, logger):
    """Aggregate results across all folds."""
    
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}\n")
    
    # Collect metrics across folds
    metrics_names = list(fold_results[0]["test_metrics"].keys())
    
    aggregated = {}
    for metric_name in metrics_names:
        values = [fold["test_metrics"][metric_name] for fold in fold_results]
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values],
        }
    
    # Log summary
    logger.info("Mean ± Std across folds:")
    for metric_name, stats in aggregated.items():
        logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save aggregated results
    cv_summary = {
        "num_folds": len(fold_results),
        "fold_results": fold_results,
        "aggregated_metrics": aggregated,
    }
    
    summary_file = output_base / "cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    logger.info(f"\n✓ CV summary saved to {summary_file}")
    
    return cv_summary


def main():
    parser = argparse.ArgumentParser(description="Regression-only 5-fold CV training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Setup output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        experiment_name = cfg.get("experiment_name", "regression_cv")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_base = Path("outputs") / experiment_name / f"{timestamp}_{args.num_folds}fold"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger("regression_cv", output_base / "train_cv.log")
    
    logger.info("="*80)
    logger.info("REGRESSION-ONLY MODEL - CROSS-VALIDATION TRAINING")
    logger.info("="*80)
    logger.info(f"Number of folds: {args.num_folds}")
    logger.info(f"Output directory: {output_base}")
    
    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Train each fold
    fold_results = []
    for fold_idx in range(args.num_folds):
        fold_result = train_single_fold(
            fold_idx=fold_idx,
            num_folds=args.num_folds,
            cfg=cfg,
            output_base=output_base,
            logger=logger,
        )
        fold_results.append(fold_result)
    
    # Aggregate results
    cv_summary = aggregate_cv_results(fold_results, output_base, logger)
    
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_base}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
