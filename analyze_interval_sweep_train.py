"""
Interval Sweep Training Analysis

This script trains separate models for different infected interval ranges and
evaluates their performance. It shows how model performance changes as you vary
the training data's time range.

For each interval configuration:
1. Filter datasets to specified time ranges
2. Train a new model from scratch on that filtered data  
3. Test the model
4. Record metrics across K-folds

Two modes:
- train-test: Both train and test use [start, x]
- test-only: Train uses full range, test uses [start, x]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import amp

from datasets import build_datasets
from models import build_model
from utils import build_transforms, load_config, set_seed, get_logger
from train import (
    get_task_config,
    get_analysis_config,
    create_dataloaders,
    train_one_epoch,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train models across different infected interval upper bounds"
    )
    parser.add_argument(
        "--config",
        default="configs/resnet50_baseline.yaml",
        help="Path to YAML config for training",
    )
    parser.add_argument(
        "--upper-hours",
        type=float,
        nargs="+",
        required=True,
        help="Upper bounds X for intervals [start, X]",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=1.0,
        help="Lower bound for infected interval",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for final evaluation",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot (e.g., auc accuracy f1). "
             "If multiple provided, creates combined plot + individual plots. "
             "If not provided, uses --metric for backward compatibility",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        help="Single metric key to summarize (e.g., auc, accuracy, f1). "
             "Use --metrics for multiple metrics",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots, logs, and JSON summaries",
    )
    parser.add_argument(
        "--save-data",
        default=None,
        help="Optional path to save raw sweep metrics as JSON",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help="Number of folds for cross-validation (overrides config if provided)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config if provided)",
    )
    return parser.parse_args()


_SPLIT_OVERRIDE_KEYS = ("train", "val", "test", "default", "eval", "evaluation")


def apply_interval_override(
    base_frames_cfg: Dict | None, start: float, end: float, mode: str
) -> Dict:
    """
    Apply interval [start, end] to dataset splits based on mode.
    
    Args:
        mode: "train-test" (both use [start, end]) or "test-only" (only test uses [start, end])
    """
    frames_cfg = copy.deepcopy(base_frames_cfg) if base_frames_cfg else {}
    
    if mode == "train-test":
        # Both train and test use the restricted interval
        frames_cfg["infected_window_hours"] = [start, end]
        for key in _SPLIT_OVERRIDE_KEYS:
            if key in frames_cfg or key in ("train", "val", "test"):
                section = copy.deepcopy(frames_cfg.get(key, {}))
                section["infected_window_hours"] = [start, end]
                frames_cfg[key] = section
    else:  # test-only
        # Only test uses restricted interval, train uses full range
        test_section = copy.deepcopy(frames_cfg.get("test", {}))
        test_section["infected_window_hours"] = [start, end]
        frames_cfg["test"] = test_section
    
    return frames_cfg


def train_and_evaluate_interval(
    cfg: Dict,
    base_data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    hour: float,
    start_hour: float,
    mode: str,
    eval_split: str,
    metric_keys: List[str],
    k_folds: int,
    device,
    logger,
    output_dir: Path,
) -> Dict[str, Tuple[List[float], float | None, float | None]]:
    """
    Train models for one interval configuration across all folds.
    
    Returns:
        Dictionary mapping metric_key -> (fold_metrics, mean, std)
    """
    # Apply interval to data config
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["frames"] = apply_interval_override(
        base_data_cfg.get("frames"), start_hour, hour, mode
    )
    
    training_cfg = cfg.get("training", {})
    task_cfg = get_task_config(cfg)
    analysis_cfg = get_analysis_config(cfg)
    model_cfg = cfg.get("model", {})
    optimizer_cfg = cfg.get("optimizer", {"lr": 1e-4})
    scheduler_cfg = cfg.get("scheduler")
    
    epochs = training_cfg.get("epochs", 10)
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    amp_device = device.type
    
    task_type = task_cfg.get("type", "classification")
    primary_metric = "auc" if task_type == "classification" else "rmse"
    greater_is_better = task_type == "classification"
    
    # Initialize storage for each metric
    fold_metrics_dict: Dict[str, List[float]] = {key: [] for key in metric_keys}
    
    for fold_idx in range(k_folds):
        fold_tag = f"fold_{fold_idx + 1:02d}of{k_folds:02d}"
        logger.info(f"[Interval [{start_hour:.1f}, {hour:.1f}]h, mode={mode}] Training {fold_tag}")
        
        # Build datasets
        train_ds, val_ds, test_ds = build_datasets(
            data_cfg,
            transforms,
            fold_index=fold_idx if k_folds > 1 else None,
            num_folds=k_folds,
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, data_cfg
        )
        
        # Build fresh model for this fold
        model = build_model(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_cfg)
        
        scheduler = None
        if scheduler_cfg:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_cfg.get("t_max", 10),
                eta_min=scheduler_cfg.get("eta_min", 1e-6),
            )
        
        if task_type == "regression":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        scaler = amp.GradScaler(amp_device, enabled=use_amp and amp_device == "cuda")
        
        best_score = -math.inf if greater_is_better else math.inf
        best_metrics = None
        
        # Training loop
        for epoch in range(1, epochs + 1):
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
                progress_desc=f"[{start_hour:.0f},{hour:.0f}]_{mode}_f{fold_idx+1}_e{epoch}",
                task_cfg=task_cfg,
            )
            
            # Validate
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
            
            # Track best model
            metric_value = val_metrics.get(primary_metric)
            is_valid = metric_value is not None and not math.isnan(metric_value)
            
            if is_valid:
                is_better = (
                    (greater_is_better and metric_value > best_score)
                    or (not greater_is_better and metric_value < best_score)
                )
                if is_better:
                    best_score = metric_value
                    # Evaluate on final split with best model
                    eval_loader = {"val": val_loader, "test": test_loader}[eval_split]
                    best_metrics = evaluate(
                        model,
                        eval_loader,
                        criterion,
                        device,
                        logger,
                        split_name=eval_split,
                        progress_desc=f"{eval_split}_f{fold_idx + 1}",
                        task_cfg=task_cfg,
                        analysis_cfg=analysis_cfg,
                    )
        
        # Collect metrics from best model
        if best_metrics:
            for metric_key in metric_keys:
                metric_value = best_metrics.get(metric_key)
                if metric_value is not None and not math.isnan(metric_value):
                    fold_metrics_dict[metric_key].append(float(metric_value))
        
        # Cleanup
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Compute statistics for each metric
    results = {}
    for metric_key in metric_keys:
        fold_values = fold_metrics_dict[metric_key]
        if fold_values:
            values = np.asarray(fold_values, dtype=np.float32)
            results[metric_key] = (fold_values, float(values.mean()), float(values.std(ddof=0)))
        else:
            results[metric_key] = ([], None, None)
    
    return results


def plot_single_metric_sweep(
    hours: List[float],
    stats: Dict[str, Dict[str, List[float]]],
    metric: str,
    output_path: Path,
) -> None:
    """Plot error bars for a single metric across both modes."""
    modes = ["train-test", "test-only"]
    labels = ["Train=[start,x], Test=[start,x]", "Train=full, Test=[start,x]"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for ax, mode, label in zip(axes, modes, labels):
        mode_stats = stats.get(mode, {})
        means = mode_stats.get("mean", [])
        stds = mode_stats.get("std", [])
        
        valid_points = [
            (hour, mean, stds[idx] if idx < len(stds) and stds[idx] is not None else 0.0)
            for idx, (hour, mean) in enumerate(zip(hours, means))
            if mean is not None
        ]
        
        if valid_points:
            v_hours, v_means, v_stds = zip(*valid_points)
            ax.errorbar(v_hours, v_means, yerr=v_stds, fmt="-o", capsize=4, linewidth=2, markersize=6)
        
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Upper bound hour (x)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    
    axes[0].set_ylabel(metric.upper(), fontsize=11)
    fig.suptitle(f"Interval Sweep Training Analysis: {metric.upper()}", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot for %s to %s", metric, output_path)


def plot_multi_metric_sweep(
    hours: List[float],
    all_stats: Dict[str, Dict[str, Dict[str, List[float]]]],
    output_path: Path,
) -> None:
    """Plot combined error bars for multiple metrics."""
    modes = ["train-test", "test-only"]
    labels = ["Train=[start,x], Test=[start,x]", "Train=full, Test=[start,x]"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_stats)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for ax, mode, label in zip(axes, modes, labels):
        for idx, (metric, stats) in enumerate(all_stats.items()):
            mode_stats = stats.get(mode, {})
            means = mode_stats.get("mean", [])
            stds = mode_stats.get("std", [])
            
            valid_points = [
                (hour, mean, stds[j] if j < len(stds) and stds[j] is not None else 0.0)
                for j, (hour, mean) in enumerate(zip(hours, means))
                if mean is not None
            ]
            
            if valid_points:
                v_hours, v_means, v_stds = zip(*valid_points)
                marker = markers[idx % len(markers)]
                ax.errorbar(
                    v_hours,
                    v_means,
                    yerr=v_stds,
                    fmt=f"-{marker}",
                    capsize=4,
                    linewidth=1.5,
                    markersize=6,
                    color=colors[idx],
                    label=metric.upper(),
                    alpha=0.8,
                )
        
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Upper bound hour (x)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="best")
    
    axes[0].set_ylabel("Metric Value", fontsize=11)
    fig.suptitle("Interval Sweep Training Analysis: Multiple Metrics", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved combined multi-metric plot to %s", output_path)


def main() -> None:
    args = parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / "interval_sweep_analysis" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        name="interval_sweep_train",
        log_dir=output_dir,
        log_file="interval_sweep_train.log"
    )
    
    logger.info("="*60)
    logger.info("Interval Sweep Training Analysis")
    logger.info("="*60)
    
    # Load configuration
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    
    # Override config with args if provided
    if args.k_folds is not None:
        cfg.setdefault("training", {})["k_folds"] = args.k_folds
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    
    data_cfg = cfg.get("data", {})
    base_data_cfg = copy.deepcopy(data_cfg)
    
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)
    
    training_cfg = cfg.get("training", {})
    k_folds = max(1, int(training_cfg.get("k_folds", 1)))
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"K-folds: {k_folds}")
    
    hours = sorted(args.upper_hours)
    if any(hour < args.start_hour for hour in hours):
        raise ValueError("All upper bound hours must be >= start-hour")
    
    # Determine metrics
    if args.metrics:
        metric_keys = args.metrics
        logger.info(f"Evaluating metrics: {', '.join(metric_keys)}")
    else:
        metric_keys = [args.metric]
        logger.info(f"Evaluating single metric: {args.metric}")
    
    modes = ("train-test", "test-only")
    
    logger.info(f"Will train {len(hours)} intervals × 2 modes × {k_folds} folds")
    logger.info(f"Total training runs: {len(hours) * 2 * k_folds}")
    logger.info("="*60)
    
    # Structure: all_stats[metric][mode] = {"fold_metrics": [...], "mean": [...], "std": [...]}
    all_stats: Dict[str, Dict[str, Dict[str, List]]] = {
        metric_key: {
            mode: {"fold_metrics": [], "mean": [], "std": []} for mode in modes
        }
        for metric_key in metric_keys
    }
    
    for mode_idx, mode in enumerate(modes, 1):
        logger.info(f"\n[Mode {mode_idx}/2] Evaluating mode={mode}")
        for hour_idx, hour in enumerate(hours, 1):
            logger.info(f"  [{hour_idx}/{len(hours)}] Interval [{args.start_hour:.1f}, {hour:.1f}]h")
            
            interval_results = train_and_evaluate_interval(
                cfg,
                base_data_cfg,
                transforms,
                hour,
                args.start_hour,
                mode,
                args.split,
                metric_keys,
                k_folds,
                device,
                logger,
                output_dir,
            )
            
            # Store results for each metric
            for metric_key in metric_keys:
                fold_values, mean_value, std_value = interval_results[metric_key]
                all_stats[metric_key][mode]["fold_metrics"].append(fold_values)
                all_stats[metric_key][mode]["mean"].append(mean_value)
                all_stats[metric_key][mode]["std"].append(std_value)
                
                if mean_value is not None:
                    logger.info(
                        f"    {metric_key.upper()} = {mean_value:.4f} ± {std_value:.4f}"
                    )
    
    # Generate plots
    if len(metric_keys) > 1:
        logger.info("\nCreating combined plot for all metrics...")
        combined_plot_path = output_dir / "interval_sweep_combined.png"
        plot_multi_metric_sweep(hours, all_stats, combined_plot_path)
    
    for metric_key in metric_keys:
        plot_path = output_dir / f"interval_sweep_{metric_key}.png"
        plot_single_metric_sweep(hours, all_stats[metric_key], metric_key, plot_path)
    
    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / "interval_sweep_data.json"
    
    payload = {
        "config": args.config,
        "timestamp": timestamp,
        "start_hour": args.start_hour,
        "hours": hours,
        "metrics": metric_keys,
        "split": args.split,
        "k_folds": k_folds,
        "stats": all_stats,
    }
    
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    
    logger.info(f"\nSaved data to {save_path}")
    logger.info("="*60)
    logger.info(f"Analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
