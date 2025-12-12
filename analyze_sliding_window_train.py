"""
Sliding Window Training & Evaluation Analysis

This script trains separate models for each time window [x, x+k] and evaluates
them on the same window. It shows which time periods contain the most useful
signal for infection classification.

For each window:
1. Filter train/val/test to use only frames from [x, x+k]
2. Train a new model from scratch on that filtered data
3. Test the model on the same window
4. Record metrics across K-folds

Finally, plot how performance varies across different time windows.
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
from torch.utils.data import DataLoader

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
        description="Train models on sliding window intervals [x, x+k] and evaluate performance"
    )
    parser.add_argument(
        "--config",
        default="configs/resnet50_baseline.yaml",
        help="Path to YAML config for training",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Window size k (hours). Train and test both use [x, x+k]",
    )
    parser.add_argument(
        "--window-starts",
        type=float,
        nargs="+",
        default=None,
        help="Starting hours x for windows [x, x+k]. E.g., 0 2 4 6 8 10. "
             "If not provided, windows are auto-generated using --start-hour, --end-hour, and --stride",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=1.0,
        help="Starting hour for first window (used when --window-starts not provided)",
    )
    parser.add_argument(
        "--end-hour",
        type=float,
        default=30.0,
        help="Maximum ending hour (used when --window-starts not provided)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Step size between consecutive windows (hours). "
             "Default is window-size (no overlap). "
             "stride < window-size creates overlap, stride > window-size creates gaps",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for final evaluation (val or test)",
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
        "--max-hour",
        type=float,
        default=None,
        help="Maximum hour to consider (optional). Windows extending beyond this are skipped.",
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
        help="Number of training epochs per window (overrides config if provided)",
    )
    parser.add_argument(
        "--match-uninfected-window",
        action="store_true",
        help="Apply the same time window to uninfected samples (default: uninfected uses all time points). "
             "When enabled, both infected and uninfected use the exact same [x, x+k] window.",
    )
    return parser.parse_args()


_SPLIT_OVERRIDE_KEYS = ("train", "val", "test", "default", "eval", "evaluation")


def apply_sliding_window(
    base_frames_cfg: Dict | None, start: float, end: float, match_uninfected: bool = False
) -> Dict:
    """
    Apply sliding window [start, end] to ALL splits (train, val, test).
    Both training and testing use the same time interval.
    
    Args:
        base_frames_cfg: Base frame configuration
        start: Window start hour
        end: Window end hour
        match_uninfected: If True, apply same window to uninfected samples.
                         If False (default), uninfected uses all time points.
    """
    frames_cfg = copy.deepcopy(base_frames_cfg) if base_frames_cfg else {}
    
    # Set the base infected window
    frames_cfg["infected_window_hours"] = [start, end]
    
    # Apply window to uninfected if requested
    if match_uninfected:
        frames_cfg["uninfected_window_hours"] = [start, end]
    
    # Override all split-specific configurations to use the same window
    for key in _SPLIT_OVERRIDE_KEYS:
        if key in frames_cfg or key in ("train", "val", "test"):
            section = copy.deepcopy(frames_cfg.get(key, {}))
            section["infected_window_hours"] = [start, end]
            if match_uninfected:
                section["uninfected_window_hours"] = [start, end]
            frames_cfg[key] = section
    
    return frames_cfg


def train_and_evaluate_window(
    cfg: Dict,
    base_data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    window_start: float,
    window_end: float,
    eval_split: str,
    metric_keys: List[str],
    k_folds: int,
    device,
    logger,
    output_dir: Path,
    match_uninfected: bool = False,
) -> Dict[str, Tuple[List[float], float | None, float | None]]:
    """
    Train models for one time window across all folds and evaluate.
    
    Args:
        window_start, window_end: Time interval for this window
        metric_keys: List of metric names to extract
        k_folds: Number of cross-validation folds
        match_uninfected: Whether to apply same window to uninfected samples
        
    Returns:
        Dictionary mapping metric_key -> (fold_metrics, mean, std)
    """
    # Apply window to data config
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["frames"] = apply_sliding_window(
        base_data_cfg.get("frames"), window_start, window_end, match_uninfected
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
        logger.info(f"[Window {window_start:.1f}-{window_end:.1f}h] Training {fold_tag}")
        
        # Build datasets with the windowed frame policy
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
                progress_desc=f"W[{window_start:.0f}-{window_end:.0f}]_f{fold_idx + 1}_e{epoch}",
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
            
            # Track best model based on validation metric
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
                    
                    # Save checkpoint for best model
                    checkpoint_dir = output_dir / "checkpoints" / f"window_{window_start:.0f}-{window_end:.0f}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"fold_{fold_idx + 1:02d}_best.pth"
                    
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'window_start': window_start,
                        'window_end': window_end,
                        'fold': fold_idx + 1,
                        'best_val_score': best_score,
                        'best_metrics': best_metrics,
                        'config': cfg,
                    }
                    if scheduler:
                        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
        
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


def plot_single_metric(
    window_centers: List[float],
    means: List[float],
    stds: List[float],
    window_size: float,
    metric: str,
    output_path: Path,
) -> None:
    """Create an error bar plot for a single metric."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    valid_points = [
        (center, mean, std)
        for center, mean, std in zip(window_centers, means, stds)
        if mean is not None
    ]
    
    if not valid_points:
        logging.warning("No valid data points to plot for metric '%s'", metric)
        plt.close(fig)
        return
    
    v_centers, v_means, v_stds = zip(*valid_points)
    
    ax.errorbar(
        v_centers,
        v_means,
        yerr=v_stds,
        fmt="-o",
        capsize=5,
        linewidth=2,
        markersize=8,
        label=f"Window size = {window_size}h",
    )
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"Sliding Window Training Analysis: {metric.upper()} vs Time Window\n"
        f"(Models trained & tested on [x, x+{window_size}h])",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour (x)", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot for %s to %s", metric, output_path)


def plot_multi_metric(
    window_centers: List[float],
    metric_results: Dict[str, Tuple[List[float], List[float]]],
    window_size: float,
    output_path: Path,
) -> None:
    """Create a combined error bar plot for multiple metrics."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    any_valid = False
    for idx, (metric, (means, stds)) in enumerate(metric_results.items()):
        valid_points = [
            (center, mean, std)
            for center, mean, std in zip(window_centers, means, stds)
            if mean is not None
        ]
        
        if not valid_points:
            logging.warning("No valid data points for metric '%s'", metric)
            continue
        
        any_valid = True
        v_centers, v_means, v_stds = zip(*valid_points)
        
        marker = markers[idx % len(markers)]
        ax.errorbar(
            v_centers,
            v_means,
            yerr=v_stds,
            fmt=f"-{marker}",
            capsize=4,
            linewidth=1.5,
            markersize=7,
            color=colors[idx],
            label=metric.upper(),
            alpha=0.8,
        )
    
    if not any_valid:
        logging.warning("No valid data to plot for any metric")
        plt.close(fig)
        return
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"Sliding Window Training Analysis: Multiple Metrics\n"
        f"(Models trained & tested on [x, x+{window_size}h])",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="best")
    
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour (x)", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
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
        output_dir = Path("outputs") / "sliding_window_analysis" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        name="sliding_window_train",
        log_dir=output_dir
    )
    
    logger.info("="*60)
    logger.info("Sliding Window Training Analysis")
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
    
    # Log uninfected matching mode
    if args.match_uninfected_window:
        logger.info("✓ Match uninfected window: ENABLED (infected and uninfected use same time window)")
    else:
        logger.info("✗ Match uninfected window: DISABLED (uninfected uses all time points)")
    
    # Validate window configurations
    window_size = args.window_size
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    stride = args.stride if args.stride is not None else window_size
    if stride <= 0:
        raise ValueError("Stride must be positive")
    
    # Generate or use provided window starts
    if args.window_starts is not None:
        window_starts = sorted(args.window_starts)
        logger.info(f"Using {len(window_starts)} user-provided window start positions")
    else:
        window_starts = []
        current = args.start_hour
        max_end = args.max_hour if args.max_hour is not None else args.end_hour
        
        while current + window_size <= max_end:
            window_starts.append(current)
            current += stride
        
        logger.info(
            f"Auto-generated {len(window_starts)} windows: "
            f"start={args.start_hour:.1f}h, end={max_end:.1f}h, "
            f"size={window_size:.1f}h, stride={stride:.1f}h"
        )
        
        if not window_starts:
            raise ValueError(
                f"No valid windows with start={args.start_hour}, "
                f"end={max_end}, size={window_size}"
            )
    
    # Build window intervals
    windows: List[Tuple[float, float, float]] = []
    for start in window_starts:
        end = start + window_size
        
        if args.max_hour is not None and end > args.max_hour:
            logger.info(f"Skipping window [{start:.1f}, {end:.1f}]h (exceeds max_hour={args.max_hour:.1f})")
            continue
        
        center = start + window_size / 2.0
        windows.append((start, end, center))
    
    if not windows:
        raise ValueError("No valid windows to evaluate")
    
    # Determine metrics
    if args.metrics:
        metric_keys = args.metrics
        logger.info(f"Evaluating metrics: {', '.join(metric_keys)}")
    else:
        metric_keys = [args.metric]
        logger.info(f"Evaluating single metric: {args.metric}")
    
    logger.info(f"Will train {len(windows)} models (one per window) with {k_folds}-fold CV")
    logger.info(f"Total training runs: {len(windows) * k_folds}")
    logger.info("="*60)
    
    # Train and evaluate each window
    results: Dict[str, Dict[str, List]] = {
        metric_key: {
            "window_starts": [],
            "window_ends": [],
            "window_centers": [],
            "fold_metrics": [],
            "means": [],
            "stds": [],
        }
        for metric_key in metric_keys
    }
    
    for window_idx, (start, end, center) in enumerate(windows, 1):
        logger.info(f"\n[{window_idx}/{len(windows)}] Training window [{start:.1f}, {end:.1f}]h (center={center:.1f})")
        
        window_results = train_and_evaluate_window(
            cfg,
            base_data_cfg,
            transforms,
            start,
            end,
            args.split,
            metric_keys,
            k_folds,
            device,
            logger,
            output_dir,
            args.match_uninfected_window,
        )
        
        # Store results for each metric
        for metric_key in metric_keys:
            fold_values, mean_value, std_value = window_results[metric_key]
            
            results[metric_key]["window_starts"].append(start)
            results[metric_key]["window_ends"].append(end)
            results[metric_key]["window_centers"].append(center)
            results[metric_key]["fold_metrics"].append(fold_values)
            results[metric_key]["means"].append(mean_value)
            results[metric_key]["stds"].append(std_value)
            
            if mean_value is not None:
                logger.info(f"  {metric_key.upper()} = {mean_value:.4f} ± {std_value:.4f}")
        
        if all(window_results[mk][1] is None for mk in metric_keys):
            logger.warning(f"  Window [{start:.1f}, {end:.1f}]h: No valid metrics")
    
    # Generate plots
    stride_suffix = f"_s{stride:.0f}" if stride != window_size else ""
    base_filename = f"sliding_window_w{window_size:.0f}{stride_suffix}"
    
    if len(metric_keys) > 1:
        logger.info("\nCreating combined plot for all metrics...")
        combined_plot_path = output_dir / f"{base_filename}_combined.png"
        
        metric_plot_data = {}
        for metric_key in metric_keys:
            metric_plot_data[metric_key] = (
                results[metric_key]["means"],
                results[metric_key]["stds"],
            )
        
        window_centers = results[metric_keys[0]]["window_centers"]
        plot_multi_metric(window_centers, metric_plot_data, window_size, combined_plot_path)
    
    for metric_key in metric_keys:
        metric_result = results[metric_key]
        plot_path = output_dir / f"{base_filename}_{metric_key}.png"
        
        plot_single_metric(
            metric_result["window_centers"],
            metric_result["means"],
            metric_result["stds"],
            window_size,
            metric_key,
            plot_path,
        )
    
    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / f"{base_filename}_data.json"
    
    payload = {
        "config": args.config,
        "timestamp": timestamp,
        "window_size": window_size,
        "stride": stride,
        "window_starts": window_starts,
        "max_hour": args.max_hour,
        "metrics": metric_keys,
        "split": args.split,
        "k_folds": k_folds,
        "results": results,
    }
    
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    
    logger.info(f"\nSaved data to {save_path}")
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("Summary Statistics:")
    logger.info("="*60)
    
    for metric_key in metric_keys:
        metric_result = results[metric_key]
        valid_means = [m for m in metric_result["means"] if m is not None]
        
        if valid_means:
            best_idx = metric_result["means"].index(max(valid_means))
            logger.info(f"\n{metric_key.upper()}:")
            logger.info(f"  Best window: [{metric_result['window_starts'][best_idx]:.1f}, "
                       f"{metric_result['window_ends'][best_idx]:.1f}]h with value={max(valid_means):.4f}")
            logger.info(f"  Overall mean: {np.mean(valid_means):.4f} ± "
                       f"{np.std(valid_means, ddof=1 if len(valid_means) > 1 else 0):.4f}")
        else:
            logger.warning(f"\n{metric_key.upper()}: No valid data")
    
    logger.info("="*60)
    logger.info(f"Analysis complete! Results saved to: {output_dir}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - Plots: {output_dir}/*.png")
    logger.info(f"  - Data: {save_path}")
    logger.info(f"  - Checkpoints: {output_dir}/checkpoints/window_*/fold_*_best.pth")
    logger.info(f"  - Log: {output_dir}/sliding_window_train.log")
    logger.info("="*60)


if __name__ == "__main__":
    main()
