"""
Final Model Sliding Window Evaluation (OPTIMIZED)

This script evaluates a pre-trained model across different sliding time windows
WITHOUT retraining. OPTIMIZED VERSION: Evaluates on ALL test data ONCE, then
groups results by time window for fast analysis.

Key optimization:
- OLD: Load data N times (once per window) - SLOW
- NEW: Load data ONCE, evaluate ONCE, group by window - FAST

For each fold:
1. Load the pre-trained model checkpoint
2. Evaluate on ALL test data (full time range)
3. Group predictions by time window [x, x+k] in post-processing
4. Compute metrics for each window

This is much faster when testing many overlapping windows!
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from datasets import build_datasets
from models import build_model
from utils import build_transforms, load_config, set_seed, get_logger
from train import (
    get_task_config,
    get_analysis_config,
    create_dataloaders,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-trained model across sliding time windows (OPTIMIZED: evaluate once, group by window)"
    )
    parser.add_argument(
        "--config",
        default="configs/resnet50_baseline.yaml",
        help="Path to YAML config (should match the config used for training)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to checkpoint directory containing fold_XX_best.pth files. "
             "Example: outputs/interval_sweep_analysis/20251212-145928/checkpoints/train-test_interval_1-46",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Window size k (hours). Results grouped into [x, x+k]",
    )
    parser.add_argument(
        "--window-starts",
        type=float,
        nargs="+",
        default=None,
        help="Starting hours x for windows [x, x+k]. E.g., 1 4 7 10. "
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
        default=46.0,
        help="Maximum ending hour (used when --window-starts not provided)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Step size between consecutive windows (hours). "
             "Default is window-size (no overlap). "
             "stride < window-size creates overlap (same samples in multiple windows)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for evaluation",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auc", "accuracy", "f1"],
        help="Metrics to compute (e.g., auc accuracy f1 precision recall)",
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
        help="Number of folds (should match checkpoint directory, overrides config if provided)",
    )
    parser.add_argument(
        "--time-field",
        default="hours_since_start",
        help="Field name in dataset metadata containing time information (default: hours_since_start)",
    )
    return parser.parse_args()


def evaluate_full_and_group_by_window(
    model: nn.Module,
    dataloader,
    device,
    windows: List[Tuple[float, float, float]],  # (start, end, center)
    time_field: str,
    metric_keys: List[str],
    logger,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model on ALL data once, then group results by time window.
    
    Args:
        model: Pre-trained model
        dataloader: DataLoader with ALL test data
        windows: List of (start, end, center) tuples for windows
        time_field: Name of time field in dataset metadata
        metric_keys: Metrics to compute
        
    Returns:
        Dictionary mapping window_center -> {metric: value}
    """
    model.eval()
    
    # Collect all predictions and metadata
    all_preds = []
    all_labels = []
    all_times = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle both tuple (image, label, metadata) and dict formats
            if isinstance(batch, (tuple, list)):
                # Tuple format: (image, label, metadata)
                images = batch[0].to(device)
                labels = batch[1].to(device)
                metadata = batch[2] if len(batch) > 2 else {}
            elif isinstance(batch, dict):
                # Dict format: {"image": ..., "label": ..., ...}
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                metadata = batch.get("metadata", batch)
            else:
                logger.warning(f"Unknown batch format: {type(batch)}, skipping")
                continue
            
            # Get time information from metadata
            if isinstance(metadata, dict):
                # Check different possible field names
                if time_field in metadata:
                    times = metadata[time_field]
                elif "hours_since_start" in metadata:
                    times = metadata["hours_since_start"]
                else:
                    logger.warning(f"Time field '{time_field}' not found in metadata keys: {metadata.keys()}, skipping batch")
                    continue
            elif isinstance(metadata, (list, tuple)):
                # Metadata is a list/tuple of dicts (one per sample in batch)
                times = [m.get(time_field, m.get("hours_since_start", None)) for m in metadata]
                if any(t is None for t in times):
                    logger.warning(f"Some samples missing time field '{time_field}', skipping batch")
                    continue
            else:
                logger.warning(f"Unknown metadata format: {type(metadata)}, skipping batch")
                continue
            
            # Forward pass
            outputs = model(images)
            
            # Handle multi-task or single-task output
            if isinstance(outputs, dict):
                logits = outputs.get("infection", outputs.get("main", None))
            else:
                logits = outputs
            
            if logits is None:
                logger.warning("Could not find infection logits in model output")
                continue
            
            # Store predictions (convert logits to probabilities)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            
            # Handle time as tensor or list
            if isinstance(times, torch.Tensor):
                times = times.cpu().numpy().flatten().tolist()
            elif not isinstance(times, list):
                times = [times]
            
            all_times.extend(times)
    
    if len(all_preds) == 0:
        logger.warning("No predictions collected!")
        return {}
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_times = np.array(all_times)
    
    logger.info(f"  Total samples evaluated: {len(all_preds)}")
    logger.info(f"  Time range: [{all_times.min():.2f}, {all_times.max():.2f}]h")
    
    # Group by windows and compute metrics
    window_results = {}
    
    for start, end, center in windows:
        # Find samples in this time window
        mask = (all_times >= start) & (all_times <= end)
        n_samples = mask.sum()
        
        if n_samples == 0:
            logger.warning(f"  Window [{start:.1f}, {end:.1f}]h: No samples found")
            window_results[center] = {metric: None for metric in metric_keys}
            continue
        
        # Get predictions and labels for this window
        window_preds = all_preds[mask]
        window_labels = all_labels[mask]
        window_pred_binary = (window_preds >= 0.5).astype(int)
        
        # Compute metrics
        metrics = {}
        
        # Check if we have both classes
        unique_labels = np.unique(window_labels)
        n_infected = np.sum(window_labels == 1)
        n_uninfected = np.sum(window_labels == 0)
        
        for metric_key in metric_keys:
            try:
                if metric_key == "auc":
                    if len(unique_labels) > 1:
                        metrics[metric_key] = roc_auc_score(window_labels, window_preds)
                    else:
                        metrics[metric_key] = None
                        if len(unique_labels) == 1:
                            only_class = "infected" if unique_labels[0] == 1 else "uninfected"
                            logger.info(f"  Window [{start:.1f}, {end:.1f}]h: Skipping AUC (only {only_class} samples: {n_samples} total)")
                
                elif metric_key == "accuracy":
                    metrics[metric_key] = accuracy_score(window_labels, window_pred_binary)
                
                elif metric_key == "f1":
                    _, _, f1, _ = precision_recall_fscore_support(
                        window_labels, window_pred_binary, average="binary", zero_division=0
                    )
                    metrics[metric_key] = f1
                
                elif metric_key == "precision":
                    prec, _, _, _ = precision_recall_fscore_support(
                        window_labels, window_pred_binary, average="binary", zero_division=0
                    )
                    metrics[metric_key] = prec
                
                elif metric_key == "recall":
                    _, rec, _, _ = precision_recall_fscore_support(
                        window_labels, window_pred_binary, average="binary", zero_division=0
                    )
                    metrics[metric_key] = rec
                
                else:
                    logger.warning(f"Unknown metric: {metric_key}")
                    metrics[metric_key] = None
            
            except Exception as e:
                logger.warning(f"  Error computing {metric_key} for window [{start:.1f}, {end:.1f}]h: {e}")
                metrics[metric_key] = None
        
        window_results[center] = metrics
        
        # Log results with class distribution
        metrics_str = ", ".join([f"{k}={v:.4f}" if v is not None else f"{k}=N/A" 
                                 for k, v in metrics.items()])
        logger.info(f"  Window [{start:.1f}, {end:.1f}]h: {n_samples} samples "
                   f"({n_infected} infected, {n_uninfected} uninfected), {metrics_str}")
    
    return window_results


def plot_single_metric(
    window_centers: List[float],
    means: List[float],
    stds: List[float],
    window_size: float,
    metric: str,
    output_path: Path,
    model_info: str = "Final Model",
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
        label=f"{model_info} (window={window_size}h)",
    )
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"{model_info} Temporal Generalization: {metric.upper()}\n"
        f"(Tested on sliding windows [x, x+{window_size}h])",
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
    model_info: str = "Final Model",
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
        f"{model_info} Temporal Generalization: Multiple Metrics\n"
        f"(Tested on sliding windows [x, x+{window_size}h])",
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
    
    # Validate checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print("\nExpected format:")
        print("  outputs/interval_sweep_analysis/20251212-145928/checkpoints/train-test_interval_1-46")
        print("  (containing fold_01_best.pth, fold_02_best.pth, ...)")
        return
    
    # Count available checkpoints
    checkpoint_files = list(checkpoint_dir.glob("fold_*_best.pth"))
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        print("Expected files like: fold_01_best.pth, fold_02_best.pth, ...")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint files in {checkpoint_dir}")
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create output dir next to checkpoint dir
        model_name = checkpoint_dir.name
        output_dir = checkpoint_dir.parent.parent / f"{model_name}_sliding_window_fast_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        name="final_model_sliding_window_fast",
        log_dir=output_dir
    )
    
    logger.info("="*60)
    logger.info("Final Model Sliding Window Evaluation (OPTIMIZED)")
    logger.info("="*60)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Number of folds: {len(checkpoint_files)}")
    logger.info("Optimization: Evaluate once per fold, group by window")
    
    # Load configuration
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    
    # Override k_folds if provided, otherwise use number of checkpoint files
    if args.k_folds is not None:
        k_folds = args.k_folds
    else:
        k_folds = len(checkpoint_files)
    
    if k_folds != len(checkpoint_files):
        logger.warning(
            f"Mismatch: k_folds={k_folds} but found {len(checkpoint_files)} checkpoints. "
            f"Using k_folds={len(checkpoint_files)}"
        )
        k_folds = len(checkpoint_files)
    
    data_cfg = cfg.get("data", {})
    
    # CRITICAL: Override time filtering to get ALL timepoints!
    # We want infected samples from [0, max_hour], not filtered by baseline.yaml's [16, 30]
    if "frames" not in data_cfg:
        data_cfg["frames"] = {}
    
    # Determine the maximum hour from the data
    max_hour = args.max_hour if args.max_hour is not None else args.end_hour
    
    # Force infected samples to cover the full range
    # Use a very wide range to ensure we get all available timepoints
    data_cfg["frames"]["infected_window_hours"] = [0.0, max_hour]
    data_cfg["frames"]["uninfected_use_all"] = True
    
    # Remove any split-specific overrides that might restrict the range
    for split_key in ["train", "val", "test", "default", "eval", "evaluation"]:
        if split_key in data_cfg["frames"]:
            split_cfg = data_cfg["frames"][split_key]
            if isinstance(split_cfg, dict):
                split_cfg.pop("infected_window_hours", None)
                split_cfg.pop("uninfected_window_hours", None)
    
    logger.info(f"Overriding data config to use infected_window_hours: [0.0, {max_hour}]")
    logger.info("This ensures test set contains infected samples from ALL timepoints")
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"K-folds: {k_folds}")
    logger.info(f"Time field: {args.time_field}")
    
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
    
    metric_keys = args.metrics
    logger.info(f"Evaluating metrics: {', '.join(metric_keys)}")
    logger.info(f"Will evaluate on {len(windows)} windows ({k_folds} folds)")
    logger.info(f"Total evaluations: {k_folds} (not {k_folds * len(windows)} - OPTIMIZED!)")
    logger.info("="*60)
    
    # Get task config
    task_cfg = get_task_config(cfg)
    model_cfg = cfg.get("model", {})
    
    # Initialize storage
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
    
    # Store per-fold results for each window
    fold_results_per_window: Dict[float, Dict[str, List[float]]] = {
        center: {metric: [] for metric in metric_keys}
        for _, _, center in windows
    }
    
    # Evaluate each fold
    for fold_idx in range(k_folds):
        fold_tag = f"fold_{fold_idx + 1:02d}of{k_folds:02d}"
        checkpoint_path = checkpoint_dir / f"fold_{fold_idx + 1:02d}_best.pth"
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping fold {fold_idx + 1}")
            continue
        
        logger.info(f"\n[Fold {fold_idx + 1}/{k_folds}] Loading checkpoint: {checkpoint_path.name}")
        
        # Build datasets (NO time filtering - get ALL data)
        train_ds, val_ds, test_ds = build_datasets(
            data_cfg,
            transforms,
            fold_index=fold_idx if k_folds > 1 else None,
            num_folds=k_folds,
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds, data_cfg
        )
        
        # Load pre-trained model
        model = build_model(model_cfg).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Select evaluation loader
        eval_loader = {"val": val_loader, "test": test_loader}[args.split]
        
        # Evaluate on ALL data once and group by window
        logger.info(f"[{fold_tag}] Evaluating on ALL {args.split} data...")
        window_metrics = evaluate_full_and_group_by_window(
            model,
            eval_loader,
            device,
            windows,
            args.time_field,
            metric_keys,
            logger,
        )
        
        # Store results for this fold
        for center, metrics in window_metrics.items():
            for metric_key, value in metrics.items():
                if value is not None and not math.isnan(value):
                    fold_results_per_window[center][metric_key].append(value)
        
        # Cleanup
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    # Aggregate across folds
    logger.info("\n" + "="*60)
    logger.info("Aggregating results across folds...")
    logger.info("="*60)
    
    for start, end, center in windows:
        logger.info(f"\nWindow [{start:.1f}, {end:.1f}]h (center={center:.1f}):")
        
        for metric_key in metric_keys:
            fold_values = fold_results_per_window[center][metric_key]
            
            # Add to results structure
            if len(results[metric_key]["window_centers"]) < len(windows):
                results[metric_key]["window_starts"].append(start)
                results[metric_key]["window_ends"].append(end)
                results[metric_key]["window_centers"].append(center)
            
            if fold_values:
                values = np.asarray(fold_values, dtype=np.float32)
                mean_value = float(values.mean())
                std_value = float(values.std(ddof=0))
                
                results[metric_key]["fold_metrics"].append(fold_values)
                results[metric_key]["means"].append(mean_value)
                results[metric_key]["stds"].append(std_value)
                
                logger.info(f"  {metric_key.upper()}: {mean_value:.4f} ± {std_value:.4f} ({len(fold_values)} folds)")
            else:
                results[metric_key]["fold_metrics"].append([])
                results[metric_key]["means"].append(None)
                results[metric_key]["stds"].append(None)
                logger.warning(f"  {metric_key.upper()}: No valid data")
    
    # Generate plots
    model_name = checkpoint_dir.name
    stride_suffix = f"_s{stride:.0f}" if stride != window_size else ""
    base_filename = f"final_model_sliding_w{window_size:.0f}{stride_suffix}"
    
    window_centers_list = results[metric_keys[0]]["window_centers"]
    
    if len(metric_keys) > 1:
        logger.info("\nCreating combined plot for all metrics...")
        combined_plot_path = output_dir / f"{base_filename}_combined.png"
        
        metric_plot_data = {}
        for metric_key in metric_keys:
            metric_plot_data[metric_key] = (
                results[metric_key]["means"],
                results[metric_key]["stds"],
            )
        
        plot_multi_metric(window_centers_list, metric_plot_data, window_size, combined_plot_path, model_name)
    
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
            model_name,
        )
    
    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / f"{base_filename}_data.json"
    
    payload = {
        "config": args.config,
        "timestamp": timestamp,
        "checkpoint_dir": str(checkpoint_dir),
        "model_name": model_name,
        "window_size": window_size,
        "stride": stride,
        "window_starts": window_starts,
        "max_hour": args.max_hour,
        "metrics": metric_keys,
        "split": args.split,
        "k_folds": k_folds,
        "time_field": args.time_field,
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
            worst_idx = metric_result["means"].index(min(valid_means))
            logger.info(f"\n{metric_key.upper()}:")
            logger.info(f"  Best window: [{metric_result['window_starts'][best_idx]:.1f}, "
                       f"{metric_result['window_ends'][best_idx]:.1f}]h with value={max(valid_means):.4f}")
            logger.info(f"  Worst window: [{metric_result['window_starts'][worst_idx]:.1f}, "
                       f"{metric_result['window_ends'][worst_idx]:.1f}]h with value={min(valid_means):.4f}")
            logger.info(f"  Overall mean: {np.mean(valid_means):.4f} ± "
                       f"{np.std(valid_means, ddof=1 if len(valid_means) > 1 else 0):.4f}")
            logger.info(f"  Performance range: {max(valid_means) - min(valid_means):.4f} "
                       f"({(max(valid_means) - min(valid_means)) / np.mean(valid_means) * 100:.2f}% relative variation)")
        else:
            logger.warning(f"\n{metric_key.upper()}: No valid data")
    
    logger.info("="*60)
    logger.info(f"Analysis complete! Results saved to: {output_dir}")
    logger.info(f"\nOutputs:")
    logger.info(f"  - Plots: {output_dir}/*.png")
    logger.info(f"  - Data: {save_path}")
    logger.info(f"  - Log: {output_dir}/final_model_sliding_window_fast.log")
    logger.info("="*60)


if __name__ == "__main__":
    main()
