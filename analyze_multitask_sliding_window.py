"""
Multitask Model Sliding Window Evaluation

Evaluates a multitask model (classification + regression) across different 
sliding time windows to assess temporal generalization.

Similar to analyze_final_model_sliding_window_fast.py but adapted for multitask models.

For each fold:
1. Load the pre-trained multitask model checkpoint
2. Evaluate on ALL test data once
3. Group predictions by time window
4. Compute classification metrics for each window
5. Plot temporal generalization curves

This allows comparison between:
- Single-task model (classification only)
- Multitask model (classification + regression)
To see if joint learning improves temporal generalization.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from datasets import build_datasets
from models import build_multitask_model
from utils import build_transforms, load_config, set_seed, get_logger
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multitask model across sliding time windows"
    )
    parser.add_argument(
        "--config",
        default="configs/multitask_example.yaml",
        help="Path to multitask config YAML",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to checkpoint directory containing best.pt or fold_XX_best.pt files",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=6.0,
        help="Window size in hours (default: 6.0)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=3.0,
        help="Stride between windows in hours (default: 3.0)",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=0.0,
        help="Starting hour for first window",
    )
    parser.add_argument(
        "--end-hour",
        type=float,
        default=48.0,
        help="Maximum ending hour",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auc", "accuracy", "f1", "precision", "recall"],
        help="Classification metrics to compute",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--time-field",
        default="hours_since_start",
        help="Field name in metadata containing time information",
    )
    return parser.parse_args()


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


def evaluate_sliding_windows(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    windows: List[Tuple[float, float, float]],  # (start, end, center)
    time_field: str,
    metric_keys: List[str],
    logger,
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate multitask model on all data once, group by time window.
    Only uses classification outputs for metrics.
    
    Args:
        model: Trained multitask model
        dataloader: DataLoader with test data
        windows: List of (start, end, center) tuples
        time_field: Name of time field in metadata
        metric_keys: Classification metrics to compute
        
    Returns:
        Dictionary mapping window_center -> {metric: value}
    """
    model.eval()
    
    # Collect all predictions and metadata
    all_cls_probs = []
    all_labels = []
    all_times = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle tuple format: (image, label, metadata)
            if isinstance(batch, (tuple, list)):
                images = batch[0].to(device)
                labels = batch[1].to(device)
                metadata = batch[2] if len(batch) > 2 else {}
            else:
                logger.warning(f"Unknown batch format: {type(batch)}")
                continue
            
            # Extract time from metadata
            if isinstance(metadata, dict):
                if time_field in metadata:
                    times = metadata[time_field]
                elif "hours_since_start" in metadata:
                    times = metadata["hours_since_start"]
                else:
                    logger.warning(f"Time field not found in metadata, skipping batch")
                    continue
            elif isinstance(metadata, (list, tuple)):
                meta_list = meta_batch_to_list(metadata) if isinstance(metadata, dict) else metadata
                times = [m.get(time_field, m.get("hours_since_start", None)) for m in meta_list]
                if any(t is None for t in times):
                    logger.warning(f"Some samples missing time field, skipping batch")
                    continue
            else:
                logger.warning(f"Unknown metadata format: {type(metadata)}")
                continue
            
            # Forward pass - multitask model returns (cls_logits, time_pred)
            cls_logits, time_pred = model(images)
            
            # Get classification probabilities (softmax for 2-class)
            cls_probs = torch.softmax(cls_logits, dim=1)[:, 1]  # Probability of class 1
            
            # Store predictions
            all_cls_probs.extend(cls_probs.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
            
            # Handle time as tensor or list
            if isinstance(times, torch.Tensor):
                times = times.cpu().numpy().flatten().tolist()
            elif not isinstance(times, list):
                times = [times]
            
            all_times.extend(times)
    
    if len(all_cls_probs) == 0:
        logger.warning("No predictions collected!")
        return {}
    
    all_cls_probs = np.array(all_cls_probs)
    all_labels = np.array(all_labels)
    all_times = np.array(all_times)
    
    logger.info(f"  Total samples evaluated: {len(all_cls_probs)}")
    logger.info(f"  Time range: [{all_times.min():.2f}, {all_times.max():.2f}]h")
    logger.info(f"  Class distribution: {np.sum(all_labels == 0)} uninfected, {np.sum(all_labels == 1)} infected")
    
    # Group by windows and compute metrics
    window_results = {}
    
    for start, end, center in windows:
        # Find samples in this time window
        mask = (all_times >= start) & (all_times <= end)
        n_samples = mask.sum()
        
        if n_samples == 0:
            logger.warning(f"  Window [{start:.1f}, {end:.1f}]h: No samples")
            window_results[center] = {metric: None for metric in metric_keys}
            continue
        
        # Get predictions and labels for this window
        window_probs = all_cls_probs[mask]
        window_labels = all_labels[mask]
        window_pred_binary = (window_probs >= 0.5).astype(int)
        
        # Check class distribution
        unique_labels = np.unique(window_labels)
        n_infected = np.sum(window_labels == 1)
        n_uninfected = np.sum(window_labels == 0)
        
        # Compute metrics
        metrics = {}
        
        for metric_key in metric_keys:
            try:
                if metric_key == "auc":
                    if len(unique_labels) > 1:
                        metrics[metric_key] = roc_auc_score(window_labels, window_probs)
                    else:
                        metrics[metric_key] = None
                        only_class = "infected" if unique_labels[0] == 1 else "uninfected"
                        logger.info(f"  Window [{start:.1f}, {end:.1f}]h: Skipping AUC (only {only_class})")
                
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
                logger.warning(f"  Error computing {metric_key}: {e}")
                metrics[metric_key] = None
        
        window_results[center] = metrics
        
        # Log results
        metrics_str = ", ".join([f"{k}={v:.4f}" if v is not None else f"{k}=N/A" 
                                 for k, v in metrics.items()])
        logger.info(f"  Window [{start:.1f}, {end:.1f}]h: n={n_samples} "
                   f"(inf={n_infected}, uninf={n_uninfected}), {metrics_str}")
    
    return window_results


def plot_temporal_generalization(
    window_centers: List[float],
    metric_results: Dict[str, Tuple[List[float], List[float]]],  # metric -> (means, stds)
    window_size: float,
    output_path: Path,
    model_info: str = "Multitask Model",
) -> None:
    """Create combined plot for multiple classification metrics."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    any_valid = False
    for idx, (metric, (means, stds)) in enumerate(metric_results.items()):
        valid_points = [
            (center, mean, std)
            for center, mean, std in zip(window_centers, means, stds)
            if mean is not None
        ]
        
        if not valid_points:
            logging.warning(f"No valid data for metric '{metric}'")
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
            linewidth=2,
            markersize=8,
            color=colors[idx],
            label=metric.upper(),
            alpha=0.85,
        )
    
    if not any_valid:
        logging.warning("No valid data to plot")
        plt.close(fig)
        return
    
    ax.set_xlabel("Window Center (hours)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=13, fontweight='bold')
    ax.set_title(
        f"{model_info}: Temporal Generalization (Classification Performance)\n"
        f"Sliding Window Size: {window_size:.1f}h",
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.set_ylim([0, 1.05])  # Classification metrics are in [0, 1]
    
    # Add secondary x-axis for window start times
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved temporal generalization plot to {output_path}")


def main() -> None:
    args = parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Check for checkpoints
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    if not checkpoint_files:
        print(f"ERROR: No checkpoint files found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoint file(s)")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir.parent / f"sliding_window_analysis_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(
        name="multitask_sliding_window",
        log_dir=output_dir
    )
    
    logger.info("="*70)
    logger.info("Multitask Model Sliding Window Evaluation")
    logger.info("="*70)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Window size: {args.window_size}h")
    logger.info(f"Stride: {args.stride}h")
    logger.info(f"Time range: [{args.start_hour:.1f}, {args.end_hour:.1f}]h")
    
    # Load config
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Build datasets
    data_cfg = cfg.get("data", {})
    
    # Override time filtering to get full range
    if "frames" not in data_cfg:
        data_cfg["frames"] = {}
    
    data_cfg["frames"]["infected_window_hours"] = [0.0, args.end_hour]
    data_cfg["frames"]["uninfected_window_hours"] = [0.0, args.end_hour]
    data_cfg["frames"]["uninfected_use_all"] = True
    
    logger.info(f"Using full time range for evaluation: [0, {args.end_hour}]h")
    
    # Build transforms and datasets
    transform_cfg = data_cfg.get("transforms", {})
    transforms = build_transforms(transform_cfg)
    
    train_ds, val_ds, test_ds = build_datasets(data_cfg, transforms)
    
    # Create test dataloader
    batch_size = data_cfg.get("batch_size", 128) * data_cfg.get("eval_batch_size_multiplier", 2)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    logger.info(f"Test set: {len(test_ds)} samples")
    
    # Generate windows
    windows = []
    current = args.start_hour
    while current + args.window_size <= args.end_hour:
        start = current
        end = current + args.window_size
        center = current + args.window_size / 2.0
        windows.append((start, end, center))
        current += args.stride
    
    logger.info(f"Generated {len(windows)} windows")
    logger.info(f"Metrics: {', '.join(args.metrics)}")
    
    # Load model
    model_cfg = cfg.get("model", {})
    model = build_multitask_model(model_cfg).to(device)
    
    # Find best checkpoint
    best_checkpoint = checkpoint_dir / "best.pt"
    if not best_checkpoint.exists():
        # Try alternative names
        for pattern in ["fold_*_best.pt", "fold_*_best.pth", "best_model.pth", "*.pt"]:
            candidates = list(checkpoint_dir.glob(pattern))
            if candidates:
                best_checkpoint = candidates[0]
                break
    
    if not best_checkpoint.exists():
        logger.error(f"Could not find checkpoint in {checkpoint_dir}")
        return
    
    logger.info(f"Loading checkpoint: {best_checkpoint.name}")
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    # Evaluate across windows
    logger.info("\n" + "="*70)
    logger.info("Evaluating on sliding windows...")
    logger.info("="*70)
    
    window_metrics = evaluate_sliding_windows(
        model,
        test_loader,
        device,
        windows,
        args.time_field,
        args.metrics,
        logger,
    )
    
    # Aggregate results
    results = {
        metric: {
            "window_centers": [],
            "means": [],
            "stds": [],
        }
        for metric in args.metrics
    }
    
    window_centers = [center for _, _, center in windows]
    
    for metric in args.metrics:
        values = [window_metrics.get(center, {}).get(metric, None) for center in window_centers]
        
        results[metric]["window_centers"] = window_centers
        results[metric]["means"] = values
        results[metric]["stds"] = [0.0] * len(values)  # Single model, no std
    
    # Create plots
    logger.info("\n" + "="*70)
    logger.info("Generating plots...")
    logger.info("="*70)
    
    model_name = checkpoint_dir.parent.name
    plot_path = output_dir / f"multitask_temporal_generalization_w{args.window_size:.0f}h.png"
    
    metric_plot_data = {
        metric: (results[metric]["means"], results[metric]["stds"])
        for metric in args.metrics
    }
    
    plot_temporal_generalization(
        window_centers,
        metric_plot_data,
        args.window_size,
        plot_path,
        model_name,
    )
    
    # Save results
    save_path = output_dir / f"multitask_sliding_window_results.json"
    payload = {
        "config": args.config,
        "timestamp": timestamp,
        "checkpoint": str(best_checkpoint),
        "window_size": args.window_size,
        "stride": args.stride,
        "time_range": [args.start_hour, args.end_hour],
        "metrics": args.metrics,
        "results": results,
        "windows": [{"start": s, "end": e, "center": c} for s, e, c in windows],
    }
    
    with save_path.open("w") as f:
        json.dump(payload, f, indent=2)
    
    logger.info(f"Saved results to {save_path}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Summary:")
    logger.info("="*70)
    
    for metric in args.metrics:
        valid_values = [v for v in results[metric]["means"] if v is not None]
        if valid_values:
            logger.info(f"\n{metric.upper()}:")
            logger.info(f"  Mean: {np.mean(valid_values):.4f}")
            logger.info(f"  Std:  {np.std(valid_values):.4f}")
            logger.info(f"  Min:  {np.min(valid_values):.4f}")
            logger.info(f"  Max:  {np.max(valid_values):.4f}")
            logger.info(f"  Range: {np.max(valid_values) - np.min(valid_values):.4f}")
    
    logger.info("\n" + "="*70)
    logger.info(f"Analysis complete! Results in: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
