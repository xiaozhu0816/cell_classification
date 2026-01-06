"""
Analyze Temporal Generalization for Cross-Validation Results

This script generates temporal generalization plots aggregated across all 5 folds.
It computes mean ± std for each metric at each time window.

Usage:
    python analyze_multitask_cv_temporal.py --result-dir outputs/multitask_cv/TIMESTAMP_5fold
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from models import build_multitask_model
from utils import build_transforms, load_config


def evaluate_temporal_generalization_fold(
    model,
    test_loader,
    device: torch.device,
    window_size: float = 6.0,
    stride: float = 3.0,
    max_time: float = 48.0,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate temporal generalization for a single fold."""
    model.eval()
    
    # Collect all predictions
    all_cls_logits = []
    all_time_preds = []
    all_hours = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, meta in tqdm(test_loader, desc="Collecting predictions", leave=False):
            images = images.to(device, non_blocking=True)
            cls_logits, time_pred = model(images)
            
            all_cls_logits.append(cls_logits.cpu().numpy())
            all_time_preds.append(time_pred.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Extract hours from metadata
            if isinstance(meta, dict):
                hours = meta.get("hours_since_start")
                if isinstance(hours, torch.Tensor):
                    hours = hours.numpy()
                all_hours.append(hours)
            elif isinstance(meta, list):
                hours = np.array([m.get("hours_since_start", 0.0) for m in meta])
                all_hours.append(hours)
    
    # Concatenate
    cls_logits = np.concatenate(all_cls_logits, axis=0)
    time_preds = np.concatenate(all_time_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    hours = np.concatenate(all_hours, axis=0)
    
    # Convert logits to probabilities
    from scipy.special import softmax
    cls_probs = softmax(cls_logits, axis=1)[:, 1]
    
    # Sliding window analysis
    window_centers = []
    metrics_by_window = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    
    start = 0.0
    while start + window_size <= max_time:
        end = start + window_size
        center = (start + end) / 2.0
        
        # Get samples in this window
        mask = (hours >= start) & (hours < end)
        n_samples = mask.sum()
        
        if n_samples < 10:  # Skip windows with too few samples
            start += stride
            continue
        
        window_probs = cls_probs[mask]
        window_labels = labels[mask]
        window_pred_binary = (window_probs >= 0.5).astype(int)
        
        unique_labels = np.unique(window_labels)
        
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
        
        except Exception as e:
            print(f"  Warning: Error in window [{start:.1f}, {end:.1f}]h: {e}")
        
        start += stride
    
    return window_centers, metrics_by_window


def aggregate_cv_temporal_results(
    fold_results: List[Tuple[List[float], Dict[str, List[float]]]],
) -> Tuple[List[float], Dict[str, Dict[str, List[float]]]]:
    """Aggregate temporal results across folds (mean ± std)."""
    
    # Assume all folds have same window centers
    window_centers = fold_results[0][0]
    n_windows = len(window_centers)
    
    # Collect all metrics across folds
    aggregated = {}
    
    metric_names = fold_results[0][1].keys()
    
    for metric in metric_names:
        values_per_window = [[] for _ in range(n_windows)]
        
        for fold_centers, fold_metrics in fold_results:
            for i, value in enumerate(fold_metrics[metric]):
                if value is not None:
                    values_per_window[i].append(value)
        
        # Compute mean and std
        means = []
        stds = []
        
        for values in values_per_window:
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(None)
                stds.append(None)
        
        aggregated[metric] = {
            "mean": means,
            "std": stds,
        }
    
    return window_centers, aggregated


def plot_cv_temporal_generalization(
    window_centers: List[float],
    aggregated_metrics: Dict[str, Dict[str, List[float]]],
    window_size: float,
    output_path: Path,
    num_folds: int = 5,
) -> None:
    """Create temporal generalization plot with mean ± std across folds."""
    
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
    
    for idx, (metric, stats) in enumerate(aggregated_metrics.items()):
        means = stats["mean"]
        stds = stats["std"]
        
        # Filter out None values
        valid_points = [
            (c, m, s) for c, m, s in zip(window_centers, means, stds)
            if m is not None and s is not None
        ]
        
        if not valid_points:
            continue
        
        centers, mean_vals, std_vals = zip(*valid_points)
        centers = np.array(centers)
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        
        # Plot mean line
        ax.plot(
            centers,
            mean_vals,
            marker=markers[idx],
            color=colors[idx],
            label=metric_labels[metric],
            linewidth=2.5,
            markersize=8,
            alpha=0.85,
        )
        
        # Add ± std shaded region
        ax.fill_between(
            centers,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=colors[idx],
            alpha=0.2,
        )
    
    ax.set_xlabel("Time Window Center (hours)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=14, fontweight='bold')
    ax.set_title(
        f"Multitask Model - Temporal Generalization ({num_folds}-Fold CV)\n"
        f"Mean ± Std (Window Size: {window_size}h, Sliding Analysis)",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Auto-scale y-axis starting from minimum value (with small padding)
    all_values = []
    for stats in aggregated_metrics.values():
        all_values.extend([v for v in stats["mean"] if v is not None])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else 0.05
        ax.set_ylim(max(0, y_min - padding), min(1.05, y_max + padding))
    else:
        ax.set_ylim(0.0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved CV temporal generalization plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze temporal generalization for CV results")
    parser.add_argument("--result-dir", type=str, required=True, help="Path to CV results directory")
    parser.add_argument("--window-size", type=float, default=6.0, help="Time window size in hours")
    parser.add_argument("--stride", type=float, default=3.0, help="Stride between windows in hours")
    parser.add_argument("--max-time", type=float, default=48.0, help="Maximum time in hours")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return 1
    
    print("="*80)
    print("MULTITASK CV - TEMPORAL GENERALIZATION ANALYSIS")
    print("="*80)
    print(f"Result directory: {result_dir}")
    
    # Find all fold directories
    fold_dirs = sorted(result_dir.glob("fold_*"))
    if not fold_dirs:
        print(f"Error: No fold directories found in {result_dir}")
        return 1
    
    num_folds = len(fold_dirs)
    print(f"Found {num_folds} folds")
    
    # Load config from first fold
    fold_1_results = fold_dirs[0] / "results.json"
    if not fold_1_results.exists():
        print(f"Error: results.json not found in {fold_dirs[0]}")
        return 1
    
    # Find config file (should be in parent directory or fold directory)
    config_file = None
    for possible_config in [result_dir / "config.yaml", fold_dirs[0] / "config.yaml"]:
        if possible_config.exists():
            config_file = possible_config
            break
    
    if config_file is None:
        # Try to find config from checkpoint
        checkpoint_file = fold_dirs[0] / "checkpoints" / "best.pt"
        if checkpoint_file.exists():
            print(f"Loading config from checkpoint: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            cfg = checkpoint.get("config")
            if cfg is None:
                print("Error: Could not find config in checkpoint or result directory")
                return 1
        else:
            print("Error: Could not find config file")
            return 1
    else:
        print(f"Loading config from: {config_file}")
        cfg = load_config(str(config_file))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build transforms and datasets
    data_cfg = cfg.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    # Process each fold
    fold_results = []
    
    for fold_idx, fold_dir in enumerate(fold_dirs):
        print(f"\nProcessing fold {fold_idx + 1}/{num_folds}...")
        
        # Load checkpoint
        checkpoint_file = fold_dir / "checkpoints" / "best.pt"
        if not checkpoint_file.exists():
            print(f"  Warning: Checkpoint not found: {checkpoint_file}, skipping fold")
            continue
        
        print(f"  Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        
        # Build model
        model_cfg = cfg.get("model", {})
        model = build_multitask_model(model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        
        # Build test dataset for this fold
        print(f"  Building test dataset...")
        _, _, test_ds = build_datasets(
            data_cfg=data_cfg,
            transforms=transforms_dict,
            fold_index=fold_idx,
            num_folds=num_folds,
        )
        
        print(f"  Test set size: {len(test_ds)}")
        
        # Create test loader
        batch_size = data_cfg.get("batch_size", 32)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
        )
        
        # Evaluate temporal generalization
        print(f"  Evaluating temporal generalization...")
        window_centers, metrics_by_window = evaluate_temporal_generalization_fold(
            model=model,
            test_loader=test_loader,
            device=device,
            window_size=args.window_size,
            stride=args.stride,
            max_time=args.max_time,
        )
        
        fold_results.append((window_centers, metrics_by_window))
        
        # Save individual fold results
        fold_temporal_file = fold_dir / "temporal_metrics.json"
        with open(fold_temporal_file, "w") as f:
            json.dump({
                "window_centers": window_centers,
                "metrics": {k: v for k, v in metrics_by_window.items()},
                "window_size": args.window_size,
                "stride": args.stride,
            }, f, indent=2)
        print(f"  ✓ Saved fold temporal metrics to {fold_temporal_file}")
    
    if not fold_results:
        print("\nError: No fold results collected")
        return 1
    
    # Aggregate across folds
    print(f"\nAggregating results across {len(fold_results)} folds...")
    window_centers, aggregated_metrics = aggregate_cv_temporal_results(fold_results)
    
    # Save aggregated results
    cv_temporal_file = result_dir / "cv_temporal_metrics.json"
    with open(cv_temporal_file, "w") as f:
        json.dump({
            "num_folds": len(fold_results),
            "window_centers": window_centers,
            "aggregated_metrics": {
                metric: {
                    "mean": [float(v) if v is not None else None for v in stats["mean"]],
                    "std": [float(v) if v is not None else None for v in stats["std"]],
                }
                for metric, stats in aggregated_metrics.items()
            },
            "window_size": args.window_size,
            "stride": args.stride,
        }, f, indent=2)
    print(f"✓ Saved CV temporal metrics to {cv_temporal_file}")
    
    # Plot aggregated results
    print("\nGenerating CV temporal generalization plot...")
    plot_path = result_dir / "cv_temporal_generalization.png"
    plot_cv_temporal_generalization(
        window_centers=window_centers,
        aggregated_metrics=aggregated_metrics,
        window_size=args.window_size,
        output_path=plot_path,
        num_folds=len(fold_results),
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for metric, stats in aggregated_metrics.items():
        means = [v for v in stats["mean"] if v is not None]
        if means:
            overall_mean = np.mean(means)
            overall_std = np.mean([v for v in stats["std"] if v is not None])
            print(f"{metric.upper():15s}: {overall_mean:.4f} ± {overall_std:.4f} (avg across windows)")
    
    print("\n✓ CV temporal generalization analysis complete!")
    print(f"✓ Results saved to {result_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
