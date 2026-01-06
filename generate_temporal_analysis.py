"""
Generate Temporal Generalization (Sliding Window) Plot for Existing Results

This script loads a trained model and generates the sliding window analysis
for results that were trained before the temporal analysis feature was added.

Usage:
    python generate_temporal_analysis.py --result-dir outputs/multitask_resnet50/20260102-163144
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from datasets import build_datasets
from models import build_multitask_model
from utils import build_transforms


def evaluate_temporal_generalization(
    predictions: Dict[str, np.ndarray],
    metadata_list: List[Dict[str, Any]],
    window_size: float = 6.0,
    stride: float = 3.0,
    start_hour: float = 0.0,
    end_hour: float = 48.0,
    time_field: str = "hours_since_start",
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
    
    print(f"\nTemporal Generalization Analysis:")
    print(f"  Window size: {window_size}h, Stride: {stride}h")
    print(f"  Number of windows: {len(windows)}")
    print(f"  Time range: [{times.min():.1f}, {times.max():.1f}]h")
    
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
            print(f"  Window [{start:.1f}, {end:.1f}]h: No samples, skipping")
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
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(
                f"  Window [{start:.1f}, {end:.1f}]h: n={n_samples} "
                f"(inf={n_infected}, uninf={n_uninfected}), "
                f"AUC={auc_str}, Acc={acc:.4f}, F1={f1:.4f}"
            )
        
        except Exception as e:
            print(f"  Window [{start:.1f}, {end:.1f}]h: Error computing metrics: {e}")
    
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
            marker=markers[idx],
            color=colors[idx],
            label=metric_labels[metric],
            linewidth=2.5,
            markersize=8,
            alpha=0.85,
        )
    
    ax.set_xlabel("Time Window Center (hours)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=14, fontweight='bold')
    ax.set_title(
        f"{model_name} - Temporal Generalization\n"
        f"(Window Size: {window_size}h, Sliding Analysis)",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Auto-scale y-axis starting from minimum value (with small padding)
    all_values = [v for values in metrics_by_window.values() for v in values if v is not None]
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
    plt.close()
    
    print(f"\n✓ Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate temporal generalization analysis for existing results")
    parser.add_argument("--result-dir", type=str, required=True, help="Path to result directory")
    parser.add_argument("--window-size", type=float, default=6.0, help="Window size in hours (default: 6.0)")
    parser.add_argument("--stride", type=float, default=3.0, help="Stride in hours (default: 3.0)")
    parser.add_argument("--start-hour", type=float, default=0.0, help="Start hour (default: 0.0)")
    parser.add_argument("--end-hour", type=float, default=48.0, help="End hour (default: 48.0)")
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"ERROR: Result directory not found: {result_dir}")
        return 1
    
    # Load results.json
    results_file = result_dir / "results.json"
    if not results_file.exists():
        print(f"ERROR: results.json not found in {result_dir}")
        return 1
    
    with open(results_file) as f:
        results = json.load(f)
    
    cfg = results["config"]
    
    # Load test predictions
    predictions_file = result_dir / "test_predictions.npz"
    if not predictions_file.exists():
        print(f"ERROR: test_predictions.npz not found in {result_dir}")
        print("This file is generated during training. You may need to rerun training.")
        return 1
    
    preds = np.load(predictions_file)
    predictions = {
        "cls_preds": preds["cls_preds"],
        "cls_targets": preds["cls_targets"],
    }
    
    print(f"Loaded predictions:")
    print(f"  Classification predictions: {predictions['cls_preds'].shape}")
    print(f"  Classification targets: {predictions['cls_targets'].shape}")
    
    # Build dataset to get metadata
    print("\nBuilding datasets to extract metadata...")
    
    # Build transforms
    data_cfg = cfg["data"]
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=None,
        num_folds=1,
    )
    
    print(f"Test dataset size: {len(test_ds)}")
    print(f"Predictions size: {len(predictions['cls_preds'])}")
    
    if len(test_ds) != len(predictions['cls_preds']):
        print(f"WARNING: Dataset size mismatch! This may cause issues.")
    
    # Collect metadata
    print("\nCollecting metadata from test dataset...")
    test_metadata = []
    for i in tqdm(range(len(test_ds)), desc="Extracting metadata"):
        meta = test_ds.get_metadata(i)
        test_metadata.append(meta)
    
    # Run temporal analysis
    window_centers, metrics_by_window = evaluate_temporal_generalization(
        predictions=predictions,
        metadata_list=test_metadata,
        window_size=args.window_size,
        stride=args.stride,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    
    # Plot
    output_path = result_dir / "temporal_generalization.png"
    experiment_name = results.get("experiment_name", "Multitask Model")
    
    plot_temporal_generalization(
        window_centers=window_centers,
        metrics_by_window=metrics_by_window,
        window_size=args.window_size,
        output_path=output_path,
        model_name=experiment_name,
    )
    
    # Save metrics
    temporal_metrics = {
        "window_size_hours": args.window_size,
        "stride_hours": args.stride,
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "window_centers": window_centers,
        "metrics_by_window": metrics_by_window,
    }
    
    metrics_file = result_dir / "temporal_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(temporal_metrics, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_file}")
    print("\nDone!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
