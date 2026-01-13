"""
Compare Multitask vs. Regression-Only Performance

This script loads predictions from:
  1. Multitask model (classification + regression)
  2. Regression-only models (infected-only and/or uninfected-only)

And reports regression performance (MAE/RMSE) overall and by time bins.

Usage:
    python compare_regression_performance.py \\
        --multitask outputs/multitask_resnet50/TIMESTAMP_5fold \\
        --regression-infected outputs/regression_infected/TIMESTAMP_5fold \\
        --regression-uninfected outputs/regression_uninfected/TIMESTAMP_5fold
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cv_predictions(result_dir: Path, class_filter: Optional[int] = None) -> Dict:
    """
    Load aggregated predictions across all CV folds.
    
    Args:
        result_dir: Path to CV results directory (contains fold_1/, fold_2/, ...)
        class_filter: If specified, only keep samples of this class (0 or 1)
        
    Returns:
        Dict with 'time_preds', 'time_targets', 'cls_targets'
    """
    result_dir = Path(result_dir)
    
    all_time_preds = []
    all_time_targets = []
    all_cls_targets = []
    
    # Find all fold directories
    fold_dirs = sorted([d for d in result_dir.glob("fold_*") if d.is_dir()])
    
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {result_dir}")
    
    for fold_dir in fold_dirs:
        pred_file = fold_dir / "test_predictions.npz"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found, skipping")
            continue
        
        data = np.load(pred_file)
        time_preds = data["time_preds"]
        time_targets = data["time_targets"]
        cls_targets = data["cls_targets"]
        
        # Apply class filter if specified
        if class_filter is not None:
            mask = (cls_targets == class_filter)
            time_preds = time_preds[mask]
            time_targets = time_targets[mask]
            cls_targets = cls_targets[mask]
        
        all_time_preds.append(time_preds)
        all_time_targets.append(time_targets)
        all_cls_targets.append(cls_targets)
    
    return {
        "time_preds": np.concatenate(all_time_preds),
        "time_targets": np.concatenate(all_time_targets),
        "cls_targets": np.concatenate(all_cls_targets),
        "n_folds": len(fold_dirs),
    }


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    diff = preds - targets
    return {
        "mae": np.abs(diff).mean(),
        "rmse": np.sqrt((diff ** 2).mean()),
        "mse": (diff ** 2).mean(),
        "n_samples": len(preds),
    }


def compute_metrics_by_time_bins(
    preds: np.ndarray,
    targets: np.ndarray,
    bin_edges: List[float],
) -> pd.DataFrame:
    """Compute metrics for each time bin."""
    records = []
    
    for i in range(len(bin_edges) - 1):
        start, end = bin_edges[i], bin_edges[i + 1]
        mask = (targets >= start) & (targets < end)
        
        if mask.sum() == 0:
            continue
        
        bin_preds = preds[mask]
        bin_targets = targets[mask]
        
        metrics = compute_metrics(bin_preds, bin_targets)
        metrics["time_bin"] = f"[{start:.0f}, {end:.0f})"
        metrics["bin_start"] = start
        metrics["bin_end"] = end
        
        records.append(metrics)
    
    return pd.DataFrame(records)


def plot_comparison(
    multitask_metrics: pd.DataFrame,
    regression_metrics: Optional[pd.DataFrame],
    title: str,
    output_path: Path,
):
    """Plot MAE comparison across time bins."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = (multitask_metrics["bin_start"] + multitask_metrics["bin_end"]) / 2
    
    # Plot multitask
    ax.plot(
        x,
        multitask_metrics["mae"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Multitask (Classification + Regression)",
        color="tab:blue",
    )
    
    # Plot regression-only if provided
    if regression_metrics is not None:
        x_reg = (regression_metrics["bin_start"] + regression_metrics["bin_end"]) / 2
        ax.plot(
            x_reg,
            regression_metrics["mae"],
            marker="s",
            linewidth=2.5,
            markersize=8,
            label="Regression-Only (Single Task)",
            color="tab:orange",
        )
    
    ax.set_xlabel("Time Bin Center (hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (hours)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"✓ Saved comparison plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare multitask vs regression-only")
    parser.add_argument("--multitask", type=str, required=True, help="Multitask CV results directory")
    parser.add_argument("--regression-infected", type=str, help="Regression-infected CV results directory")
    parser.add_argument("--regression-uninfected", type=str, help="Regression-uninfected CV results directory")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MULTITASK vs. REGRESSION-ONLY COMPARISON")
    print("="*80)
    
    # Time bins for analysis
    bin_edges = list(range(0, 49, 6))  # [0, 6), [6, 12), ..., [42, 48)
    
    # ========== Load Multitask ==========
    print(f"\nLoading multitask predictions from: {args.multitask}")
    multitask_all = load_cv_predictions(Path(args.multitask))
    print(f"  Total samples: {multitask_all['n_samples']}")
    
    # Split by class
    multitask_infected = {
        "time_preds": multitask_all["time_preds"][multitask_all["cls_targets"] == 1],
        "time_targets": multitask_all["time_targets"][multitask_all["cls_targets"] == 1],
    }
    multitask_uninfected = {
        "time_preds": multitask_all["time_preds"][multitask_all["cls_targets"] == 0],
        "time_targets": multitask_all["time_targets"][multitask_all["cls_targets"] == 0],
    }
    
    print(f"  Infected samples: {len(multitask_infected['time_preds'])}")
    print(f"  Uninfected samples: {len(multitask_uninfected['time_preds'])}")
    
    # Overall metrics
    multitask_overall = compute_metrics(multitask_all["time_preds"], multitask_all["time_targets"])
    multitask_inf_overall = compute_metrics(multitask_infected["time_preds"], multitask_infected["time_targets"])
    multitask_uninf_overall = compute_metrics(multitask_uninfected["time_preds"], multitask_uninfected["time_targets"])
    
    # By time bin
    multitask_inf_bins = compute_metrics_by_time_bins(
        multitask_infected["time_preds"], multitask_infected["time_targets"], bin_edges
    )
    multitask_uninf_bins = compute_metrics_by_time_bins(
        multitask_uninfected["time_preds"], multitask_uninfected["time_targets"], bin_edges
    )
    
    # ========== Load Regression-Infected ==========
    regression_inf_overall = None
    regression_inf_bins = None
    if args.regression_infected:
        print(f"\nLoading regression-infected predictions from: {args.regression_infected}")
        regression_infected = load_cv_predictions(Path(args.regression_infected), class_filter=1)
        print(f"  Infected samples: {regression_infected['n_samples']}")
        
        regression_inf_overall = compute_metrics(
            regression_infected["time_preds"], regression_infected["time_targets"]
        )
        regression_inf_bins = compute_metrics_by_time_bins(
            regression_infected["time_preds"], regression_infected["time_targets"], bin_edges
        )
    
    # ========== Load Regression-Uninfected ==========
    regression_uninf_overall = None
    regression_uninf_bins = None
    if args.regression_uninfected:
        print(f"\nLoading regression-uninfected predictions from: {args.regression_uninfected}")
        regression_uninfected = load_cv_predictions(Path(args.regression_uninfected), class_filter=0)
        print(f"  Uninfected samples: {regression_uninfected['n_samples']}")
        
        regression_uninf_overall = compute_metrics(
            regression_uninfected["time_preds"], regression_uninfected["time_targets"]
        )
        regression_uninf_bins = compute_metrics_by_time_bins(
            regression_uninfected["time_preds"], regression_uninfected["time_targets"], bin_edges
        )
    
    # ========== Print Summary ==========
    print("\n" + "="*80)
    print("OVERALL REGRESSION PERFORMANCE")
    print("="*80)
    
    print("\nMultitask Model:")
    print(f"  All samples:     MAE={multitask_overall['mae']:.3f}h, RMSE={multitask_overall['rmse']:.3f}h")
    print(f"  Infected only:   MAE={multitask_inf_overall['mae']:.3f}h, RMSE={multitask_inf_overall['rmse']:.3f}h")
    print(f"  Uninfected only: MAE={multitask_uninf_overall['mae']:.3f}h, RMSE={multitask_uninf_overall['rmse']:.3f}h")
    
    if regression_inf_overall:
        print("\nRegression-Only (Infected):")
        print(f"  MAE={regression_inf_overall['mae']:.3f}h, RMSE={regression_inf_overall['rmse']:.3f}h")
        improvement = ((regression_inf_overall['mae'] - multitask_inf_overall['mae']) / regression_inf_overall['mae']) * 100
        print(f"  → Multitask improvement: {improvement:+.1f}% MAE")
    
    if regression_uninf_overall:
        print("\nRegression-Only (Uninfected):")
        print(f"  MAE={regression_uninf_overall['mae']:.3f}h, RMSE={regression_uninf_overall['rmse']:.3f}h")
        improvement = ((regression_uninf_overall['mae'] - multitask_uninf_overall['mae']) / regression_uninf_overall['mae']) * 100
        print(f"  → Multitask improvement: {improvement:+.1f}% MAE")
    
    # ========== Save Results ==========
    results = {
        "multitask": {
            "overall": multitask_overall,
            "infected": multitask_inf_overall,
            "uninfected": multitask_uninf_overall,
        },
    }
    
    if regression_inf_overall:
        results["regression_infected"] = regression_inf_overall
    if regression_uninf_overall:
        results["regression_uninfected"] = regression_uninf_overall
    
    results_file = output_dir / "comparison_summary.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved summary to {results_file}")
    
    # ========== Plot Comparisons ==========
    if regression_inf_bins is not None:
        plot_comparison(
            multitask_metrics=multitask_inf_bins,
            regression_metrics=regression_inf_bins,
            title="Regression Performance on Infected Samples\n(Multitask vs. Regression-Only)",
            output_path=output_dir / "comparison_infected.png",
        )
    
    if regression_uninf_bins is not None:
        plot_comparison(
            multitask_metrics=multitask_uninf_bins,
            regression_metrics=regression_uninf_bins,
            title="Regression Performance on Uninfected Samples\n(Multitask vs. Regression-Only)",
            output_path=output_dir / "comparison_uninfected.png",
        )
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
