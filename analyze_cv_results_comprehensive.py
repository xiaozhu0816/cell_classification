"""
Run All Analysis Scripts on 5-Fold CV Results

This script runs the same comprehensive analysis that was done for the single-run
(20260102-163144) on the aggregated 5-fold CV results.

Generates:
1. prediction_scatter.png - Regression predictions vs ground truth
2. error_analysis_by_time.png - Error distribution across time ranges
3. error_vs_classification_confidence.png - Confidence vs error relationship
4. valley_period_analysis.png - Valley (13-19h) specific analysis
5. worst_predictions_report.txt - Top misclassifications and errors

Usage:
    python analyze_cv_results_comprehensive.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score


def load_all_fold_predictions(cv_dir: Path):
    """Load and aggregate predictions from all folds."""
    all_data = {
        "cls_probs": [],
        "cls_preds": [],
        "cls_labels": [],
        "reg_preds": [],
        "reg_labels": [],
        "image_paths": [],
        "frame_index": [],
        "hours_since_start": [],
        "position": [],
        "fold_id": [],
    }
    
    for fold_idx in range(1, 6):
        fold_dir = cv_dir / f"fold_{fold_idx}"
        pred_file = fold_dir / "test_predictions.npz"
        
        if not pred_file.exists():
            print(f"⚠ Warning: {pred_file} not found, skipping fold {fold_idx}")
            continue
        
        data = np.load(pred_file)
        # Keys from CV export: cls_preds (probability), cls_targets (0/1), time_preds, time_targets
        all_data["cls_probs"].append(data["cls_preds"])  # alias used downstream
        all_data["cls_preds"].append(data["cls_preds"])
        all_data["cls_labels"].append(data["cls_targets"])
        all_data["reg_preds"].append(data["time_preds"])
        all_data["reg_labels"].append(data["time_targets"])

        meta_file = fold_dir / "test_metadata.jsonl"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = json.loads(line)
                    all_data["image_paths"].append(m.get("path", ""))
                    all_data["frame_index"].append(m.get("frame_index", -1))
                    all_data["hours_since_start"].append(m.get("hours_since_start", float("nan")))
                    all_data["position"].append(m.get("position", ""))
                    all_data["fold_id"].append(fold_idx)
        else:
            # Keep lengths aligned (fallback placeholders)
            fold_n = int(data["cls_targets"].shape[0])
            all_data["image_paths"].extend([""] * fold_n)
            all_data["frame_index"].extend([-1] * fold_n)
            all_data["hours_since_start"].extend([float("nan")] * fold_n)
            all_data["position"].extend([""] * fold_n)
            all_data["fold_id"].extend([fold_idx] * fold_n)
    
    # Concatenate all arrays
    for key in ["cls_probs", "cls_preds", "cls_labels", "reg_preds", "reg_labels"]:
        if all_data[key]:
            all_data[key] = np.concatenate(all_data[key])
        else:
            all_data[key] = np.array([])

    # Convert metadata lists to numpy arrays for convenience
    all_data["frame_index"] = np.asarray(all_data["frame_index"], dtype=np.int64)
    all_data["hours_since_start"] = np.asarray(all_data["hours_since_start"], dtype=np.float32)
    all_data["fold_id"] = np.asarray(all_data["fold_id"], dtype=np.int64)
    
    return all_data


def plot_prediction_scatter(data, output_dir: Path):
    """Generate prediction scatter plots (All + split by infected/uninfected)."""

    reg_preds = data["reg_preds"]
    reg_labels = data["reg_labels"]
    cls_labels = data["cls_labels"]

    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0

    def _safe_stats(y_true: np.ndarray, y_pred: np.ndarray):
        if len(y_true) == 0:
            return float("nan"), float("nan"), float("nan")
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        denom = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = float("nan") if denom == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denom)
        return mae, rmse, r2

    # 3 panels: All / Infected / Uninfected
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    panels = [
        ("All", np.ones_like(cls_labels, dtype=bool)),
        ("Infected", infected_mask),
        ("Uninfected", uninfected_mask),
    ]

    # Global min/max for consistent axes
    min_val = float(min(np.min(reg_labels), np.min(reg_preds)))
    max_val = float(max(np.max(reg_labels), np.max(reg_preds)))

    for ax, (name, mask) in zip(axes, panels):
        y_true = reg_labels[mask]
        y_pred = reg_preds[mask]
        mae, rmse, r2 = _safe_stats(y_true, y_pred)

        if name == "All":
            ax.scatter(
                reg_labels[infected_mask],
                reg_preds[infected_mask],
                alpha=0.35,
                s=18,
                c="red",
                label=f"Infected (n={infected_mask.sum()})",
                edgecolors="none",
            )
            ax.scatter(
                reg_labels[uninfected_mask],
                reg_preds[uninfected_mask],
                alpha=0.35,
                s=18,
                c="blue",
                label=f"Uninfected (n={uninfected_mask.sum()})",
                edgecolors="none",
            )
            ax.legend(loc="upper left", fontsize=10)
        else:
            color = "red" if name == "Infected" else "blue"
            ax.scatter(y_true, y_pred, alpha=0.45, s=18, c=color, edgecolors="none")

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=2, alpha=0.6)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("True Time (hours)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Time (hours)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"{name}\nMAE={mae:.2f}h, RMSE={rmse:.2f}h, R²={r2:.4f}",
            fontsize=13,
            fontweight="bold",
        )

    fig.suptitle("Time Prediction Scatter: 5-Fold CV Aggregated", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = output_dir / "prediction_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_error_analysis_by_time(data, output_dir: Path):
    """Error analysis across time ranges (like analyze_regression_errors.py)."""
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_labels = data['cls_labels']
    
    errors = np.abs(reg_preds - reg_labels)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error histogram
    ax = axes[0, 0]
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(errors):.2f}h')
    ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
              label=f'Median: {np.median(errors):.2f}h')
    ax.set_xlabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution (All Test Data)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Error by class
    ax = axes[0, 1]
    infected_errors = errors[cls_labels == 1]
    uninfected_errors = errors[cls_labels == 0]
    
    bp = ax.boxplot([infected_errors, uninfected_errors],
                     tick_labels=['Infected', 'Uninfected'],
                     patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title(f'Error by Cell Type\nInfected: {np.mean(infected_errors):.2f}h, Uninfected: {np.mean(uninfected_errors):.2f}h',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Error vs time (scatter)
    ax = axes[1, 0]
    ax.scatter(reg_labels, errors, alpha=0.3, s=15, c='steelblue', edgecolors='none')
    
    # Smoothed trend
    from scipy.ndimage import gaussian_filter1d
    sorted_idx = np.argsort(reg_labels)
    window_size = max(1, len(reg_labels) // 30)
    smoothed = gaussian_filter1d(errors[sorted_idx], sigma=window_size)
    ax.plot(reg_labels[sorted_idx], smoothed, 'r-', linewidth=3, alpha=0.8, label='Trend')
    
    ax.set_xlabel('True Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Time (Temporal Pattern)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Error percentiles by time bins
    ax = axes[1, 1]
    time_bins = np.arange(0, 49, 6)
    bin_centers = time_bins[:-1] + 3
    
    percentiles = []
    for i in range(len(time_bins) - 1):
        mask = (reg_labels >= time_bins[i]) & (reg_labels < time_bins[i+1])
        if mask.sum() > 0:
            percentiles.append({
                '25': np.percentile(errors[mask], 25),
                '50': np.percentile(errors[mask], 50),
                '75': np.percentile(errors[mask], 75),
                'mean': np.mean(errors[mask])
            })
        else:
            percentiles.append({'25': 0, '50': 0, '75': 0, 'mean': 0})
    
    p25 = [p['25'] for p in percentiles]
    p50 = [p['50'] for p in percentiles]
    p75 = [p['75'] for p in percentiles]
    pmean = [p['mean'] for p in percentiles]
    
    ax.fill_between(bin_centers, p25, p75, alpha=0.3, color='steelblue', label='25-75 percentile')
    ax.plot(bin_centers, p50, 'o-', color='darkblue', linewidth=2, markersize=8, label='Median')
    ax.plot(bin_centers, pmean, 's--', color='red', linewidth=2, markersize=6, label='Mean', alpha=0.7)
    
    # Highlight valley
    ax.axvspan(13, 19, alpha=0.15, color='red', label='Valley (13-19h)')
    
    ax.set_xlabel('Time Window Center (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error Statistics by Time Period', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "error_analysis_by_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_error_vs_confidence(data, output_dir: Path):
    """Error vs classification confidence (like analyze_regression_errors.py)."""
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_probs = data['cls_probs']
    cls_labels = data['cls_labels']
    
    errors = np.abs(reg_preds - reg_labels)
    
    # Get confidence in the predicted class.
    # In our pipeline cls_probs is typically a 1D array of p(infected).
    cls_probs = np.asarray(cls_probs)
    if cls_probs.ndim == 1:
        p_inf = cls_probs.astype(np.float32)
        pred = (p_inf >= 0.5).astype(int)
        confidence = np.where(pred == 1, p_inf, 1.0 - p_inf)
    else:
        # If Nx2 provided, take max probability as confidence
        confidence = np.max(cls_probs, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Scatter: Confidence vs Error
    ax = axes[0]
    ax.scatter(confidence, errors, alpha=0.3, s=20, c='steelblue', edgecolors='none')
    
    # Bin and show trend
    bins = np.linspace(confidence.min(), 1.0, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_errors = []
    for i in range(len(bins) - 1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() > 0:
            binned_errors.append(np.mean(errors[mask]))
        else:
            binned_errors.append(np.nan)
    
    ax.plot(bin_centers, binned_errors, 'r-', linewidth=3, alpha=0.8, label='Binned Mean')
    
    # Correlation
    corr, p_val = stats.pearsonr(confidence, errors)
    
    ax.set_xlabel('Classification Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regression Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title(f'Error vs Confidence\nCorrelation: {corr:.3f} (p={p_val:.1e})',
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Confidence bins boxplot
    ax = axes[1]
    
    conf_bins = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    conf_labels = ['0.5-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.95', '0.95-1.0']
    
    binned_data = []
    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i+1])
        binned_data.append(errors[mask])
    
    bp = ax.boxplot(binned_data, tick_labels=conf_labels, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.set_xlabel('Classification Confidence Range', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regression Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Confidence Level', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    output_file = output_dir / "error_vs_classification_confidence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_valley_analysis(data, output_dir: Path):
    """Valley period (13-19h) specific analysis (like analyze_regression_by_class.py)."""
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_labels = data['cls_labels']
    
    errors = np.abs(reg_preds - reg_labels)
    
    # Define valley and non-valley
    valley_mask = (reg_labels >= 13) & (reg_labels < 19)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error by time range and class
    ax = axes[0, 0]
    
    time_ranges = [(0, 6), (6, 13), (13, 19), (19, 48)]
    range_labels = ['0-6h', '6-13h', '13-19h\n(Valley)', '19-48h']
    
    infected_data = []
    uninfected_data = []
    
    for start, end in time_ranges:
        mask = (reg_labels >= start) & (reg_labels < end)
        infected_mask = mask & (cls_labels == 1)
        uninfected_mask = mask & (cls_labels == 0)
        
        infected_data.append(errors[infected_mask] if infected_mask.sum() > 0 else [])
        uninfected_data.append(errors[uninfected_mask] if uninfected_mask.sum() > 0 else [])
    
    x = np.arange(len(range_labels))
    width = 0.35
    
    infected_means = [np.mean(d) if len(d) > 0 else 0 for d in infected_data]
    uninfected_means = [np.mean(d) if len(d) > 0 else 0 for d in uninfected_data]
    
    bars1 = ax.bar(x - width/2, infected_means, width, label='Infected',
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, uninfected_means, width, label='Uninfected',
                   color='blue', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Mean Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Range', fontsize=12, fontweight='bold')
    ax.set_title('Mean Error by Time Range and Cell Type', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(range_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Valley vs non-valley boxplot
    ax = axes[0, 1]
    
    valley_errors_inf = errors[valley_mask & (cls_labels == 1)]
    nonvalley_errors_inf = errors[~valley_mask & (cls_labels == 1)]
    valley_errors_uninf = errors[valley_mask & (cls_labels == 0)]
    nonvalley_errors_uninf = errors[~valley_mask & (cls_labels == 0)]
    
    bp = ax.boxplot([nonvalley_errors_inf, valley_errors_inf,
                     nonvalley_errors_uninf, valley_errors_uninf],
                    tick_labels=['Infected\nNon-Valley', 'Infected\nValley',
                           'Uninfected\nNon-Valley', 'Uninfected\nValley'],
                    patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    
    # Color boxes
    colors = ['lightcoral', 'darkred', 'lightblue', 'darkblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Valley vs Non-Valley Error Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
    
    # 3. Statistical tests
    ax = axes[1, 0]
    ax.axis('off')
    
    # T-tests
    from scipy.stats import ttest_ind, mannwhitneyu
    
    # Infected: valley vs non-valley
    t_stat_inf, p_val_inf = ttest_ind(valley_errors_inf, nonvalley_errors_inf)
    u_stat_inf, p_val_mw_inf = mannwhitneyu(valley_errors_inf, nonvalley_errors_inf)
    
    # Uninfected: valley vs non-valley
    t_stat_uninf, p_val_uninf = ttest_ind(valley_errors_uninf, nonvalley_errors_uninf)
    u_stat_uninf, p_val_mw_uninf = mannwhitneyu(valley_errors_uninf, nonvalley_errors_uninf)
    
    text = f"""Statistical Tests: Valley (13-19h) vs Non-Valley

    INFECTED CELLS:
    • Valley mean error: {np.mean(valley_errors_inf):.3f}h (n={len(valley_errors_inf)})
    • Non-valley mean error: {np.mean(nonvalley_errors_inf):.3f}h (n={len(nonvalley_errors_inf)})
    • T-test: t={t_stat_inf:.3f}, p={p_val_inf:.4f}
    • Mann-Whitney U: U={u_stat_inf:.1f}, p={p_val_mw_inf:.4f}
    • Significant? {'YES' if p_val_mw_inf < 0.05 else 'NO'} (α=0.05)
    
    UNINFECTED CELLS:
    • Valley mean error: {np.mean(valley_errors_uninf):.3f}h (n={len(valley_errors_uninf)})
    • Non-valley mean error: {np.mean(nonvalley_errors_uninf):.3f}h (n={len(nonvalley_errors_uninf)})
    • T-test: t={t_stat_uninf:.3f}, p={p_val_uninf:.4f}
    • Mann-Whitney U: U={u_stat_uninf:.1f}, p={p_val_mw_uninf:.4f}
    • Significant? {'YES' if p_val_mw_uninf < 0.05 else 'NO'} (α=0.05)
    
    INTERPRETATION:
    {'Valley period shows SIGNIFICANTLY higher errors' if (p_val_mw_inf < 0.05 or p_val_mw_uninf < 0.05) else 'Valley period does NOT show significantly different errors'}
    {'(especially for uninfected cells)' if p_val_mw_uninf < p_val_mw_inf else ''}
    """
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 4. Error distribution histograms
    ax = axes[1, 1]
    
    bins = np.linspace(0, max(errors.max(), 10), 30)
    
    ax.hist(valley_errors_uninf, bins=bins, alpha=0.5, color='darkblue',
           label=f'Uninfected Valley (n={len(valley_errors_uninf)})', edgecolor='black')
    ax.hist(nonvalley_errors_uninf, bins=bins, alpha=0.5, color='lightblue',
           label=f'Uninfected Non-Valley (n={len(nonvalley_errors_uninf)})', edgecolor='black')
    
    ax.axvline(np.median(valley_errors_uninf), color='darkblue', linestyle='--', linewidth=2,
              label=f'Valley median: {np.median(valley_errors_uninf):.2f}h')
    ax.axvline(np.median(nonvalley_errors_uninf), color='lightblue', linestyle='--', linewidth=2,
              label=f'Non-valley median: {np.median(nonvalley_errors_uninf):.2f}h')
    
    ax.set_xlabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Uninfected Cells: Error Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "valley_period_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_classification_by_time_window(
    data,
    output_dir: Path,
    window_size: float = 6.0,
    stride: float = 3.0,
    start_hour: float = 0.0,
    end_hour: float = 48.0,
):
    """Classification performance vs time using sliding windows.

    Uses hours_since_start from per-sample metadata and cls_probs (p(infected)).
    Plots Accuracy, F1, and AUC over window centers.
    """
    hours = np.asarray(data.get("hours_since_start", []), dtype=np.float32)
    probs = np.asarray(data.get("cls_probs", []), dtype=np.float32)
    y = np.asarray(data.get("cls_labels", []), dtype=np.int64)
    fold_id = np.asarray(data.get("fold_id", []), dtype=np.int64)

    if len(hours) == 0 or len(probs) == 0 or len(y) == 0 or len(fold_id) == 0:
        print("⚠ Skipping temporal classification chart (missing hours/cls_probs/labels)")
        return

    # Guard against length mismatch (shouldn't happen, but keep robust)
    n = min(len(hours), len(probs), len(y), len(fold_id))
    hours = hours[:n]
    probs = probs[:n]
    y = y[:n]
    fold_id = fold_id[:n]

    # Compute per-fold metrics per window, then aggregate mean/std across folds.
    fold_ids = sorted(set(int(x) for x in fold_id.tolist()))
    if not fold_ids:
        print("⚠ Skipping temporal classification chart (no fold ids)")
        return

    centers: List[float] = []
    per_fold_acc: List[List[float]] = []
    per_fold_f1: List[List[float]] = []
    per_fold_auc: List[List[float]] = []

    # Build window centers once
    current = start_hour
    while current + window_size <= end_hour + 1e-6:
        centers.append((current + current + window_size) / 2.0)
        current += stride
    centers_arr = np.asarray(centers, dtype=np.float32)

    def _metrics_for_mask(mask: np.ndarray) -> Tuple[float, float, float]:
        count = int(mask.sum())
        if count == 0:
            return np.nan, np.nan, np.nan
        y_win = y[mask]
        p_win = probs[mask]
        pred_win = (p_win >= 0.5).astype(int)
        acc = float((pred_win == y_win).mean())
        tp = float(np.logical_and(pred_win == 1, y_win == 1).sum())
        fp = float(np.logical_and(pred_win == 1, y_win == 0).sum())
        fn = float(np.logical_and(pred_win == 0, y_win == 1).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        try:
            auc = np.nan if len(np.unique(y_win)) < 2 else float(roc_auc_score(y_win, p_win))
        except Exception:
            auc = np.nan
        return acc, float(f1), auc

    for fid in fold_ids:
        acc_list: List[float] = []
        f1_list: List[float] = []
        auc_list: List[float] = []
        fold_mask = fold_id == fid
        for center in centers_arr:
            win_start = float(center - window_size / 2.0)
            win_end = float(center + window_size / 2.0)
            mask = fold_mask & (hours >= win_start) & (hours < win_end)
            a, f1v, aucv = _metrics_for_mask(mask)
            acc_list.append(a)
            f1_list.append(f1v)
            auc_list.append(aucv)
        per_fold_acc.append(acc_list)
        per_fold_f1.append(f1_list)
        per_fold_auc.append(auc_list)

    acc_mat = np.asarray(per_fold_acc, dtype=np.float32)
    f1_mat = np.asarray(per_fold_f1, dtype=np.float32)
    auc_mat = np.asarray(per_fold_auc, dtype=np.float32)

    acc_mean = np.nanmean(acc_mat, axis=0)
    acc_std = np.nanstd(acc_mat, axis=0)
    f1_mean = np.nanmean(f1_mat, axis=0)
    f1_std = np.nanstd(f1_mat, axis=0)
    auc_mean = np.nanmean(auc_mat, axis=0)
    auc_std = np.nanstd(auc_mat, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    # mean curves
    ax.plot(centers_arr, acc_mean, "o-", label="Accuracy", linewidth=2)
    ax.plot(centers_arr, f1_mean, "s-", label="F1 (infected)", linewidth=2)
    ax.plot(centers_arr, auc_mean, "^-", label="AUC", linewidth=2)
    # std bands
    ax.fill_between(centers_arr, acc_mean - acc_std, acc_mean + acc_std, alpha=0.15)
    ax.fill_between(centers_arr, f1_mean - f1_std, f1_mean + f1_std, alpha=0.15)
    ax.fill_between(centers_arr, auc_mean - auc_std, auc_mean + auc_std, alpha=0.15)
    ax.set_xlabel("Window center (hours)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    # Auto-zoom Y-axis to make differences easier to see.
    vals = np.concatenate(
        [
            acc_mean - acc_std,
            acc_mean + acc_std,
            f1_mean - f1_std,
            f1_mean + f1_std,
            auc_mean - auc_std,
            auc_mean + auc_std,
        ]
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) > 0:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        pad = max(0.02, 0.10 * (hi - lo))
        ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))
    else:
        ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title(
        f"Classification vs Time (sliding window)\nwindow={window_size:.1f}h stride={stride:.1f}h",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_file = output_dir / "classification_by_time_window.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_file}")
    plt.close()


def generate_worst_predictions_report(data, output_dir: Path):
    """Generate report of worst predictions (like analyze_regression_errors.py)."""
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_preds = data['cls_preds']
    cls_labels = data['cls_labels']
    cls_probs = data['cls_probs']
    image_paths = data['image_paths']
    
    errors = np.abs(reg_preds - reg_labels)
    
    # Sort by error
    sorted_idx = np.argsort(errors)[::-1]
    
    report = []
    report.append("="*80)
    report.append("WORST PREDICTIONS REPORT - 5-FOLD CV AGGREGATED")
    report.append("="*80)
    report.append("")
    
    # Top 20 worst regression errors
    report.append("TOP 20 WORST REGRESSION ERRORS:")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'True Time':<12} {'Pred Time':<12} {'Error':<10} {'Class':<12} {'Confidence':<12} {'Image Path'}")
    report.append("-"*80)
    
    for rank, idx in enumerate(sorted_idx[:20], 1):
        true_time = reg_labels[idx]
        pred_time = reg_preds[idx]
        error = errors[idx]
        true_class = 'Infected' if cls_labels[idx] == 1 else 'Uninfected'
        confidence = np.max(cls_probs[idx])
        image_path = image_paths[idx] if idx < len(image_paths) else 'N/A'
        
        report.append(f"{rank:<6} {true_time:<12.2f} {pred_time:<12.2f} {error:<10.2f} "
                     f"{true_class:<12} {confidence:<12.4f} {image_path}")
    
    report.append("")
    report.append("="*80)
    
    # Misclassifications
    misclass_mask = cls_preds != cls_labels
    num_misclass = misclass_mask.sum()
    
    report.append(f"CLASSIFICATION ERRORS: {num_misclass} total misclassifications")
    report.append("-"*80)
    
    if num_misclass > 0:
        report.append(f"{'Type':<20} {'True Class':<15} {'Pred Class':<15} {'Confidence':<12} {'True Time':<12} {'Image Path'}")
        report.append("-"*80)
        
        misclass_idx = np.where(misclass_mask)[0]
        for idx in misclass_idx[:20]:  # Top 20
            true_class = 'Infected' if cls_labels[idx] == 1 else 'Uninfected'
            pred_class = 'Infected' if cls_preds[idx] == 1 else 'Uninfected'
            confidence = np.max(cls_probs[idx])
            true_time = reg_labels[idx]
            image_path = image_paths[idx] if idx < len(image_paths) else 'N/A'
            
            error_type = f"{true_class}→{pred_class}"
            report.append(f"{error_type:<20} {true_class:<15} {pred_class:<15} "
                         f"{confidence:<12.4f} {true_time:<12.2f} {image_path}")
    else:
        report.append("No misclassifications! Perfect classification performance.")
    
    report.append("")
    report.append("="*80)
    
    # Summary statistics
    report.append("SUMMARY STATISTICS:")
    report.append("-"*80)
    report.append(f"Total samples: {len(errors)}")
    report.append(f"Mean absolute error: {np.mean(errors):.3f} hours")
    report.append(f"Median absolute error: {np.median(errors):.3f} hours")
    report.append(f"Std absolute error: {np.std(errors):.3f} hours")
    report.append(f"Max error: {np.max(errors):.3f} hours")
    report.append(f"90th percentile error: {np.percentile(errors, 90):.3f} hours")
    report.append(f"95th percentile error: {np.percentile(errors, 95):.3f} hours")
    report.append("")
    report.append(f"Classification accuracy: {(cls_preds == cls_labels).mean()*100:.2f}%")
    report.append(f"Misclassifications: {num_misclass} / {len(cls_labels)}")
    report.append("="*80)
    
    output_file = output_dir / "worst_predictions_report.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved {output_file}")


def plot_error_distribution_by_time_range(data, output_dir: Path):
    """
    Generate box plots + trend lines showing error distribution across time ranges,
    separated by infected/uninfected cells (matching the first image).
    """
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_labels = data['cls_labels']
    
    errors = np.abs(reg_preds - reg_labels)
    
    # Define time ranges (bins)
    time_bins = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 48)]
    bin_labels = ['0-6h', '6-12h', '12-18h', '18-24h', '24-30h', '30-36h', '36-48h']
    
    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Top row: Box plots for infected (left) and uninfected (right)
    for ax, mask, title, color in [
        (axes[0, 0], infected_mask, 'Infected Cells: Error Distribution by Time Range', 'salmon'),
        (axes[0, 1], uninfected_mask, 'Uninfected Cells: Error Distribution by Time Range', 'lightblue')
    ]:
        binned_errors = []
        bin_means = []
        
        for start, end in time_bins:
            time_mask = (reg_labels >= start) & (reg_labels < end) & mask
            if np.sum(time_mask) > 0:
                binned_errors.append(errors[time_mask])
                bin_means.append(np.mean(errors[time_mask]))
            else:
                binned_errors.append([])
                bin_means.append(0)
        
        # Box plot
        bp = ax.boxplot(binned_errors, tick_labels=bin_labels, patch_artist=True,
                        medianprops=dict(color='darkred', linewidth=2),
                        boxprops=dict(facecolor=color, alpha=0.6, edgecolor='black'),
                        showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.3))
        
        # Add mean values as text on boxes
        for i, mean_val in enumerate(bin_means):
            if mean_val > 0:
                ax.text(i + 1, mean_val, f'{mean_val:.2f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Time Range (hours since infection/start)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # Bottom row: Error trend over continuous time (infected and uninfected)
    for ax, mask, title, color in [
        (axes[1, 0], infected_mask, 'Infected Cells: Error Trend Over Time', 'red'),
        (axes[1, 1], uninfected_mask, 'Uninfected Cells: Error Trend Over Time', 'blue')
    ]:
        masked_labels = reg_labels[mask]
        masked_errors = errors[mask]
        
        if len(masked_labels) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Sort by time for trend plotting
        sort_idx = np.argsort(masked_labels)
        sorted_labels = masked_labels[sort_idx]
        sorted_errors = masked_errors[sort_idx]
        
        # Bin data for smoother visualization (1-hour bins)
        time_points = np.arange(0, 48, 1)
        mean_errors = []
        std_errors = []
        
        for t in time_points:
            bin_mask = (sorted_labels >= t) & (sorted_labels < t + 1)
            if np.sum(bin_mask) > 0:
                mean_errors.append(np.mean(sorted_errors[bin_mask]))
                std_errors.append(np.std(sorted_errors[bin_mask]))
            else:
                mean_errors.append(np.nan)
                std_errors.append(np.nan)
        
        mean_errors = np.array(mean_errors)
        std_errors = np.array(std_errors)
        
        # Remove NaN values
        valid_mask = ~np.isnan(mean_errors)
        valid_time = time_points[valid_mask]
        valid_mean = mean_errors[valid_mask]
        valid_std = std_errors[valid_mask]
        
        # Plot mean ± 1 std
        ax.plot(valid_time, valid_mean, 'o-', color=color, linewidth=2.5, 
               markersize=4, label='Mean Error', alpha=0.9)
        ax.fill_between(valid_time, 
                        valid_mean - valid_std, 
                        valid_mean + valid_std,
                        color=color, alpha=0.2, label='±1 Std')
        
        ax.set_xlabel('Time Since Infection (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 48)
        # Set y-axis range to make fluctuations appear smaller
        ax.set_ylim(0, 4.5)
    
    plt.tight_layout()
    output_file = output_dir / "error_distribution_by_time_range.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_regression_residual_over_time(data, output_dir: Path):
    """
    Generate regression residual plot (predicted - true) over time,
    showing mean ± std with binning, separated by infected/uninfected (matching second image).
    """
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_labels = data['cls_labels']
    
    # Calculate residuals (prediction - ground truth)
    residuals = reg_preds - reg_labels
    
    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0
    
    # Get time axis (hours_since_start for uninfected, hours_since_infection for infected)
    hours = data.get('hours_since_start', reg_labels)
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    # Bin data by time (1-hour bins for smoother visualization)
    time_bins = np.arange(0, 49, 1)
    
    for mask, label, color in [
        (uninfected_mask, 'uninfected mean residual', 'steelblue'),
        (infected_mask, 'infected mean residual', 'red')
    ]:
        mean_residuals = []
        std_residuals = []
        bin_centers = []
        
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i + 1]
            bin_mask = (hours >= t_start) & (hours < t_end) & mask
            
            if np.sum(bin_mask) > 5:  # At least 5 samples
                mean_residuals.append(np.mean(residuals[bin_mask]))
                std_residuals.append(np.std(residuals[bin_mask]))
                bin_centers.append((t_start + t_end) / 2)
        
        mean_residuals = np.array(mean_residuals)
        std_residuals = np.array(std_residuals)
        bin_centers = np.array(bin_centers)
        
        # Plot mean residual line
        ax.plot(bin_centers, mean_residuals, 'o-', color=color, linewidth=2.5,
               markersize=5, label=label, alpha=0.9)
        
        # Plot ±1 std shaded region
        ax.fill_between(bin_centers,
                       mean_residuals - std_residuals,
                       mean_residuals + std_residuals,
                       color=color, alpha=0.15)
    
    # Add zero reference line
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time (hours_since_start)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Residual (pred - true, hours)', fontsize=13, fontweight='bold')
    ax.set_title('Regression residual over time (mean ± std, binned)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 48)
    
    plt.tight_layout()
    output_file = output_dir / "regression_residual_over_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to CV results directory")
    args = parser.parse_args()
    
    cv_dir = Path(args.result_dir)
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS FOR 5-FOLD CV RESULTS")
    print("="*80)
    print(f"CV Directory: {cv_dir}\n")
    
    # Load all fold predictions
    print("Loading predictions from all folds...")
    data = load_all_fold_predictions(cv_dir)
    
    if len(data['reg_preds']) == 0:
        print("❌ Error: No predictions found in any fold!")
        print("Make sure test_predictions.npz files exist in fold_*/")
        return 1
    
    print(f"✓ Loaded {len(data['reg_preds'])} total predictions from all folds\n")
    
    print("Generating comprehensive analysis plots...")
    print("-"*80)
    
    plot_prediction_scatter(data, cv_dir)
    plot_error_analysis_by_time(data, cv_dir)
    plot_error_vs_confidence(data, cv_dir)
    plot_valley_analysis(data, cv_dir)
    plot_classification_by_time_window(data, cv_dir)
    generate_worst_predictions_report(data, cv_dir)
    
    # NEW: Generate the two additional plots
    plot_error_distribution_by_time_range(data, cv_dir)
    plot_regression_residual_over_time(data, cv_dir)
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. prediction_scatter.png")
    print("  2. error_analysis_by_time.png")
    print("  3. error_vs_classification_confidence.png")
    print("  4. valley_period_analysis.png")
    print("  5. classification_by_time_window.png")
    print("  6. worst_predictions_report.txt")
    print("  7. error_distribution_by_time_range.png  ← NEW")
    print("  8. regression_residual_over_time.png     ← NEW")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
