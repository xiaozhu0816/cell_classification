"""
Analyze Regression Errors - Find Worst Predictions and Patterns

This script identifies samples with largest prediction errors and analyzes:
1. Which time ranges have highest errors?
2. Are errors correlated with classification confidence?
3. Do certain cells consistently have high errors?
4. Visualize worst predictions

Usage:
    python analyze_regression_errors.py --result-dir outputs/multitask_resnet50/20260102-163144
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(result_dir: Path):
    """Load predictions and metadata."""
    
    # Load predictions
    pred_file = result_dir / "test_predictions.npz"
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_file}")
    
    data = np.load(pred_file)
    
    results = {
        'time_preds': data['time_preds'],
        'time_targets': data['time_targets'],
        'cls_preds': data['cls_preds'],
        'cls_targets': data['cls_targets'],
    }
    
    # Load metadata if available
    meta_file = result_dir / "test_metadata.json"
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            results['metadata'] = json.load(f)
    
    return results


def analyze_errors_by_time_range(time_preds, time_targets, cls_targets, output_dir):
    """Analyze errors across different time ranges."""
    
    errors = np.abs(time_preds - time_targets)
    
    # Define time bins
    time_bins = [0, 8, 12, 16, 20, 24, 30, 36, 48]
    bin_labels = ['0-8h', '8-12h', '12-16h', '16-20h', '20-24h', '24-30h', '30-36h', '36-48h']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Error distribution by time range (infected)
    ax = axes[0, 0]
    infected_mask = cls_targets == 1
    infected_targets = time_targets[infected_mask]
    infected_errors = errors[infected_mask]
    
    bin_indices = np.digitize(infected_targets, time_bins) - 1
    
    bin_errors = [infected_errors[bin_indices == i] for i in range(len(bin_labels))]
    
    bp = ax.boxplot(bin_errors, labels=bin_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    
    ax.set_xlabel('Time Range (hours since infection)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Infected Cells: Error Distribution by Time Range', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add median values on top
    for i, (label, data) in enumerate(zip(bin_labels, bin_errors)):
        if len(data) > 0:
            median = np.median(data)
            ax.text(i+1, median, f'{median:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Error distribution by time range (uninfected)
    ax = axes[0, 1]
    uninfected_mask = cls_targets == 0
    uninfected_targets = time_targets[uninfected_mask]
    uninfected_errors = errors[uninfected_mask]
    
    bin_indices = np.digitize(uninfected_targets, time_bins) - 1
    
    bin_errors = [uninfected_errors[bin_indices == i] for i in range(len(bin_labels))]
    
    bp = ax.boxplot(bin_errors, labels=bin_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_xlabel('Time Range (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Uninfected Cells: Error Distribution by Time Range', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    for i, (label, data) in enumerate(zip(bin_labels, bin_errors)):
        if len(data) > 0:
            median = np.median(data)
            ax.text(i+1, median, f'{median:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Mean error trend over time (infected)
    ax = axes[1, 0]
    
    # Group by 2-hour windows
    window_size = 2.0
    max_time = 48.0
    window_starts = np.arange(0, max_time, window_size)
    
    infected_mean_errors = []
    infected_std_errors = []
    window_centers = []
    
    for start in window_starts:
        end = start + window_size
        mask = infected_mask & (time_targets >= start) & (time_targets < end)
        if mask.sum() > 0:
            window_centers.append(start + window_size/2)
            infected_mean_errors.append(errors[mask].mean())
            infected_std_errors.append(errors[mask].std())
    
    window_centers = np.array(window_centers)
    infected_mean_errors = np.array(infected_mean_errors)
    infected_std_errors = np.array(infected_std_errors)
    
    ax.plot(window_centers, infected_mean_errors, 'o-', color='red', linewidth=2, markersize=6, label='Mean Error')
    ax.fill_between(window_centers, 
                     infected_mean_errors - infected_std_errors,
                     infected_mean_errors + infected_std_errors,
                     alpha=0.3, color='red', label='±1 Std')
    
    ax.set_xlabel('Time Since Infection (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Infected Cells: Error Trend Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Mean error trend over time (uninfected)
    ax = axes[1, 1]
    
    uninfected_mean_errors = []
    uninfected_std_errors = []
    window_centers = []
    
    for start in window_starts:
        end = start + window_size
        mask = uninfected_mask & (time_targets >= start) & (time_targets < end)
        if mask.sum() > 0:
            window_centers.append(start + window_size/2)
            uninfected_mean_errors.append(errors[mask].mean())
            uninfected_std_errors.append(errors[mask].std())
    
    window_centers = np.array(window_centers)
    uninfected_mean_errors = np.array(uninfected_mean_errors)
    uninfected_std_errors = np.array(uninfected_std_errors)
    
    ax.plot(window_centers, uninfected_mean_errors, 'o-', color='blue', linewidth=2, markersize=6, label='Mean Error')
    ax.fill_between(window_centers,
                     uninfected_mean_errors - uninfected_std_errors,
                     uninfected_mean_errors + uninfected_std_errors,
                     alpha=0.3, color='blue', label='±1 Std')
    
    ax.set_xlabel('Experiment Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Uninfected Cells: Error Trend Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "error_analysis_by_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error analysis by time to {output_file}")
    plt.close()


def analyze_classification_vs_regression_errors(time_preds, time_targets, cls_preds, cls_targets, output_dir):
    """Analyze relationship between classification confidence and regression errors."""
    
    errors = np.abs(time_preds - time_targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Error vs Classification Confidence (Infected)
    ax = axes[0]
    infected_mask = cls_targets == 1
    infected_conf = cls_preds[infected_mask]
    infected_errors = errors[infected_mask]
    
    scatter = ax.scatter(infected_conf, infected_errors, alpha=0.5, c=infected_errors, 
                        cmap='Reds', s=20, edgecolors='none')
    
    # Bin by confidence and plot trend
    conf_bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_mean_errors = []
    
    for i in range(len(conf_bins)-1):
        mask = (infected_conf >= conf_bins[i]) & (infected_conf < conf_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
            bin_mean_errors.append(infected_errors[mask].mean())
    
    ax.plot(bin_centers, bin_mean_errors, 'k-', linewidth=2, label='Mean Error per Bin')
    
    ax.set_xlabel('Classification Confidence (Probability Infected)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regression Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Infected Cells: Regression Error vs Classification Confidence', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error (hours)', fontsize=10)
    
    # 2. Error vs Classification Confidence (Uninfected)
    ax = axes[1]
    uninfected_mask = cls_targets == 0
    uninfected_conf = 1 - cls_preds[uninfected_mask]  # Confidence in uninfected
    uninfected_errors = errors[uninfected_mask]
    
    scatter = ax.scatter(uninfected_conf, uninfected_errors, alpha=0.5, c=uninfected_errors,
                        cmap='Blues', s=20, edgecolors='none')
    
    # Bin by confidence and plot trend
    bin_centers = []
    bin_mean_errors = []
    
    for i in range(len(conf_bins)-1):
        mask = (uninfected_conf >= conf_bins[i]) & (uninfected_conf < conf_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
            bin_mean_errors.append(uninfected_errors[mask].mean())
    
    ax.plot(bin_centers, bin_mean_errors, 'k-', linewidth=2, label='Mean Error per Bin')
    
    ax.set_xlabel('Classification Confidence (Probability Uninfected)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regression Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Uninfected Cells: Regression Error vs Classification Confidence', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Error (hours)', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / "error_vs_classification_confidence.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error vs confidence plot to {output_file}")
    plt.close()


def analyze_regression_error_on_misclassifications(time_preds, time_targets, cls_preds, cls_targets, output_dir):
    """How does regression error behave on wrongly-classified samples?

    We compare regression abs error distributions for:
      - correctly classified
      - misclassified
    and stratify by true class (infected vs uninfected).
    """

    errors = np.abs(time_preds - time_targets)

    # interpret classification as probability infected + threshold 0.5
    y_true = cls_targets.astype(int)
    y_pred = (cls_preds >= 0.5).astype(int)
    correct = y_pred == y_true
    wrong = ~correct

    # Prepare plotting dataframe
    rows = []
    for cls_name, cls_val in [("Infected", 1), ("Uninfected", 0)]:
        for corr_name, mask in [("Correct", correct), ("Wrong", wrong)]:
            m = (y_true == cls_val) & mask
            if m.sum() == 0:
                continue
            for e in errors[m]:
                rows.append({"True Class": cls_name, "Classification": corr_name, "AbsError": float(e)})

    if len(rows) == 0:
        print("⚠ No samples available for misclassification regression analysis")
        return

    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Left: overall correct vs wrong
    ax = axes[0]
    overall = df.copy()
    overall["Group"] = overall["Classification"]
    sns.boxplot(data=overall, x="Group", y="AbsError", ax=ax, palette={"Correct": "#4C78A8", "Wrong": "#F58518"})
    sns.stripplot(data=overall, x="Group", y="AbsError", ax=ax, color="black", alpha=0.25, size=2, jitter=0.25)
    ax.set_title("Regression Error vs Classification Correctness", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Regression Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # annotate counts / medians
    for i, grp in enumerate(["Correct", "Wrong"]):
        sub = overall[overall["Group"] == grp]["AbsError"].values
        if sub.size:
            ax.text(i, np.nanmedian(sub), f"n={sub.size}\nmed={np.nanmedian(sub):.2f}", ha="center", va="bottom", fontsize=10)

    # Right: by class
    ax = axes[1]
    sns.boxplot(
        data=df,
        x="True Class",
        y="AbsError",
        hue="Classification",
        ax=ax,
        palette={"Correct": "#4C78A8", "Wrong": "#F58518"},
    )
    # light scatter overlay
    sns.stripplot(
        data=df,
        x="True Class",
        y="AbsError",
        hue="Classification",
        dodge=True,
        ax=ax,
        color="black",
        alpha=0.20,
        size=2,
        jitter=0.25,
    )
    # remove duplicated legends (stripplot adds another)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], title="Classification", loc="best")
    ax.set_title("Regression Error on Misclassifications (by True Class)", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = Path(output_dir) / "regression_error_by_classification_correctness.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved misclassification regression error plot to {out}")


def analyze_high_confidence_misclassifications(
    time_preds,
    time_targets,
    cls_preds,
    cls_targets,
    output_dir,
    conf_threshold: float = 0.9,
):
    """Focus on *high-confidence* mistakes and their regression error.

    We define confidence as probability of the predicted class:
      conf = p if predicted infected else (1-p)

    Then we compare regression error distributions for:
      - Correct (all)
      - Wrong (all)
      - Wrong & High-Confidence (conf >= threshold)

    And also show a by-true-class breakdown.
    """

    errors = np.abs(time_preds - time_targets)
    y_true = cls_targets.astype(int)
    y_pred = (cls_preds >= 0.5).astype(int)
    correct = y_pred == y_true
    wrong = ~correct
    pred_conf = np.where(y_pred == 1, cls_preds, 1 - cls_preds)
    high_conf_wrong = wrong & (pred_conf >= conf_threshold)

    rows = []
    # overall groups
    for name, mask in [
        ("Correct", correct),
        ("Wrong", wrong),
        (f"Wrong+HighConf≥{conf_threshold:.2f}", high_conf_wrong),
    ]:
        if mask.sum() == 0:
            continue
        for e in errors[mask]:
            rows.append({"Group": name, "AbsError": float(e)})

    if len(rows) == 0:
        print("⚠ No samples available for high-confidence misclassification analysis")
        return

    df_overall = pd.DataFrame(rows)

    # by true class
    rows = []
    for cls_name, cls_val in [("Infected", 1), ("Uninfected", 0)]:
        for name, mask in [
            ("Correct", correct),
            ("Wrong", wrong),
            (f"Wrong+HighConf≥{conf_threshold:.2f}", high_conf_wrong),
        ]:
            m = (y_true == cls_val) & mask
            if m.sum() == 0:
                continue
            for e in errors[m]:
                rows.append({"True Class": cls_name, "Group": name, "AbsError": float(e)})
    df_byclass = pd.DataFrame(rows) if rows else None

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    ax = axes[0]
    order = [
        "Correct",
        "Wrong",
        f"Wrong+HighConf≥{conf_threshold:.2f}",
    ]
    order = [o for o in order if o in set(df_overall["Group"].tolist())]
    sns.boxplot(data=df_overall, x="Group", y="AbsError", ax=ax, order=order)
    sns.stripplot(data=df_overall, x="Group", y="AbsError", ax=ax, color="black", alpha=0.25, size=2, jitter=0.25, order=order)
    ax.set_title("Regression Error: High-Confidence Misclassifications", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Regression Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    if df_byclass is not None and len(df_byclass) > 0:
        sns.boxplot(data=df_byclass, x="True Class", y="AbsError", hue="Group", ax=ax)
        ax.set_title("High-Confidence Misclassifications (by True Class)", fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Group", loc="best")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No by-class samples", ha="center", va="center")

    plt.tight_layout()
    out = Path(output_dir) / "regression_error_high_conf_wrong.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved high-confidence misclassification plot to {out}")


def find_worst_predictions(time_preds, time_targets, cls_preds, cls_targets, output_dir, top_k=20):
    """Find and report worst predictions."""
    
    errors = np.abs(time_preds - time_targets)
    
    # Find worst predictions for infected
    infected_mask = cls_targets == 1
    infected_indices = np.where(infected_mask)[0]
    infected_errors = errors[infected_mask]
    worst_infected_idx = infected_indices[np.argsort(infected_errors)[-top_k:]]
    
    # Find worst predictions for uninfected
    uninfected_mask = cls_targets == 0
    uninfected_indices = np.where(uninfected_mask)[0]
    uninfected_errors = errors[uninfected_mask]
    worst_uninfected_idx = uninfected_indices[np.argsort(uninfected_errors)[-top_k:]]
    
    # Create summary report
    report = []
    report.append("="*80)
    report.append(f"TOP {top_k} WORST PREDICTIONS")
    report.append("="*80)
    
    report.append(f"\nINFECTED CELLS (Worst {top_k}):")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'True Time':<12} {'Pred Time':<12} {'Error':<10} {'Cls Conf':<10}")
    report.append("-"*80)
    
    for rank, idx in enumerate(worst_infected_idx[::-1], 1):
        true_time = time_targets[idx]
        pred_time = time_preds[idx]
        error = errors[idx]
        cls_conf = cls_preds[idx]
        report.append(f"{rank:<6} {true_time:<12.2f} {pred_time:<12.2f} {error:<10.2f} {cls_conf:<10.4f}")
    
    report.append(f"\nUNINFECTED CELLS (Worst {top_k}):")
    report.append("-"*80)
    report.append(f"{'Rank':<6} {'True Time':<12} {'Pred Time':<12} {'Error':<10} {'Cls Conf':<10}")
    report.append("-"*80)
    
    for rank, idx in enumerate(worst_uninfected_idx[::-1], 1):
        true_time = time_targets[idx]
        pred_time = time_preds[idx]
        error = errors[idx]
        cls_conf = cls_preds[idx]
        report.append(f"{rank:<6} {true_time:<12.2f} {pred_time:<12.2f} {error:<10.2f} {cls_conf:<10.4f}")
    
    # Statistics
    report.append("\n" + "="*80)
    report.append("ERROR STATISTICS")
    report.append("="*80)
    
    report.append(f"\nINFECTED CELLS:")
    report.append(f"  Mean Error: {infected_errors.mean():.3f}h")
    report.append(f"  Median Error: {np.median(infected_errors):.3f}h")
    report.append(f"  Std Error: {infected_errors.std():.3f}h")
    report.append(f"  Max Error: {infected_errors.max():.3f}h")
    report.append(f"  95th Percentile: {np.percentile(infected_errors, 95):.3f}h")
    
    report.append(f"\nUNINFECTED CELLS:")
    report.append(f"  Mean Error: {uninfected_errors.mean():.3f}h")
    report.append(f"  Median Error: {np.median(uninfected_errors):.3f}h")
    report.append(f"  Std Error: {uninfected_errors.std():.3f}h")
    report.append(f"  Max Error: {uninfected_errors.max():.3f}h")
    report.append(f"  95th Percentile: {np.percentile(uninfected_errors, 95):.3f}h")
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    output_file = output_dir / "worst_predictions_report.txt"
    with open(output_file, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Saved worst predictions report to {output_file}")
    
    return worst_infected_idx, worst_uninfected_idx


def main():
    parser = argparse.ArgumentParser(description="Analyze regression errors in multitask model")
    parser.add_argument("--result-dir", type=str, required=True, help="Path to results directory")
    parser.add_argument("--top-k", type=int, default=20, help="Number of worst predictions to report")
    parser.add_argument(
        "--high-conf-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for the 'high-confidence wrong' analysis (probability of predicted class)",
    )
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return 1
    
    print("="*80)
    print("REGRESSION ERROR ANALYSIS")
    print("="*80)
    print(f"Result directory: {result_dir}\n")
    
    # Load results
    print("Loading predictions...")
    results = load_results(result_dir)
    
    time_preds = results['time_preds']
    time_targets = results['time_targets']
    cls_preds = results['cls_preds']
    cls_targets = results['cls_targets']
    
    print(f"Loaded {len(time_preds)} predictions")
    print(f"  Infected: {(cls_targets == 1).sum()}")
    print(f"  Uninfected: {(cls_targets == 0).sum()}\n")
    
    # Run analyses
    print("1. Analyzing errors by time range...")
    analyze_errors_by_time_range(time_preds, time_targets, cls_targets, result_dir)
    
    print("\n2. Analyzing classification vs regression errors...")
    analyze_classification_vs_regression_errors(time_preds, time_targets, cls_preds, cls_targets, result_dir)

    print("\n2b. Analyzing regression error on misclassified samples...")
    analyze_regression_error_on_misclassifications(time_preds, time_targets, cls_preds, cls_targets, result_dir)

    print(f"\n2c. High-confidence misclassifications (threshold={args.high_conf_threshold:.2f})...")
    analyze_high_confidence_misclassifications(
        time_preds,
        time_targets,
        cls_preds,
        cls_targets,
        result_dir,
        conf_threshold=args.high_conf_threshold,
    )
    
    print(f"\n3. Finding worst {args.top_k} predictions...")
    find_worst_predictions(time_preds, time_targets, cls_preds, cls_targets, result_dir, top_k=args.top_k)
    
    print("\n" + "="*80)
    print("✓ Error analysis complete!")
    print(f"✓ Results saved to {result_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
