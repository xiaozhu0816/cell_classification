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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def load_all_fold_predictions(cv_dir: Path):
    """Load and aggregate predictions from all folds."""
    all_data = {
        'cls_probs': [],
        'cls_preds': [],
        'cls_labels': [],
        'reg_preds': [],
        'reg_labels': [],
        'image_paths': []
    }
    
    for fold_idx in range(1, 6):
        fold_dir = cv_dir / f"fold_{fold_idx}"
        pred_file = fold_dir / "test_predictions.npz"
        
        if not pred_file.exists():
            print(f"⚠ Warning: {pred_file} not found, skipping fold {fold_idx}")
            continue
        
        data = np.load(pred_file)
        all_data['cls_probs'].append(data['cls_probs'])
        all_data['cls_preds'].append(data['cls_preds'])
        all_data['cls_labels'].append(data['cls_labels'])
        all_data['reg_preds'].append(data['reg_preds'])
        all_data['reg_labels'].append(data['reg_labels'])
        all_data['image_paths'].extend(data['image_paths'])
    
    # Concatenate all arrays
    for key in ['cls_probs', 'cls_preds', 'cls_labels', 'reg_preds', 'reg_labels']:
        if all_data[key]:
            all_data[key] = np.concatenate(all_data[key])
        else:
            all_data[key] = np.array([])
    
    return all_data


def plot_prediction_scatter(data, output_dir: Path):
    """Generate prediction scatter plot (like analyze_multitask_results.py)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    reg_preds = data['reg_preds']
    reg_labels = data['reg_labels']
    cls_labels = data['cls_labels']
    
    # Separate by class
    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0
    
    # Scatter plots
    ax.scatter(reg_labels[infected_mask], reg_preds[infected_mask],
              alpha=0.5, s=30, c='red', label='Infected', edgecolors='none')
    ax.scatter(reg_labels[uninfected_mask], reg_preds[uninfected_mask],
              alpha=0.5, s=30, c='blue', label='Uninfected', edgecolors='none')
    
    # Perfect prediction line
    min_val = min(reg_labels.min(), reg_preds.min())
    max_val = max(reg_labels.max(), reg_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7,
            label='Perfect Prediction')
    
    # Statistics
    mae = np.mean(np.abs(reg_preds - reg_labels))
    rmse = np.sqrt(np.mean((reg_preds - reg_labels) ** 2))
    r2 = 1 - np.sum((reg_labels - reg_preds) ** 2) / np.sum((reg_labels - reg_labels.mean()) ** 2)
    
    ax.set_xlabel('True Time (hours post-infection)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Time (hours)', fontsize=13, fontweight='bold')
    ax.set_title(f'Time Prediction: 5-Fold CV Aggregated\nMAE={mae:.2f}h, RMSE={rmse:.2f}h, R²={r2:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_file = output_dir / "prediction_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
                     labels=['Infected', 'Uninfected'],
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
    
    # Get confidence (probability of predicted class)
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
    
    bp = ax.boxplot(binned_data, labels=conf_labels, patch_artist=True,
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
                    labels=['Infected\nNon-Valley', 'Infected\nValley',
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
    generate_worst_predictions_report(data, cv_dir)
    
    print("\n" + "="*80)
    print("✅ ALL ANALYSES COMPLETE!")
    print("="*80)
    print("\nGenerated files (same as 20260102-163144):")
    print("  1. prediction_scatter.png")
    print("  2. error_analysis_by_time.png")
    print("  3. error_vs_classification_confidence.png")
    print("  4. valley_period_analysis.png")
    print("  5. worst_predictions_report.txt")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
