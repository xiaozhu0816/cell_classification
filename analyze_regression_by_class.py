"""
Compare Multitask Performance at Critical Time Windows

This script specifically analyzes the 13-19h "valley" period that showed
problems in interval sweep experiments. Does multitask learning help?

Usage:
    python analyze_regression_by_class.py --result-dir outputs/multitask_resnet50/20260102-163144
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    # Load predictions
    pred_file = result_dir / "test_predictions.npz"
    data = np.load(pred_file)
    
    time_preds = data['time_preds']
    time_targets = data['time_targets']
    cls_preds = data['cls_preds']
    cls_targets = data['cls_targets']
    
    errors = np.abs(time_preds - time_targets)
    
    # Focus on the "valley" period: 13-19h
    valley_mask_infected = (cls_targets == 1) & (time_targets >= 13) & (time_targets <= 19)
    valley_mask_uninfected = (cls_targets == 0) & (time_targets >= 13) & (time_targets <= 19)
    
    # Compare with other periods
    early_mask_infected = (cls_targets == 1) & (time_targets < 6)
    mid_mask_infected = (cls_targets == 1) & (time_targets >= 6) & (time_targets < 13)
    late_mask_infected = (cls_targets == 1) & (time_targets > 19)
    
    print("="*80)
    print("CRITICAL TIME WINDOW ANALYSIS - The 13-19h Valley")
    print("="*80)
    
    print("\n📊 INFECTED CELLS - Error by Time Period:")
    print("-"*80)
    print(f"{'Period':<20} {'N Samples':<12} {'Mean Error':<12} {'Median Error':<12} {'95th %ile'}")
    print("-"*80)
    
    for name, mask in [
        ("Early (0-6h)", early_mask_infected),
        ("Mid (6-13h)", mid_mask_infected),
        ("Valley (13-19h)", valley_mask_infected),
        ("Late (>19h)", late_mask_infected),
    ]:
        if mask.sum() > 0:
            period_errors = errors[mask]
            print(f"{name:<20} {mask.sum():<12} {period_errors.mean():<12.3f} "
                  f"{np.median(period_errors):<12.3f} {np.percentile(period_errors, 95):.3f}")
    
    print("\n🔍 UNINFECTED CELLS - Error by Time Period:")
    print("-"*80)
    
    early_mask_uninf = (cls_targets == 0) & (time_targets < 13)
    late_mask_uninf = (cls_targets == 0) & (time_targets > 19)
    
    for name, mask in [
        ("Early (<13h)", early_mask_uninf),
        ("Valley (13-19h)", valley_mask_uninfected),
        ("Late (>19h)", late_mask_uninf),
    ]:
        if mask.sum() > 0:
            period_errors = errors[mask]
            print(f"{name:<20} {mask.sum():<12} {period_errors.mean():<12.3f} "
                  f"{np.median(period_errors):<12.3f} {np.percentile(period_errors, 95):.3f}")
    
    # Statistical test: Is valley significantly worse?
    if valley_mask_infected.sum() > 0 and mid_mask_infected.sum() > 0:
        valley_errors = errors[valley_mask_infected]
        mid_errors = errors[mid_mask_infected]
        
        t_stat, p_value = stats.ttest_ind(valley_errors, mid_errors)
        
        print("\n📈 STATISTICAL TEST:")
        print("-"*80)
        print(f"Valley (13-19h) vs Mid (6-13h) - Infected Cells")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if valley_errors.mean() > mid_errors.mean():
                print(f"  ⚠️  Valley period is SIGNIFICANTLY WORSE (p < 0.05)")
            else:
                print(f"  ✓ Valley period is SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            print(f"  → No significant difference (p >= 0.05)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Infected errors by period
    ax = axes[0]
    periods_inf = []
    labels_inf = []
    colors_inf = []
    
    for name, mask, color in [
        ("Early\n(0-6h)", early_mask_infected, 'lightcoral'),
        ("Mid\n(6-13h)", mid_mask_infected, 'coral'),
        ("Valley\n(13-19h)", valley_mask_infected, 'darkred'),
        ("Late\n(>19h)", late_mask_infected, 'indianred'),
    ]:
        if mask.sum() > 0:
            periods_inf.append(errors[mask])
            labels_inf.append(name)
            colors_inf.append(color)
    
    bp = ax.boxplot(periods_inf, labels=labels_inf, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_inf):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Infected Cells: Error Distribution by Time Period', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight valley
    ax.axvspan(2.5, 3.5, alpha=0.2, color='red', label='Valley Period')
    
    # Right: Uninfected errors by period
    ax = axes[1]
    periods_uninf = []
    labels_uninf = []
    colors_uninf = []
    
    for name, mask, color in [
        ("Early\n(<13h)", early_mask_uninf, 'lightblue'),
        ("Valley\n(13-19h)", valley_mask_uninfected, 'darkblue'),
        ("Late\n(>19h)", late_mask_uninf, 'steelblue'),
    ]:
        if mask.sum() > 0:
            periods_uninf.append(errors[mask])
            labels_uninf.append(name)
            colors_uninf.append(color)
    
    bp = ax.boxplot(periods_uninf, labels=labels_uninf, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_uninf):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Uninfected Cells: Error Distribution by Time Period', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight valley
    ax.axvspan(1.5, 2.5, alpha=0.2, color='blue', label='Valley Period')
    
    plt.tight_layout()
    output_file = result_dir / "valley_period_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved valley analysis plot to {output_file}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Check if valley (13-19h) has higher errors than adjacent periods")
    print("2. Compare with your interval sweep results - is it better here?")
    print("3. Uninfected valley errors reveal temporal confusion patterns")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
