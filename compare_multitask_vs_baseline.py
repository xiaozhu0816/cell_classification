"""
Compare Multitask CV Results with Single-Task Baseline

Compares:
- Multitask 5-fold CV (classification + regression)
- Single-task baseline (classification only, from interval sweep)

Usage:
    python compare_multitask_vs_baseline.py \\
        --multitask-dir outputs/multitask_resnet50/20260105-155852_5fold \\
        --baseline-dir outputs/interval_sweep_analysis/20251212-145928/train-test_interval_1-46_sliding_window_fast_20251231-161811
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_multitask_results(cv_dir: Path):
    """Load multitask CV summary."""
    summary_file = cv_dir / "cv_summary.json"
    with open(summary_file, 'r') as f:
        data = json.load(f)
    
    # Extract aggregated metrics
    fold_results = data['fold_results']
    
    metrics = {}
    for key in fold_results[0]['test_metrics'].keys():
        values = [fold['test_metrics'][key] for fold in fold_results]
        metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return metrics


def load_baseline_temporal(baseline_dir: Path):
    """Load baseline sliding window results."""
    data_file = baseline_dir / "final_model_sliding_w6_s3_data.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    return data


def load_multitask_temporal(cv_dir: Path):
    """Load multitask temporal generalization results."""
    temporal_file = cv_dir / "cv_temporal_metrics.json"
    with open(temporal_file, 'r') as f:
        data = json.load(f)
    
    return data


def create_comparison_table(multitask_metrics, baseline_data):
    """Create comparison table."""
    
    # For baseline, compute overall metrics from sliding window
    # Structure: fold_metrics[window_idx] = [fold0, fold1, fold2, fold3, fold4]
    baseline_auc_values = []
    for window_folds in baseline_data['results']['auc']['fold_metrics']:
        baseline_auc_values.extend(window_folds)
    baseline_auc_mean = np.mean(baseline_auc_values)
    baseline_auc_std = np.std(baseline_auc_values)
    
    baseline_acc_values = []
    for window_folds in baseline_data['results']['accuracy']['fold_metrics']:
        baseline_acc_values.extend(window_folds)
    baseline_acc_mean = np.mean(baseline_acc_values)
    baseline_acc_std = np.std(baseline_acc_values)
    
    baseline_f1_values = []
    for window_folds in baseline_data['results']['f1']['fold_metrics']:
        baseline_f1_values.extend(window_folds)
    baseline_f1_mean = np.mean(baseline_f1_values)
    baseline_f1_std = np.std(baseline_f1_values)
    
    print("="*80)
    print("MULTITASK vs SINGLE-TASK COMPARISON")
    print("="*80)
    
    print("\n📊 CLASSIFICATION METRICS:")
    print("-"*80)
    print(f"{'Metric':<20} {'Multitask (5-Fold)':<30} {'Baseline (5-Fold)':<30} {'Difference'}")
    print("-"*80)
    
    # AUC
    mt_auc = multitask_metrics['cls_auc']
    diff_auc = mt_auc['mean'] - baseline_auc_mean
    symbol = "✓" if diff_auc > 0 else "✗"
    print(f"{'AUC':<20} {mt_auc['mean']:.4f} ± {mt_auc['std']:.4f}            "
          f"{baseline_auc_mean:.4f} ± {baseline_auc_std:.4f}            {symbol} {diff_auc:+.4f}")
    
    # Accuracy
    mt_acc = multitask_metrics['cls_accuracy']
    diff_acc = mt_acc['mean'] - baseline_acc_mean
    symbol = "✓" if diff_acc > 0 else "✗"
    print(f"{'Accuracy':<20} {mt_acc['mean']:.4f} ± {mt_acc['std']:.4f}            "
          f"{baseline_acc_mean:.4f} ± {baseline_acc_std:.4f}            {symbol} {diff_acc:+.4f}")
    
    # F1
    mt_f1 = multitask_metrics['cls_f1']
    diff_f1 = mt_f1['mean'] - baseline_f1_mean
    symbol = "✓" if diff_f1 > 0 else "✗"
    print(f"{'F1 Score':<20} {mt_f1['mean']:.4f} ± {mt_f1['std']:.4f}            "
          f"{baseline_f1_mean:.4f} ± {baseline_f1_std:.4f}            {symbol} {diff_f1:+.4f}")
    
    print("\n📈 REGRESSION METRICS (Multitask Only):")
    print("-"*80)
    mt_mae = multitask_metrics['reg_mae']
    mt_rmse = multitask_metrics['reg_rmse']
    print(f"{'MAE (hours)':<20} {mt_mae['mean']:.3f} ± {mt_mae['std']:.3f}")
    print(f"{'RMSE (hours)':<20} {mt_rmse['mean']:.3f} ± {mt_rmse['std']:.3f}")
    
    print("\n🎯 COMBINED METRIC (Multitask Only):")
    print("-"*80)
    mt_combined = multitask_metrics['combined']
    print(f"{'Combined Score':<20} {mt_combined['mean']:.4f} ± {mt_combined['std']:.4f}")
    print(f"  (0.6×F1 + 0.4×(1-MAE/48))")
    
    return {
        'multitask': {
            'auc': mt_auc,
            'accuracy': mt_acc,
            'f1': mt_f1,
            'mae': mt_mae,
        },
        'baseline': {
            'auc': {'mean': baseline_auc_mean, 'std': baseline_auc_std},
            'accuracy': {'mean': baseline_acc_mean, 'std': baseline_acc_std},
            'f1': {'mean': baseline_f1_mean, 'std': baseline_f1_std},
        }
    }


def plot_temporal_comparison(multitask_temporal, baseline_data, output_dir):
    """Plot temporal generalization comparison."""
    
    # Get baseline temporal data
    baseline_centers = baseline_data['results']['auc']['window_centers']
    baseline_auc_folds = baseline_data['results']['auc']['fold_metrics']
    baseline_acc_folds = baseline_data['results']['accuracy']['fold_metrics']
    baseline_f1_folds = baseline_data['results']['f1']['fold_metrics']
    
    # Compute mean and std for baseline
    # Structure: fold_metrics[window_idx] = [fold0, fold1, fold2, fold3, fold4]
    baseline_auc_mean = np.array([np.mean(window_folds) for window_folds in baseline_auc_folds])
    baseline_auc_std = np.array([np.std(window_folds) for window_folds in baseline_auc_folds])
    
    baseline_acc_mean = np.array([np.mean(window_folds) for window_folds in baseline_acc_folds])
    baseline_acc_std = np.array([np.std(window_folds) for window_folds in baseline_acc_folds])
    
    baseline_f1_mean = np.array([np.mean(window_folds) for window_folds in baseline_f1_folds])
    baseline_f1_std = np.array([np.std(window_folds) for window_folds in baseline_f1_folds])
    
    # Get multitask temporal data
    mt_centers = multitask_temporal['window_centers']
    mt_metrics = multitask_temporal['aggregated_metrics']
    
    mt_auc_mean = np.array([v for v in mt_metrics['auc']['mean'] if v is not None])
    mt_auc_std = np.array([v for v in mt_metrics['auc']['std'] if v is not None])
    mt_acc_mean = np.array([v for v in mt_metrics['accuracy']['mean'] if v is not None])
    mt_acc_std = np.array([v for v in mt_metrics['accuracy']['std'] if v is not None])
    mt_f1_mean = np.array([v for v in mt_metrics['f1']['mean'] if v is not None])
    mt_f1_std = np.array([v for v in mt_metrics['f1']['std'] if v is not None])
    
    # Create 3-panel comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # AUC
    ax = axes[0]
    ax.plot(baseline_centers, baseline_auc_mean, 'o-', color='gray', linewidth=2.5, 
            markersize=8, label='Baseline (Classification Only)', alpha=0.8)
    ax.fill_between(baseline_centers, baseline_auc_mean - baseline_auc_std,
                     baseline_auc_mean + baseline_auc_std, color='gray', alpha=0.2)
    
    ax.plot(mt_centers, mt_auc_mean, 's-', color='darkgreen', linewidth=2.5,
            markersize=8, label='Multitask (Classification + Regression)', alpha=0.8)
    ax.fill_between(mt_centers, mt_auc_mean - mt_auc_std, mt_auc_mean + mt_auc_std,
                     color='darkgreen', alpha=0.2)
    
    ax.set_xlabel('Time Window Center (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('AUC: Multitask vs Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Highlight valley
    ax.axvspan(13, 19, alpha=0.15, color='red', label='Valley Period')
    
    # Accuracy
    ax = axes[1]
    ax.plot(baseline_centers, baseline_acc_mean, 'o-', color='gray', linewidth=2.5,
            markersize=8, label='Baseline', alpha=0.8)
    ax.fill_between(baseline_centers, baseline_acc_mean - baseline_acc_std,
                     baseline_acc_mean + baseline_acc_std, color='gray', alpha=0.2)
    
    ax.plot(mt_centers, mt_acc_mean, 's-', color='darkblue', linewidth=2.5,
            markersize=8, label='Multitask', alpha=0.8)
    ax.fill_between(mt_centers, mt_acc_mean - mt_acc_std, mt_acc_mean + mt_acc_std,
                     color='darkblue', alpha=0.2)
    
    ax.set_xlabel('Time Window Center (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy: Multitask vs Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(13, 19, alpha=0.15, color='red')
    
    # F1
    ax = axes[2]
    ax.plot(baseline_centers, baseline_f1_mean, 'o-', color='gray', linewidth=2.5,
            markersize=8, label='Baseline', alpha=0.8)
    ax.fill_between(baseline_centers, baseline_f1_mean - baseline_f1_std,
                     baseline_f1_mean + baseline_f1_std, color='gray', alpha=0.2)
    
    ax.plot(mt_centers, mt_f1_mean, 's-', color='darkred', linewidth=2.5,
            markersize=8, label='Multitask', alpha=0.8)
    ax.fill_between(mt_centers, mt_f1_mean - mt_f1_std, mt_f1_mean + mt_f1_std,
                     color='darkred', alpha=0.2)
    
    ax.set_xlabel('Time Window Center (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score: Multitask vs Baseline', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(13, 19, alpha=0.15, color='red')
    
    plt.tight_layout()
    output_file = output_dir / "multitask_vs_baseline_temporal.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved temporal comparison plot to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multitask-dir", type=str, required=True,
                        help="Path to multitask CV directory")
    parser.add_argument("--baseline-dir", type=str, required=True,
                        help="Path to baseline sliding window directory")
    args = parser.parse_args()
    
    multitask_dir = Path(args.multitask_dir)
    baseline_dir = Path(args.baseline_dir)
    
    print("="*80)
    print("LOADING RESULTS")
    print("="*80)
    print(f"Multitask: {multitask_dir}")
    print(f"Baseline:  {baseline_dir}\n")
    
    # Load data
    multitask_metrics = load_multitask_results(multitask_dir)
    baseline_data = load_baseline_temporal(baseline_dir)
    multitask_temporal = load_multitask_temporal(multitask_dir)
    
    # Create comparison table
    comparison = create_comparison_table(multitask_metrics, baseline_data)
    
    # Plot temporal comparison
    print("\n" + "="*80)
    print("GENERATING TEMPORAL COMPARISON PLOT")
    print("="*80)
    plot_temporal_comparison(multitask_temporal, baseline_data, multitask_dir)
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("1. Compare AUC/Accuracy/F1 - Is multitask better or similar?")
    print("2. Check temporal plot - Does multitask reduce valley effect?")
    print("3. Multitask adds time prediction with ~1.2h MAE")
    print("4. Combined metric balances both tasks")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
