"""
Generate Additional Visualizations for Multitask CV Results

Creates:
1. Training curves across all folds
2. Per-fold performance comparison
3. Prediction scatter plots (infected vs uninfected)
4. Confusion matrices
5. Error distribution plots

Usage:
    python generate_cv_visualizations.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def load_fold_results(cv_dir: Path, fold_idx: int):
    """Load results for a specific fold."""
    fold_dir = cv_dir / f"fold_{fold_idx}"
    
    # Load predictions
    pred_file = fold_dir / "test_predictions.npz"
    if not pred_file.exists():
        return None
    
    data = np.load(pred_file)
    return {
        'cls_probs': data['cls_probs'],
        'cls_preds': data['cls_preds'],
        'cls_labels': data['cls_labels'],
        'reg_preds': data['reg_preds'],
        'reg_labels': data['reg_labels'],
        'image_paths': data['image_paths']
    }


def plot_training_curves(cv_dir: Path, output_dir: Path):
    """Plot training curves for all folds."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_to_plot = [
        ('train_cls_loss', 'Classification Loss (Train)', axes[0, 0]),
        ('val_cls_loss', 'Classification Loss (Val)', axes[0, 1]),
        ('train_reg_loss', 'Regression Loss (Train)', axes[0, 2]),
        ('val_reg_loss', 'Regression Loss (Val)', axes[1, 0]),
        ('val_cls_auc', 'Classification AUC (Val)', axes[1, 1]),
        ('val_reg_mae', 'Regression MAE (Val)', axes[1, 2])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for fold_idx in range(1, 6):  # Folds are numbered 1-5
        fold_dir = cv_dir / f"fold_{fold_idx}"
        history_file = fold_dir / "training_history.json"
        
        if not history_file.exists():
            continue
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs = list(range(1, len(history['train_cls_loss']) + 1))
        
        for metric_key, title, ax in metrics_to_plot:
            if metric_key in history:
                ax.plot(epochs, history[metric_key], label=f'Fold {fold_idx}',
                       color=colors[fold_idx-1], linewidth=2, alpha=0.7)  # fold_idx-1 for color indexing
    
    for metric_key, title, ax in metrics_to_plot:
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(title.split('(')[0].strip(), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "cv_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved training curves to {output_file}")
    plt.close()


def plot_fold_performance(cv_dir: Path, output_dir: Path):
    """Plot performance comparison across folds."""
    summary_file = cv_dir / "cv_summary.json"
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    fold_results = summary['fold_results']
    
    # Extract metrics
    folds = [f"Fold {i}" for i in range(5)]
    cls_auc = [fold['test_metrics']['cls_auc'] for fold in fold_results]
    cls_f1 = [fold['test_metrics']['cls_f1'] for fold in fold_results]
    reg_mae = [fold['test_metrics']['reg_mae'] for fold in fold_results]
    combined = [fold['test_metrics']['combined'] for fold in fold_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AUC
    ax = axes[0, 0]
    bars = ax.bar(folds, cls_auc, color='darkgreen', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(cls_auc), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(cls_auc):.4f} ± {np.std(cls_auc):.4f}')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Classification AUC by Fold', fontsize=13, fontweight='bold')
    ax.set_ylim(min(cls_auc) - 0.001, max(cls_auc) + 0.001)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1
    ax = axes[0, 1]
    bars = ax.bar(folds, cls_f1, color='darkblue', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(cls_f1), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(cls_f1):.4f} ± {np.std(cls_f1):.4f}')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Classification F1 by Fold', fontsize=13, fontweight='bold')
    ax.set_ylim(min(cls_f1) - 0.001, max(cls_f1) + 0.001)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # MAE
    ax = axes[1, 0]
    bars = ax.bar(folds, reg_mae, color='darkred', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(reg_mae), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(reg_mae):.3f} ± {np.std(reg_mae):.3f} hours')
    ax.set_ylabel('MAE (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Regression MAE by Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Combined
    ax = axes[1, 1]
    bars = ax.bar(folds, combined, color='purple', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(combined), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(combined):.4f} ± {np.std(combined):.4f}')
    ax.set_ylabel('Combined Score', fontsize=12, fontweight='bold')
    ax.set_title('Combined Metric by Fold', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / "cv_fold_performance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved fold performance to {output_file}")
    plt.close()


def plot_prediction_scatter(cv_dir: Path, output_dir: Path):
    """Plot prediction scatter for regression task."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_preds = []
    all_labels = []
    all_classes = []
    
    for fold_idx in range(1, 6):  # Folds are numbered 1-5
        fold_data = load_fold_results(cv_dir, fold_idx)
        if fold_data is None:
            continue
        
        reg_preds = fold_data['reg_preds']
        reg_labels = fold_data['reg_labels']
        cls_labels = fold_data['cls_labels']
        
        all_preds.extend(reg_preds)
        all_labels.extend(reg_labels)
        all_classes.extend(cls_labels)
        
        # Plot for this fold
        ax = axes[fold_idx-1]  # fold_idx-1 for axes indexing
        
        # Separate by class
        infected_mask = cls_labels == 1
        uninfected_mask = cls_labels == 0
        
        ax.scatter(reg_labels[infected_mask], reg_preds[infected_mask],
                  alpha=0.5, s=20, c='red', label='Infected', edgecolors='none')
        ax.scatter(reg_labels[uninfected_mask], reg_preds[uninfected_mask],
                  alpha=0.5, s=20, c='blue', label='Uninfected', edgecolors='none')
        
        # Perfect prediction line
        min_val = min(reg_labels.min(), reg_preds.min())
        max_val = max(reg_labels.max(), reg_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
        
        # Calculate MAE
        mae = np.mean(np.abs(reg_preds - reg_labels))
        
        ax.set_xlabel('True Time (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Time (hours)', fontsize=11, fontweight='bold')
        ax.set_title(f'Fold {fold_idx} - MAE: {mae:.2f}h', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Aggregate plot
    ax = axes[5]
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_classes = np.array(all_classes)
    
    if len(all_preds) == 0:
        ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('All Folds Combined - No Data', fontsize=12, fontweight='bold')
    else:
        infected_mask = all_classes == 1
        uninfected_mask = all_classes == 0
        
        ax.scatter(all_labels[infected_mask], all_preds[infected_mask],
                  alpha=0.3, s=15, c='red', label='Infected', edgecolors='none')
        ax.scatter(all_labels[uninfected_mask], all_preds[uninfected_mask],
                  alpha=0.3, s=15, c='blue', label='Uninfected', edgecolors='none')
        
        min_val = min(all_labels.min(), all_preds.min())
        max_val = max(all_labels.max(), all_preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7)
        
        mae = np.mean(np.abs(all_preds - all_labels))
        
        ax.set_xlabel('True Time (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted Time (hours)', fontsize=11, fontweight='bold')
        ax.set_title(f'All Folds Combined - MAE: {mae:.2f}h', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    output_file = output_dir / "cv_prediction_scatter.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved prediction scatter to {output_file}")
    plt.close()


def plot_confusion_matrices(cv_dir: Path, output_dir: Path):
    """Plot confusion matrices for classification task."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    all_preds = []
    all_labels = []
    
    for fold_idx in range(1, 6):  # Folds are numbered 1-5
        fold_data = load_fold_results(cv_dir, fold_idx)
        if fold_data is None:
            continue
        
        cls_preds = fold_data['cls_preds']
        cls_labels = fold_data['cls_labels']
        
        all_preds.extend(cls_preds)
        all_labels.extend(cls_labels)
        
        # Plot for this fold
        ax = axes[fold_idx-1]  # fold_idx-1 for axes indexing
        cm = confusion_matrix(cls_labels, cls_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Uninfected', 'Infected'],
                   yticklabels=['Uninfected', 'Infected'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title(f'Fold {fold_idx} Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Aggregate confusion matrix
    ax = axes[5]
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if len(all_preds) == 0:
        ax.text(0.5, 0.5, 'No predictions found', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('All Folds Combined - No Data', fontsize=12, fontweight='bold')
    else:
        cm = confusion_matrix(all_labels, all_preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Uninfected', 'Infected'],
                   yticklabels=['Uninfected', 'Infected'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_title('All Folds Combined Confusion Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / "cv_confusion_matrices.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to {output_file}")
    plt.close()


def plot_error_distributions(cv_dir: Path, output_dir: Path):
    """Plot error distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    all_errors = []
    all_classes = []
    all_labels = []
    
    for fold_idx in range(1, 6):  # Folds are numbered 1-5
        fold_data = load_fold_results(cv_dir, fold_idx)
        if fold_data is None:
            continue
        
        reg_preds = fold_data['reg_preds']
        reg_labels = fold_data['reg_labels']
        cls_labels = fold_data['cls_labels']
        
        errors = np.abs(reg_preds - reg_labels)
        all_errors.extend(errors)
        all_classes.extend(cls_labels)
        all_labels.extend(reg_labels)
    
    all_errors = np.array(all_errors)
    all_classes = np.array(all_classes)
    all_labels = np.array(all_labels)
    
    if len(all_errors) == 0:
        print("⚠ Warning: No error data found, skipping error distribution plots")
        plt.close()
        return
    
    # Error histogram
    ax = axes[0, 0]
    ax.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(all_errors):.2f}h')
    ax.axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2,
              label=f'Median: {np.median(all_errors):.2f}h')
    ax.set_xlabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution (All Folds)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error by class
    ax = axes[0, 1]
    infected_errors = all_errors[all_classes == 1]
    uninfected_errors = all_errors[all_classes == 0]
    
    bp = ax.boxplot([infected_errors, uninfected_errors],
                     labels=['Infected', 'Uninfected'],
                     patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error by Cell Type', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Error vs time
    ax = axes[1, 0]
    ax.scatter(all_labels, all_errors, alpha=0.3, s=10, c='steelblue', edgecolors='none')
    
    # Add smoothed trend
    from scipy.ndimage import gaussian_filter1d
    sorted_idx = np.argsort(all_labels)
    window_size = len(all_labels) // 20
    smoothed = gaussian_filter1d(all_errors[sorted_idx], sigma=window_size)
    ax.plot(all_labels[sorted_idx], smoothed, 'r-', linewidth=3, alpha=0.8, label='Trend')
    
    ax.set_xlabel('True Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Time', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error percentiles by time bins
    ax = axes[1, 1]
    time_bins = np.arange(0, 48, 6)
    bin_centers = time_bins[:-1] + 3
    
    percentiles = []
    for i in range(len(time_bins) - 1):
        mask = (all_labels >= time_bins[i]) & (all_labels < time_bins[i+1])
        if mask.sum() > 0:
            percentiles.append({
                '25': np.percentile(all_errors[mask], 25),
                '50': np.percentile(all_errors[mask], 50),
                '75': np.percentile(all_errors[mask], 75),
            })
        else:
            percentiles.append({'25': 0, '50': 0, '75': 0})
    
    p25 = [p['25'] for p in percentiles]
    p50 = [p['50'] for p in percentiles]
    p75 = [p['75'] for p in percentiles]
    
    ax.fill_between(bin_centers, p25, p75, alpha=0.3, color='steelblue', label='25-75 percentile')
    ax.plot(bin_centers, p50, 'o-', color='darkblue', linewidth=2, markersize=8, label='Median')
    
    ax.set_xlabel('Time Window Center (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Error Percentiles by Time Period', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "cv_error_distributions.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved error distributions to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to CV results directory (e.g., outputs/.../TIMESTAMP_5fold)")
    args = parser.parse_args()
    
    cv_dir = Path(args.result_dir)
    
    print("="*80)
    print("GENERATING ADDITIONAL CV VISUALIZATIONS")
    print("="*80)
    print(f"CV Directory: {cv_dir}\n")
    
    # Check if results exist
    if not (cv_dir / "cv_summary.json").exists():
        print(f"❌ Error: cv_summary.json not found in {cv_dir}")
        return 1
    
    print("📊 Generating visualizations...")
    print("-"*80)
    
    plot_training_curves(cv_dir, cv_dir)
    plot_fold_performance(cv_dir, cv_dir)
    plot_prediction_scatter(cv_dir, cv_dir)
    plot_confusion_matrices(cv_dir, cv_dir)
    plot_error_distributions(cv_dir, cv_dir)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  1. cv_training_curves.png      - Training/validation curves for all folds")
    print("  2. cv_fold_performance.png      - Per-fold performance comparison")
    print("  3. cv_prediction_scatter.png    - Regression predictions vs ground truth")
    print("  4. cv_confusion_matrices.png    - Classification confusion matrices")
    print("  5. cv_error_distributions.png   - Error analysis and distributions")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
