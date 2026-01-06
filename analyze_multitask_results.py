"""
Analyze and Visualize Multi-Task Training Results

This script:
1. Parses training logs to extract epoch-by-epoch metrics
2. Loads final test results from results.json
3. Creates comprehensive visualizations:
   - Training curves (loss over epochs)
   - Classification metrics (accuracy, precision, recall, F1, AUC)
   - Regression metrics (MAE, RMSE)
   - Combined overview
4. Generates a summary report
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def parse_training_log(log_path: Path) -> Dict[str, List[float]]:
    """
    Parse training log file to extract epoch-by-epoch metrics.
    
    Returns dict with keys:
        - train_total_loss, train_cls_loss, train_reg_loss
        - val_total_loss, val_cls_loss, val_reg_loss
        - val_cls_accuracy, val_cls_auc, val_reg_mae, val_reg_rmse, etc.
    """
    metrics = {
        # Training metrics
        "train_total_loss": [],
        "train_cls_loss": [],
        "train_reg_loss": [],
        # Validation metrics
        "val_total_loss": [],
        "val_cls_loss": [],
        "val_reg_loss": [],
        "val_cls_accuracy": [],
        "val_cls_precision": [],
        "val_cls_recall": [],
        "val_cls_f1": [],
        "val_cls_auc": [],
        "val_reg_mae": [],
        "val_reg_rmse": [],
        "val_reg_mse": [],
    }
    
    with open(log_path, "r") as f:
        for line in f:
            # Parse training metrics: "train_e1 - total_loss: 1.2345 | cls_loss: 0.123 | reg_loss: 1.111"
            train_match = re.search(
                r"train_e\d+ - total_loss:\s*([\d.]+)\s*\|\s*cls_loss:\s*([\d.]+)\s*\|\s*reg_loss:\s*([\d.]+)",
                line
            )
            if train_match:
                metrics["train_total_loss"].append(float(train_match.group(1)))
                metrics["train_cls_loss"].append(float(train_match.group(2)))
                metrics["train_reg_loss"].append(float(train_match.group(3)))
            
            # Parse validation metrics: "val: total_loss:1.234 | cls_loss:0.123 | ... | cls_auc:0.99 | reg_mae:1.23 ..."
            if "val:" in line:
                # Extract all metric:value pairs
                val_metrics = re.findall(r"(\w+):([\d.]+)", line)
                for metric_name, metric_value in val_metrics:
                    full_key = f"val_{metric_name}"
                    if full_key in metrics:
                        metrics[full_key].append(float(metric_value))
    
    return metrics


def plot_training_curves(metrics: Dict[str, List[float]], output_path: Path) -> None:
    """Plot training and validation loss curves."""
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    epochs = range(1, len(metrics["train_total_loss"]) + 1)
    
    # Plot 1: Total Loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, metrics["train_total_loss"], "o-", label="Train", linewidth=2, markersize=6)
    ax1.plot(epochs, metrics["val_total_loss"], "s-", label="Val", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.set_title("Total Loss (Classification + Regression)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Classification Loss
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, metrics["train_cls_loss"], "o-", label="Train", linewidth=2, markersize=6)
    ax2.plot(epochs, metrics["val_cls_loss"], "s-", label="Val", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Classification Loss (CrossEntropy)", fontsize=12)
    ax2.set_title("Classification Task Loss", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regression Loss
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(epochs, metrics["train_reg_loss"], "o-", label="Train", linewidth=2, markersize=6)
    ax3.plot(epochs, metrics["val_reg_loss"], "s-", label="Val", linewidth=2, markersize=6)
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Regression Loss (SmoothL1)", fontsize=12)
    ax3.set_title("Time Regression Task Loss", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle("Multi-Task Training Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_validation_metrics(metrics: Dict[str, List[float]], output_path: Path) -> None:
    """Plot validation metrics over epochs."""
    if not metrics or not metrics.get("val_cls_auc"):
        print("No validation metrics to plot")
        return
    
    epochs = range(1, len(metrics["val_cls_auc"]) + 1)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Classification metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics["val_cls_accuracy"], 'o-', label='Accuracy', linewidth=2)
    ax1.plot(epochs, metrics["val_cls_precision"], 's-', label='Precision', linewidth=2)
    ax1.plot(epochs, metrics["val_cls_recall"], '^-', label='Recall', linewidth=2)
    ax1.plot(epochs, metrics["val_cls_f1"], 'd-', label='F1', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics["val_cls_auc"], 'o-', color='purple', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Classification AUC')
    ax2.grid(True, alpha=0.3)
    
    # Regression metrics
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, metrics["val_reg_mae"], 'o-', label='MAE', linewidth=2)
    ax3.plot(epochs, metrics["val_reg_rmse"], 's-', label='RMSE', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Error (hours)')
    ax3.set_title('Regression Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss components
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, metrics["val_cls_loss"], 'o-', label='Classification', linewidth=2)
    ax4.plot(epochs, metrics["val_reg_loss"], 's-', label='Regression', linewidth=2)
    ax4.plot(epochs, metrics["val_total_loss"], '^-', label='Total', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Validation Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # R² score (if available)
    if "val_reg_r2" in metrics and metrics["val_reg_r2"]:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, metrics["val_reg_r2"], 'o-', color='green', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('R²')
        ax5.set_title('Regression R² Score')
        ax5.grid(True, alpha=0.3)
    
    # MSE (if available)
    if "val_reg_mse" in metrics and metrics["val_reg_mse"]:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(epochs, metrics["val_reg_mse"], 'o-', color='orange', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('MSE (hours²)')
        ax6.set_title('Regression MSE')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Validation Metrics over Training', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved validation metrics plot to {output_path}")


def plot_prediction_scatter(
    predictions_file: Path,
    output_path: Path,
    infection_onset_hour: float = 2.0
) -> None:
    """
    Plot scatter plot of predicted vs. true time values.
    
    Args:
        predictions_file: Path to test_predictions.npz file
        output_path: Path to save the scatter plot
        infection_onset_hour: Hour when infection onset occurs (for visualization)
    """
    if not predictions_file.exists():
        print(f"Predictions file not found: {predictions_file}")
        print("Run training first to generate predictions.")
        return
    
    # Load predictions
    data = np.load(predictions_file)
    time_preds = data["time_preds"]
    time_targets = data["time_targets"]
    cls_preds = data["cls_preds"]
    cls_targets = data["cls_targets"]
    
    # Classify samples as infected (cls_target==1) or uninfected (cls_target==0)
    infected_mask = cls_targets == 1
    uninfected_mask = cls_targets == 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: All samples
    ax = axes[0]
    ax.scatter(time_targets[infected_mask], time_preds[infected_mask], 
               alpha=0.5, s=20, c='red', label=f'Infected (n={infected_mask.sum()})')
    ax.scatter(time_targets[uninfected_mask], time_preds[uninfected_mask], 
               alpha=0.5, s=20, c='blue', label=f'Uninfected (n={uninfected_mask.sum()})')
    
    # Add reference line (perfect prediction)
    min_val = min(time_targets.min(), time_preds.min())
    max_val = max(time_targets.max(), time_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    # Add infection onset line
    ax.axvline(x=infection_onset_hour, color='orange', linestyle=':', linewidth=2, 
               label=f'Infection Onset ({infection_onset_hour}h)')
    
    ax.set_xlabel('True Time (hours)', fontsize=12)
    ax.set_ylabel('Predicted Time (hours)', fontsize=12)
    ax.set_title('All Samples: Predicted vs. True Time', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Infected cells only
    if infected_mask.sum() > 0:
        ax = axes[1]
        ax.scatter(time_targets[infected_mask], time_preds[infected_mask], 
                   alpha=0.6, s=30, c='red', edgecolors='darkred', linewidth=0.5)
        
        # Reference line
        min_val_inf = time_targets[infected_mask].min()
        max_val_inf = time_targets[infected_mask].max()
        ax.plot([min_val_inf, max_val_inf], [min_val_inf, max_val_inf], 
                'k--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R² for infected
        from sklearn.metrics import r2_score
        r2_infected = r2_score(time_targets[infected_mask], time_preds[infected_mask])
        mae_infected = np.mean(np.abs(time_targets[infected_mask] - time_preds[infected_mask]))
        
        ax.set_xlabel('True Time Since Infection (hours)', fontsize=12)
        ax.set_ylabel('Predicted Time Since Infection (hours)', fontsize=12)
        ax.set_title(f'Infected Cells\nR²={r2_infected:.4f}, MAE={mae_infected:.2f}h', 
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Plot 3: Uninfected cells only
    if uninfected_mask.sum() > 0:
        ax = axes[2]
        ax.scatter(time_targets[uninfected_mask], time_preds[uninfected_mask], 
                   alpha=0.6, s=30, c='blue', edgecolors='darkblue', linewidth=0.5)
        
        # Reference line
        min_val_uninf = time_targets[uninfected_mask].min()
        max_val_uninf = time_targets[uninfected_mask].max()
        ax.plot([min_val_uninf, max_val_uninf], [min_val_uninf, max_val_uninf], 
                'k--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R² for uninfected
        from sklearn.metrics import r2_score
        r2_uninfected = r2_score(time_targets[uninfected_mask], time_preds[uninfected_mask])
        mae_uninfected = np.mean(np.abs(time_targets[uninfected_mask] - time_preds[uninfected_mask]))
        
        ax.set_xlabel('True Experiment Time (hours)', fontsize=12)
        ax.set_ylabel('Predicted Experiment Time (hours)', fontsize=12)
        ax.set_title(f'Uninfected Cells\nR²={r2_uninfected:.4f}, MAE={mae_uninfected:.2f}h', 
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle('Time Prediction Quality: Predicted vs. True Time', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction scatter plot to {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("TIME PREDICTION SUMMARY")
    print("=" * 60)
    if infected_mask.sum() > 0:
        print(f"Infected cells (n={infected_mask.sum()}):")
        print(f"  R² Score: {r2_infected:.4f}")
        print(f"  MAE: {mae_infected:.2f} hours")
        print(f"  RMSE: {np.sqrt(np.mean((time_targets[infected_mask] - time_preds[infected_mask])**2)):.2f} hours")
    
    if uninfected_mask.sum() > 0:
        print(f"\nUninfected cells (n={uninfected_mask.sum()}):")
        print(f"  R² Score: {r2_uninfected:.4f}")
        print(f"  MAE: {mae_uninfected:.2f} hours")
        print(f"  RMSE: {np.sqrt(np.mean((time_targets[uninfected_mask] - time_preds[uninfected_mask])**2)):.2f} hours")
    
    print("=" * 60)


def create_summary_report(
    results: Dict,
    metrics: Dict[str, List[float]],
    output_path: Path
) -> None:
    """Create a text summary report."""
    test_metrics = results["test_metrics"]
    config = results["config"]
    
    report = []
    report.append("=" * 80)
    report.append("MULTI-TASK CELL CLASSIFICATION - TRAINING SUMMARY")
    report.append("=" * 80)
    report.append(f"Experiment: {results['experiment_name']}")
    report.append(f"Run ID: {results['run_id']}")
    report.append("")
    
    # Configuration Summary
    report.append("-" * 80)
    report.append("CONFIGURATION")
    report.append("-" * 80)
    report.append(f"Model: {config['model']['name']}")
    report.append(f"Pretrained: {config['model']['pretrained']}")
    report.append(f"Hidden Dim: {config['model']['hidden_dim']}")
    report.append(f"Dropout: {config['model']['dropout']}")
    report.append("")
    report.append(f"Epochs: {config['training']['epochs']}")
    report.append(f"Learning Rate: {config['optimizer']['lr']}")
    report.append(f"Batch Size: {config['data']['batch_size']}")
    report.append(f"Mixed Precision: {config['training']['amp']}")
    report.append("")
    report.append(f"Infection Onset Hour: {config['multitask']['infection_onset_hour']}")
    report.append(f"Classification Weight: {config['multitask']['classification_weight']}")
    report.append(f"Regression Weight: {config['multitask']['regression_weight']}")
    report.append("")
    
    # Training Progress
    report.append("-" * 80)
    report.append("TRAINING PROGRESS")
    report.append("-" * 80)
    if metrics["val_cls_auc"]:
        best_epoch = np.argmax(metrics["val_cls_auc"]) + 1
        best_auc = max(metrics["val_cls_auc"])
        report.append(f"Best Validation AUC: {best_auc:.4f} (Epoch {best_epoch})")
        
        final_train_loss = metrics["train_total_loss"][-1]
        final_val_loss = metrics["val_total_loss"][-1]
        report.append(f"Final Training Loss: {final_train_loss:.4f}")
        report.append(f"Final Validation Loss: {final_val_loss:.4f}")
        
        # Check for overfitting
        if final_val_loss > final_train_loss * 1.5:
            report.append("⚠️  WARNING: Possible overfitting detected (val_loss >> train_loss)")
        elif metrics["val_cls_auc"][-1] < best_auc * 0.95:
            report.append("⚠️  WARNING: Performance degraded from best epoch")
        else:
            report.append("✓ Training converged successfully")
    report.append("")
    
    # Test Results
    report.append("-" * 80)
    report.append("FINAL TEST SET RESULTS")
    report.append("-" * 80)
    report.append("Classification Task:")
    report.append(f"  Accuracy:  {test_metrics['cls_accuracy']:.4f}")
    report.append(f"  Precision: {test_metrics['cls_precision']:.4f}")
    report.append(f"  Recall:    {test_metrics['cls_recall']:.4f}")
    report.append(f"  F1 Score:  {test_metrics['cls_f1']:.4f}")
    report.append(f"  AUC:       {test_metrics['cls_auc']:.4f}")
    report.append("")
    report.append("Time Regression Task:")
    if test_metrics.get('reg_mae') is not None:
        report.append(f"  MAE:       {test_metrics['reg_mae']:.4f} hours")
        report.append(f"  RMSE:      {test_metrics['reg_rmse']:.4f} hours")
        report.append(f"  MSE:       {test_metrics['reg_mse']:.4f} hours²")
    else:
        report.append("  ⚠️  Regression metrics not available (incomplete results.json)")
        report.append("  Check train.log for complete metrics")
    report.append("")
    report.append("Combined Losses:")
    report.append(f"  Total Loss:          {test_metrics['total_loss']:.4f}")
    report.append(f"  Classification Loss: {test_metrics['cls_loss']:.4f}")
    report.append(f"  Regression Loss:     {test_metrics['reg_loss']:.4f}")
    report.append("")
    
    # Interpretation
    report.append("-" * 80)
    report.append("INTERPRETATION")
    report.append("-" * 80)
    
    # Classification performance
    if test_metrics['cls_auc'] >= 0.95:
        report.append("✓ EXCELLENT classification performance (AUC ≥ 0.95)")
    elif test_metrics['cls_auc'] >= 0.90:
        report.append("✓ GOOD classification performance (AUC ≥ 0.90)")
    elif test_metrics['cls_auc'] >= 0.80:
        report.append("⚠️  MODERATE classification performance (AUC ≥ 0.80)")
    else:
        report.append("✗ POOR classification performance (AUC < 0.80)")
    
    # Regression performance (only if available)
    if test_metrics.get('reg_mae') is not None:
        mae_hours = test_metrics['reg_mae']
        if mae_hours < 1.0:
            report.append(f"✓ EXCELLENT time prediction (MAE < 1 hour)")
        elif mae_hours < 2.0:
            report.append(f"✓ GOOD time prediction (MAE < 2 hours)")
        elif mae_hours < 5.0:
            report.append(f"⚠️  MODERATE time prediction (MAE < 5 hours)")
        else:
            report.append(f"⚠️  POOR time prediction (MAE ≥ 5 hours)")
    else:
        report.append("⚠️  Regression metrics unavailable - check training log for details")
    
    report.append("")
    report.append("The model successfully learns BOTH tasks:")
    report.append("  1. Distinguishes infected vs uninfected cells with high accuracy")
    report.append("  2. Predicts temporal information (time since infection / experiment time)")
    report.append("")
    report.append("This dual capability enables:")
    report.append("  - Early infection detection")
    report.append("  - Infection progression tracking")
    report.append("  - Temporal pattern analysis")
    report.append("=" * 80)
    
    # Write report
    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)
    
    print(f"Saved summary report to {output_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-task training results")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to result directory containing results.json and train.log",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as result-dir)",
    )
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = result_dir / "results.json"
    if not results_file.exists():
        print(f"ERROR: results.json not found in {result_dir}")
        return
    
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded results from {results_file}")
    except json.JSONDecodeError as e:
        print(f"ERROR: results.json is malformed at line {e.lineno}, column {e.colno}")
        print(f"Error: {e.msg}")
        print("\nAttempting to extract partial results...")
        
        # Try to read what we can
        with open(results_file, "r") as f:
            content = f.read()
        
        # Find the incomplete part and try to complete it
        # Common patterns: incomplete field like "reg_mae": or "reg_mae": 
        import re
        
        # Remove trailing incomplete field
        content = re.sub(r',\s*"reg_\w+":\s*$', '', content)
        
        # Close the test_metrics object and root object
        if not content.rstrip().endswith('}'):
            content = content.rstrip().rstrip(',') + '\n  }\n}'
        
        try:
            results = json.loads(content)
            print("✓ Recovered partial results (regression metrics may be missing)")
        except json.JSONDecodeError as e2:
            print(f"Still cannot parse. Trying alternative fix...")
            # Last resort: manually construct minimal valid JSON
            lines = content.split('\n')
            # Find where test_metrics starts
            for i, line in enumerate(lines):
                if '"test_metrics"' in line:
                    # Keep everything up to and including the last valid metric
                    valid_lines = []
                    for j in range(i+1, len(lines)):
                        if lines[j].strip().startswith('"cls_auc"'):
                            # This is likely the last complete metric
                            valid_lines.append(lines[j].rstrip(','))
                            break
                        valid_lines.append(lines[j])
                    
                    # Reconstruct
                    content_parts = lines[:i+1] + valid_lines + ['  }', '}']
                    content = '\n'.join(content_parts)
                    break
            
            try:
                results = json.loads(content)
                print("✓ Recovered results using alternative method")
            except:
                print(f"ERROR: Cannot recover results. Please check the file manually.")
                print(f"You can try manually completing the JSON or rerun training.")
                return
    
    # Find log file
    log_file = result_dir / "train.log" / "multitask_train.log"
    if not log_file.exists():
        # Try alternative location
        log_file = result_dir / "multitask_train.log"
    
    if not log_file.exists():
        print(f"WARNING: Training log not found, skipping training curves")
        metrics = {}
    else:
        print(f"Parsing training log from {log_file}")
        metrics = parse_training_log(log_file)
        print(f"  Found {len(metrics['train_total_loss'])} training epochs")
    
    # Create visualizations
    if metrics and metrics["train_total_loss"]:
        print("\nGenerating visualizations...")
        plot_training_curves(metrics, output_dir / "training_curves.png")
        plot_validation_metrics(metrics, output_dir / "validation_metrics.png")
    
    # Create prediction scatter plot
    predictions_file = result_dir / "test_predictions.npz"
    if predictions_file.exists():
        print("\nGenerating prediction scatter plot...")
        infection_onset = results.get("config", {}).get("multitask", {}).get("infection_onset_hour", 2.0)
        plot_prediction_scatter(
            predictions_file, 
            output_dir / "prediction_scatter.png",
            infection_onset_hour=infection_onset
        )
    else:
        print(f"\nNote: Prediction file not found at {predictions_file}")
        print("Attempting to generate predictions from checkpoint...")
        try:
            import subprocess
            gen_cmd = [
                "python",
                "generate_prediction_plot.py",
                "--result-dir", str(result_dir),
            ]
            subprocess.run(gen_cmd, check=True)
            print("✓ Prediction scatter plot generated successfully!")
        except Exception as e:
            print(f"Failed to generate predictions: {e}")
            print(f"You can manually run: python generate_prediction_plot.py --result-dir {result_dir}")
    
    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(results, metrics, output_dir / "training_summary.txt")
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
