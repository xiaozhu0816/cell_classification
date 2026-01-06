"""
Plot combined chart from existing data:
- Left: train-test from original JSON
- Right: test-only from eval log file
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_metrics(log_path):
    """Parse log file to extract test-only metrics from individual fold results."""
    
    metrics_data = {
        'auc': {'mean': [], 'std': []},
        'accuracy': {'mean': [], 'std': []},
        'f1': {'mean': [], 'std': []}
    }
    
    hours = []
    current_hour = None
    current_fold_metrics = {
        'auc': [],
        'accuracy': [],
        'f1': []
    }
    fold_count = 0
    expected_folds = 5  # We know there are 5 folds
    
    with open(log_path, 'r') as f:
        for line in f:
            # Match: [Interval [1.0, 7.0]h, mode=test-only] Evaluating fold_XX
            interval_match = re.search(r'\[Interval \[[\d.]+, ([\d.]+)\]h, mode=test-only\] Evaluating fold_(\d+)', line)
            if interval_match:
                new_hour = float(interval_match.group(1))
                fold_num = int(interval_match.group(2))
                
                # If this is a new interval (hour changed)
                if new_hour != current_hour:
                    # Save previous interval if we have all folds
                    if current_hour is not None and fold_count == expected_folds:
                        hours.append(current_hour)
                        for m in ['auc', 'accuracy', 'f1']:
                            if len(current_fold_metrics[m]) == expected_folds:
                                mean_val = np.mean(current_fold_metrics[m])
                                std_val = np.std(current_fold_metrics[m], ddof=0)
                                metrics_data[m]['mean'].append(float(mean_val))
                                metrics_data[m]['std'].append(float(std_val))
                            else:
                                metrics_data[m]['mean'].append(None)
                                metrics_data[m]['std'].append(None)
                    
                    # Start new interval
                    current_hour = new_hour
                    current_fold_metrics = {'auc': [], 'accuracy': [], 'f1': []}
                    fold_count = 0
                
                continue
            
            # Match individual fold result: test: accuracy:0.9856 | precision:1.0000 | recall:0.9712 | f1:0.9854 | auc:0.9996
            fold_result_match = re.search(r'test:\s+accuracy:([\d.]+)\s+\|\s+precision:[\d.]+\s+\|\s+recall:[\d.]+\s+\|\s+f1:([\d.]+)\s+\|\s+auc:([\d.]+)', line)
            if fold_result_match and current_hour is not None:
                accuracy_val = float(fold_result_match.group(1))
                f1_val = float(fold_result_match.group(2))
                auc_val = float(fold_result_match.group(3))
                
                current_fold_metrics['accuracy'].append(accuracy_val)
                current_fold_metrics['f1'].append(f1_val)
                current_fold_metrics['auc'].append(auc_val)
                fold_count += 1
    
    # Save last interval if we have all folds
    if current_hour is not None and fold_count == expected_folds:
        hours.append(current_hour)
        for m in ['auc', 'accuracy', 'f1']:
            if len(current_fold_metrics[m]) == expected_folds:
                mean_val = np.mean(current_fold_metrics[m])
                std_val = np.std(current_fold_metrics[m], ddof=0)
                metrics_data[m]['mean'].append(float(mean_val))
                metrics_data[m]['std'].append(float(std_val))
            else:
                metrics_data[m]['mean'].append(None)
                metrics_data[m]['std'].append(None)
    
    return hours, metrics_data


def main():
    # Paths
    original_json = Path("outputs/interval_sweep_analysis/20251212-145928/interval_sweep_data.json")
    test_only_log = Path("outputs/interval_sweep_analysis/20251212-145928_eval_20251222-160546/interval_sweep_train.log")
    output_dir = Path("outputs/interval_sweep_analysis/20251212-145928_combined_new")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train-test data
    print("Loading train-test data from JSON...")
    with open(original_json, 'r') as f:
        original_data = json.load(f)
    
    train_test_stats = original_data['stats']
    hours = original_data['hours']
    metrics = original_data['metrics']
    
    # Parse test-only data from log
    print("Parsing test-only data from log...")
    test_only_hours, test_only_stats = parse_log_metrics(test_only_log)
    
    print(f"✓ Train-test: {len(hours)} intervals")
    print(f"✓ Test-only: {len(test_only_hours)} intervals")
    print(f"✓ Metrics: {metrics}")
    
    # Create plot
    print("\nCreating combined plot...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for ax, mode in zip(axes, ["train-test", "test-only"]):
        if mode == "train-test":
            title = "Train=[start,x], Test=[start,x]"
        else:
            title = "Train=[start,MAX], Test=[start,x]"
        
        for idx, metric in enumerate(metrics):
            # Original JSON: stats[metric][mode]["mean"]
            # Test-only: test_only_stats[metric]["mean"]
            if mode == "train-test":
                means = np.array([m if m is not None else np.nan for m in train_test_stats[metric][mode]["mean"]])
                stds = np.array([s if s is not None else 0.0 for s in train_test_stats[metric][mode]["std"]])
            else:
                means = np.array([m if m is not None else np.nan for m in test_only_stats[metric]["mean"]])
                stds = np.array([s if s is not None else 0.0 for s in test_only_stats[metric]["std"]])
            
            ax.errorbar(
                hours, means, yerr=stds,
                marker='o', color=colors[idx], label=metric.upper(),
                capsize=5, capthick=2, linewidth=2, markersize=6
            )
        
        ax.set_xlabel("Upper Hour (x) for Interval [1, x]", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.5, 1.05])
    
    fig.suptitle("Interval Sweep Training Analysis (CORRECTED)", fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    output_path = output_dir / "interval_sweep_combined_new.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    
    # Save combined data
    combined_data = {
        "hours": hours,
        "metrics": metrics,
        "train-test": train_test_stats,
        "test-only": test_only_stats
    }
    
    json_path = output_dir / "combined_data.json"
    with open(json_path, "w") as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"✓ Saved: {json_path}")
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
