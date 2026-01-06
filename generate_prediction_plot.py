"""
Generate Prediction Scatter Plot from Trained Multitask Model

This script loads a trained checkpoint and generates predictions on the test set,
then creates a scatter plot with regression line showing predicted vs. true time.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import linregress
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_datasets
from models import build_multitask_model
from utils.config import load_config


def meta_batch_to_list(meta_batch: Any) -> List[Dict[str, Any]]:
    """Convert batch metadata to list of dictionaries."""
    if isinstance(meta_batch, list):
        return meta_batch
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        if not keys:
            return []
        length = len(meta_batch[keys[0]])
        meta_list: List[Dict[str, Any]] = []
        for i in range(length):
            entry: Dict[str, Any] = {}
            for key in keys:
                value = meta_batch[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                entry[key] = value
            meta_list.append(entry)
        return meta_list
    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")


def build_multitask_targets(
    labels: torch.Tensor,
    meta_list: List[Dict[str, Any]],
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build targets for multi-task learning.
    
    Args:
        labels: Binary labels (0=uninfected, 1=infected)
        meta_list: List of metadata dicts with 'hours_since_start'
        infection_onset_hour: Hour when infection occurs (e.g., 2.0)
        clamp_range: (min, max) for clamping time values
        device: Target device
        
    Returns:
        cls_targets: Class indices [batch_size] for CrossEntropyLoss
        time_targets: Regression targets [batch_size, 1]
            - For infected: time since infection onset
            - For uninfected: elapsed time from experiment start
    """
    clamp_min, clamp_max = clamp_range
    
    # Classification targets: class indices for CrossEntropyLoss
    cls_targets = labels.long().to(device, non_blocking=True)
    
    # Regression targets: different time references
    time_list: List[float] = []
    for label, meta in zip(labels.tolist(), meta_list):
        hours = float(meta.get("hours_since_start", 0.0))
        
        if int(label) == 1:  # Infected
            # Time since infection onset (how long infected)
            time_value = max(hours - infection_onset_hour, 0.0)
        else:  # Uninfected
            # Elapsed time from experiment start (preserves temporal info!)
            time_value = hours
        
        # Apply clamping
        if clamp_min is not None:
            time_value = max(time_value, clamp_min)
        if clamp_max is not None:
            time_value = min(time_value, clamp_max)
        
        time_list.append(time_value)
    
    time_targets = torch.tensor(time_list, dtype=torch.float32, device=device).unsqueeze(1)
    
    return cls_targets, time_targets


def load_checkpoint_and_config(checkpoint_path: Path) -> Tuple[Dict, nn.Module, torch.device]:
    """Load checkpoint and create model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    
    # Build model
    model_cfg = config.get("model", {})
    model = build_multitask_model(model_cfg)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model: {model_cfg.get('name', 'resnet50')}")
    print(f"✓ Device: {device}")
    
    return config, model, device


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    """Run model on test set and collect predictions."""
    model.eval()
    
    cls_preds_list = []
    cls_targets_list = []
    time_preds_list = []
    time_targets_list = []
    
    print("Collecting predictions on test set...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            # Handle both tuple and dict formats
            if isinstance(batch, (tuple, list)):
                images, labels, meta = batch
            else:
                images = batch["image"]
                labels = batch["label"]
                meta = batch.get("meta", {})
            
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            
            # Build targets
            cls_targets, time_targets = build_multitask_targets(
                labels, meta_list, infection_onset_hour, clamp_range, device
            )
            
            # Forward pass
            cls_logits, time_pred = model(images)
            
            # Convert to probabilities
            cls_probs = torch.softmax(cls_logits, dim=1)[:, 1]  # Prob of class 1
            
            # Collect
            cls_preds_list.append(cls_probs.cpu().numpy())
            cls_targets_list.append(cls_targets.cpu().numpy())
            time_preds_list.append(time_pred.detach().cpu().numpy().squeeze(-1))
            time_targets_list.append(time_targets.cpu().numpy().squeeze(-1))
    
    predictions = {
        "cls_preds": np.concatenate(cls_preds_list, axis=0),
        "cls_targets": np.concatenate(cls_targets_list, axis=0),
        "time_preds": np.concatenate(time_preds_list, axis=0),
        "time_targets": np.concatenate(time_targets_list, axis=0),
    }
    
    print(f"✓ Collected {len(predictions['time_preds'])} predictions")
    return predictions


def plot_scatter_with_regression(
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    infection_onset_hour: float = 2.0,
) -> None:
    """Create scatter plot with regression lines for predicted vs. true time."""
    time_preds = predictions["time_preds"]
    time_targets = predictions["time_targets"]
    cls_targets = predictions["cls_targets"]
    
    # Separate by class
    infected_mask = cls_targets == 1
    uninfected_mask = cls_targets == 0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ========== Plot 1: All Samples ==========
    ax = axes[0]
    
    # Scatter infected
    if infected_mask.sum() > 0:
        ax.scatter(
            time_targets[infected_mask],
            time_preds[infected_mask],
            alpha=0.4,
            s=30,
            c='red',
            edgecolors='darkred',
            linewidth=0.3,
            label=f'Infected (n={infected_mask.sum()})',
        )
    
    # Scatter uninfected
    if uninfected_mask.sum() > 0:
        ax.scatter(
            time_targets[uninfected_mask],
            time_preds[uninfected_mask],
            alpha=0.4,
            s=30,
            c='blue',
            edgecolors='darkblue',
            linewidth=0.3,
            label=f'Uninfected (n={uninfected_mask.sum()})',
        )
    
    # Perfect prediction line (y=x)
    min_val = min(time_targets.min(), time_preds.min())
    max_val = max(time_targets.max(), time_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5, label='Perfect Prediction', zorder=10)
    
    # Overall regression line
    slope_all, intercept_all, r_all, _, _ = linregress(time_targets, time_preds)
    regression_line = slope_all * time_targets + intercept_all
    # Sort for smooth line
    sort_idx = np.argsort(time_targets)
    ax.plot(
        time_targets[sort_idx],
        regression_line[sort_idx],
        'g-',
        linewidth=2.5,
        label=f'Regression (R={r_all:.3f})',
        zorder=9,
    )
    
    # Infection onset marker
    ax.axvline(
        x=infection_onset_hour,
        color='orange',
        linestyle=':',
        linewidth=2,
        label=f'Infection Onset ({infection_onset_hour}h)',
        zorder=8,
    )
    
    # Metrics
    r2_all = r2_score(time_targets, time_preds)
    mae_all = np.mean(np.abs(time_targets - time_preds))
    rmse_all = np.sqrt(np.mean((time_targets - time_preds) ** 2))
    
    ax.set_xlabel('True Time (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Time (hours)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'All Samples\nR²={r2_all:.4f}, MAE={mae_all:.2f}h, RMSE={rmse_all:.2f}h',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    # ========== Plot 2: Infected Cells Only ==========
    if infected_mask.sum() > 0:
        ax = axes[1]
        
        time_targets_inf = time_targets[infected_mask]
        time_preds_inf = time_preds[infected_mask]
        
        # Scatter
        ax.scatter(
            time_targets_inf,
            time_preds_inf,
            alpha=0.5,
            s=40,
            c='red',
            edgecolors='darkred',
            linewidth=0.5,
        )
        
        # Perfect prediction line
        min_inf = min(time_targets_inf.min(), time_preds_inf.min())
        max_inf = max(time_targets_inf.max(), time_preds_inf.max())
        ax.plot([min_inf, max_inf], [min_inf, max_inf], 'k--', linewidth=2.5, label='Perfect Prediction', zorder=10)
        
        # Regression line
        slope_inf, intercept_inf, r_inf, _, _ = linregress(time_targets_inf, time_preds_inf)
        regression_line_inf = slope_inf * time_targets_inf + intercept_inf
        sort_idx_inf = np.argsort(time_targets_inf)
        ax.plot(
            time_targets_inf[sort_idx_inf],
            regression_line_inf[sort_idx_inf],
            'darkred',
            linewidth=3,
            label=f'Regression (R={r_inf:.3f})',
            zorder=9,
        )
        
        # Metrics
        r2_inf = r2_score(time_targets_inf, time_preds_inf)
        mae_inf = np.mean(np.abs(time_targets_inf - time_preds_inf))
        rmse_inf = np.sqrt(np.mean((time_targets_inf - time_preds_inf) ** 2))
        
        ax.set_xlabel('True Time Since Infection (hours)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Time Since Infection (hours)', fontsize=13, fontweight='bold')
        ax.set_title(
            f'Infected Cells Only (n={infected_mask.sum()})\n'
            f'R²={r2_inf:.4f}, MAE={mae_inf:.2f}h, RMSE={rmse_inf:.2f}h\n'
            f'Regression: y = {slope_inf:.3f}x + {intercept_inf:.3f}',
            fontsize=13,
            fontweight='bold',
        )
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    # ========== Plot 3: Uninfected Cells Only ==========
    if uninfected_mask.sum() > 0:
        ax = axes[2]
        
        time_targets_uninf = time_targets[uninfected_mask]
        time_preds_uninf = time_preds[uninfected_mask]
        
        # Scatter
        ax.scatter(
            time_targets_uninf,
            time_preds_uninf,
            alpha=0.5,
            s=40,
            c='blue',
            edgecolors='darkblue',
            linewidth=0.5,
        )
        
        # Perfect prediction line
        min_uninf = min(time_targets_uninf.min(), time_preds_uninf.min())
        max_uninf = max(time_targets_uninf.max(), time_preds_uninf.max())
        ax.plot(
            [min_uninf, max_uninf],
            [min_uninf, max_uninf],
            'k--',
            linewidth=2.5,
            label='Perfect Prediction',
            zorder=10,
        )
        
        # Regression line
        slope_uninf, intercept_uninf, r_uninf, _, _ = linregress(time_targets_uninf, time_preds_uninf)
        regression_line_uninf = slope_uninf * time_targets_uninf + intercept_uninf
        sort_idx_uninf = np.argsort(time_targets_uninf)
        ax.plot(
            time_targets_uninf[sort_idx_uninf],
            regression_line_uninf[sort_idx_uninf],
            'darkblue',
            linewidth=3,
            label=f'Regression (R={r_uninf:.3f})',
            zorder=9,
        )
        
        # Metrics
        r2_uninf = r2_score(time_targets_uninf, time_preds_uninf)
        mae_uninf = np.mean(np.abs(time_targets_uninf - time_preds_uninf))
        rmse_uninf = np.sqrt(np.mean((time_targets_uninf - time_preds_uninf) ** 2))
        
        ax.set_xlabel('True Experiment Time (hours)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Predicted Experiment Time (hours)', fontsize=13, fontweight='bold')
        ax.set_title(
            f'Uninfected Cells Only (n={uninfected_mask.sum()})\n'
            f'R²={r2_uninf:.4f}, MAE={mae_uninf:.2f}h, RMSE={rmse_uninf:.2f}h\n'
            f'Regression: y = {slope_uninf:.3f}x + {intercept_uninf:.3f}',
            fontsize=13,
            fontweight='bold',
        )
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(
        'Time Prediction Analysis: Predicted vs. True Time with Regression Lines',
        fontsize=16,
        fontweight='bold',
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved scatter plot to {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREDICTION ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Overall (n={len(time_targets)}):")
    print(f"  R² Score:  {r2_all:.4f}")
    print(f"  Pearson R: {r_all:.4f}")
    print(f"  MAE:       {mae_all:.2f} hours")
    print(f"  RMSE:      {rmse_all:.2f} hours")
    print(f"  Regression: y = {slope_all:.3f}x + {intercept_all:.3f}")
    
    if infected_mask.sum() > 0:
        print(f"\nInfected Cells (n={infected_mask.sum()}):")
        print(f"  R² Score:  {r2_inf:.4f}")
        print(f"  Pearson R: {r_inf:.4f}")
        print(f"  MAE:       {mae_inf:.2f} hours")
        print(f"  RMSE:      {rmse_inf:.2f} hours")
        print(f"  Regression: y = {slope_inf:.3f}x + {intercept_inf:.3f}")
    
    if uninfected_mask.sum() > 0:
        print(f"\nUninfected Cells (n={uninfected_mask.sum()}):")
        print(f"  R² Score:  {r2_uninf:.4f}")
        print(f"  Pearson R: {r_uninf:.4f}")
        print(f"  MAE:       {mae_uninf:.2f} hours")
        print(f"  RMSE:      {rmse_uninf:.2f} hours")
        print(f"  Regression: y = {slope_uninf:.3f}x + {intercept_uninf:.3f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate prediction scatter plot from trained model")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to result directory (e.g., outputs/multitask_resnet50/20251215-164539)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pt",
        help="Checkpoint filename (default: best.pt)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="prediction_scatter_regression.png",
        help="Output plot filename",
    )
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    checkpoint_path = result_dir / "checkpoints" / args.checkpoint_name
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint and config
    config, model, device = load_checkpoint_and_config(checkpoint_path)
    
    # Get multitask config
    mt_cfg = config.get("multitask", {})
    infection_onset_hour = float(mt_cfg.get("infection_onset_hour", 2.0))
    clamp_range = tuple(mt_cfg.get("clamp_range", [0.0, 48.0]))
    
    print(f"\nMultitask config:")
    print(f"  Infection onset: {infection_onset_hour}h")
    print(f"  Clamp range: {clamp_range}")
    
    # Build datasets (we only need test set)
    from utils.transforms import build_transforms
    
    data_cfg = config.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    print("\nBuilding datasets...")
    train_dataset, val_dataset, test_dataset = build_datasets(
        data_cfg, transforms_dict
    )
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get("batch_size", 128) * data_cfg.get("eval_batch_size_multiplier", 2),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    # Collect predictions
    predictions = collect_predictions(
        model, test_loader, device, infection_onset_hour, clamp_range
    )
    
    # Save predictions
    predictions_file = result_dir / "test_predictions.npz"
    np.savez(predictions_file, **predictions)
    print(f"✓ Saved predictions to {predictions_file}")
    
    # Create scatter plot
    output_path = result_dir / args.output_name
    plot_scatter_with_regression(predictions, output_path, infection_onset_hour)
    
    print(f"\n✓ Done! Check {output_path}")


if __name__ == "__main__":
    main()
