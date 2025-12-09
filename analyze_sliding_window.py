from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import build_datasets
from models import build_model
from utils import build_transforms, load_config
from train import get_analysis_config, get_task_config
from test_folds import build_loader, evaluate_loader, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze performance using sliding window intervals [x, x+k]"
    )
    parser.add_argument(
        "--config",
        default="configs/resnet50_baseline.yaml",
        help="Path to YAML config used for training",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Checkpoint run directory containing fold_* subfolders",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Window size k (hours). Both train and test use [x, x+k]",
    )
    parser.add_argument(
        "--window-starts",
        type=float,
        nargs="+",
        required=True,
        help="Starting hours x for windows [x, x+k]. E.g., 0 2 4 6 8 10",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for evaluation",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        help="Metric key to summarize (e.g., auc, accuracy, f1)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Load checkpoints with weights_only=True",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots and JSON summaries",
    )
    parser.add_argument(
        "--save-data",
        default=None,
        help="Optional path to save raw sweep metrics as JSON",
    )
    parser.add_argument(
        "--max-hour",
        type=float,
        default=None,
        help="Maximum hour to consider (optional). Windows extending beyond this are skipped.",
    )
    return parser.parse_args()


_SPLIT_OVERRIDE_KEYS = ("train", "val", "test", "default", "eval", "evaluation")


def _copy_section(section: Dict | None) -> Dict:
    if isinstance(section, dict):
        return copy.deepcopy(section)
    return {}


def _base_section(frames_cfg: Dict) -> Dict:
    base: Dict = {}
    for key, value in frames_cfg.items():
        if key not in _SPLIT_OVERRIDE_KEYS:
            base[key] = copy.deepcopy(value)
    return base


def apply_sliding_window(
    base_frames_cfg: Dict | None, start: float, end: float
) -> Dict:
    """
    Apply sliding window [start, end] to ALL splits (train, val, test).
    Both training and testing use the same time interval.
    """
    frames_cfg = copy.deepcopy(base_frames_cfg) if base_frames_cfg else {}
    
    # Set the base infected window
    frames_cfg["infected_window_hours"] = [start, end]
    
    # Override all split-specific configurations
    for key in _SPLIT_OVERRIDE_KEYS:
        if key in frames_cfg or key in ("train", "val", "test"):
            section = _copy_section(frames_cfg.get(key))
            section["infected_window_hours"] = [start, end]
            frames_cfg[key] = section
    
    return frames_cfg


def evaluate_window(
    cfg: Dict,
    base_data_cfg: Dict,
    transforms,
    run_dir: Path,
    device,
    fold_count: int,
    window_start: float,
    window_end: float,
    split: str,
    metric_key: str,
    task_cfg: Dict,
    analysis_cfg: Dict,
    weights_only: bool,
) -> Tuple[List[float], float | None, float | None]:
    """
    Evaluate model on data from window [window_start, window_end].
    Both train and test data are from this same window.
    
    Returns:
        fold_metrics: List of metric values for each fold
        mean: Mean across folds (or None if no valid metrics)
        std: Standard deviation across folds (or None if no valid metrics)
    """
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["frames"] = apply_sliding_window(
        base_data_cfg.get("frames"), window_start, window_end
    )
    
    batch_size = data_cfg.get("batch_size", 256)
    num_workers = data_cfg.get("num_workers", 4)

    fold_metrics: List[float] = []
    
    for fold_idx in range(fold_count):
        fold_name = f"fold_{fold_idx + 1:02d}of{fold_count:02d}"
        ckpt_path = run_dir / fold_name / "best.pt"
        
        if not ckpt_path.exists():
            logging.warning("[%s] Missing checkpoint: %s", fold_name, ckpt_path)
            continue

        # Build datasets with the windowed frame policy
        train_ds, val_ds, test_ds = build_datasets(
            data_cfg,
            transforms,
            fold_index=fold_idx if fold_count > 1 else None,
            num_folds=fold_count,
        )
        
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
        loader = build_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        # Load model and checkpoint
        model = build_model(cfg.get("model", {})).to(device)
        checkpoint = load_checkpoint(ckpt_path, device, weights_only)
        model.load_state_dict(checkpoint["model_state"])

        # Evaluate
        logger = logging.getLogger(f"sliding_window_{fold_name}")
        logger.setLevel(logging.WARNING)  # Reduce verbosity
        
        metrics = evaluate_loader(model, loader, task_cfg, analysis_cfg, device, logger)
        metric_value = metrics.get(metric_key)
        
        if metric_value is None or math.isnan(metric_value):
            logging.warning(
                "[%s] Metric '%s' unavailable for window [%.1f, %.1f]h",
                fold_name,
                metric_key,
                window_start,
                window_end,
            )
            continue
        
        fold_metrics.append(float(metric_value))
        
        # Cleanup
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not fold_metrics:
        return [], None, None
    
    values = np.asarray(fold_metrics, dtype=np.float32)
    return fold_metrics, float(values.mean()), float(values.std(ddof=0))


def plot_sliding_window_results(
    window_centers: List[float],
    means: List[float],
    stds: List[float],
    window_size: float,
    metric: str,
    output_path: Path,
) -> None:
    """
    Create an error bar plot showing metric performance for each sliding window.
    
    Args:
        window_centers: Center point of each window (x + k/2)
        means: Mean metric values
        stds: Standard deviation values
        window_size: Width of each window (k)
        metric: Name of the metric being plotted
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter out None values
    valid_points = [
        (center, mean, std)
        for center, mean, std in zip(window_centers, means, stds)
        if mean is not None
    ]
    
    if not valid_points:
        logging.warning("No valid data points to plot")
        plt.close(fig)
        return
    
    v_centers, v_means, v_stds = zip(*valid_points)
    
    # Create error bar plot
    ax.errorbar(
        v_centers,
        v_means,
        yerr=v_stds,
        fmt="-o",
        capsize=5,
        linewidth=2,
        markersize=8,
        label=f"Window size = {window_size}h",
    )
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"Sliding Window Analysis: {metric.upper()} vs Time Window\n"
        f"(Train & Test both use [x, x+{window_size}h])",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    
    # Add window interval annotations on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour (x)", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved sliding window plot to %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    # Load configuration
    cfg = load_config(args.config)
    task_cfg = get_task_config(cfg)
    analysis_cfg = get_analysis_config(cfg)
    data_cfg = cfg.get("data", {})
    base_data_cfg = copy.deepcopy(data_cfg)
    
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)
    
    training_cfg = cfg.get("training", {})
    fold_count = max(1, int(training_cfg.get("k_folds", 1)))
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = torch.device(args.device)
    
    # Validate window configurations
    window_size = args.window_size
    window_starts = sorted(args.window_starts)
    
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    # Build window intervals
    windows: List[Tuple[float, float, float]] = []  # (start, end, center)
    for start in window_starts:
        end = start + window_size
        
        # Skip if window extends beyond max_hour
        if args.max_hour is not None and end > args.max_hour:
            logging.info(
                "Skipping window [%.1f, %.1f]h (exceeds max_hour=%.1f)",
                start,
                end,
                args.max_hour,
            )
            continue
        
        center = start + window_size / 2.0
        windows.append((start, end, center))
    
    if not windows:
        raise ValueError("No valid windows to evaluate")
    
    logging.info(
        "Evaluating %d sliding windows with size=%.1fh",
        len(windows),
        window_size,
    )
    
    # Evaluate each window
    results: Dict[str, List] = {
        "window_starts": [],
        "window_ends": [],
        "window_centers": [],
        "fold_metrics": [],
        "means": [],
        "stds": [],
    }
    
    for start, end, center in windows:
        logging.info("Evaluating window [%.1f, %.1f]h (center=%.1f)", start, end, center)
        
        fold_values, mean_value, std_value = evaluate_window(
            cfg,
            base_data_cfg,
            transforms,
            run_dir,
            device,
            fold_count,
            start,
            end,
            args.split,
            args.metric,
            task_cfg,
            analysis_cfg,
            args.weights_only,
        )
        
        results["window_starts"].append(start)
        results["window_ends"].append(end)
        results["window_centers"].append(center)
        results["fold_metrics"].append(fold_values)
        results["means"].append(mean_value)
        results["stds"].append(std_value)
        
        if mean_value is not None:
            logging.info(
                "  Window [%.1f, %.1f]h: %s = %.4f ± %.4f",
                start,
                end,
                args.metric,
                mean_value,
                std_value,
            )
        else:
            logging.warning("  Window [%.1f, %.1f]h: No valid metrics", start, end)
    
    # Generate output
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    
    # Save plot
    plot_path = output_dir / f"sliding_window_k{window_size:.0f}_{args.metric}.png"
    plot_sliding_window_results(
        results["window_centers"],
        results["means"],
        results["stds"],
        window_size,
        args.metric,
        plot_path,
    )
    
    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / f"sliding_window_k{window_size:.0f}_{args.metric}.json"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "config": args.config,
        "run_dir": str(run_dir),
        "window_size": window_size,
        "window_starts": window_starts,
        "max_hour": args.max_hour,
        "metric": args.metric,
        "split": args.split,
        "results": results,
    }
    
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    
    logging.info("Saved sliding window data to %s", save_path)
    
    # Summary statistics
    valid_means = [m for m in results["means"] if m is not None]
    if valid_means:
        logging.info("\nSummary:")
        logging.info("  Best window: [%.1f, %.1f]h with %s=%.4f",
                    results["window_starts"][results["means"].index(max(valid_means))],
                    results["window_ends"][results["means"].index(max(valid_means))],
                    args.metric,
                    max(valid_means))
        logging.info("  Overall mean %s: %.4f ± %.4f",
                    args.metric,
                    np.mean(valid_means),
                    np.std(valid_means, ddof=1 if len(valid_means) > 1 else 0))


if __name__ == "__main__":
    main()
