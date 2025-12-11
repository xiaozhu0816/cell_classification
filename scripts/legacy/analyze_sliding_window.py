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
        default=None,
        help="Starting hours x for windows [x, x+k]. E.g., 0 2 4 6 8 10. "
             "If not provided, windows are auto-generated using --start-hour, --end-hour, and --stride",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=1.0,
        help="Starting hour for first window (used when --window-starts not provided)",
    )
    parser.add_argument(
        "--end-hour",
        type=float,
        default=30.0,
        help="Maximum ending hour (used when --window-starts not provided)",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=None,
        help="Step size between consecutive windows (hours). "
             "Default is window-size (no overlap). "
             "stride < window-size creates overlap, stride > window-size creates gaps",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split used for evaluation",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to plot (e.g., auc accuracy f1). "
             "If multiple provided, creates combined plot + individual plots. "
             "If not provided, uses --metric for backward compatibility",
    )
    parser.add_argument(
        "--metric",
        default="auc",
        help="Single metric key to summarize (e.g., auc, accuracy, f1). "
             "Use --metrics for multiple metrics",
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
    metric_keys: List[str],
    task_cfg: Dict,
    analysis_cfg: Dict,
    weights_only: bool,
) -> Dict[str, Tuple[List[float], float | None, float | None]]:
    """
    Evaluate model on data from window [window_start, window_end].
    Both train and test data are from this same window.
    
    Args:
        metric_keys: List of metric names to extract
    
    Returns:
        Dictionary mapping metric_key -> (fold_metrics, mean, std)
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

    # Initialize storage for each metric
    fold_metrics_dict: Dict[str, List[float]] = {key: [] for key in metric_keys}
    
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
        
        # Collect all requested metrics
        for metric_key in metric_keys:
            metric_value = metrics.get(metric_key)
            
            if metric_value is not None and not math.isnan(metric_value):
                fold_metrics_dict[metric_key].append(float(metric_value))
        
        # Cleanup
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute statistics for each metric
    results = {}
    for metric_key in metric_keys:
        fold_values = fold_metrics_dict[metric_key]
        if fold_values:
            values = np.asarray(fold_values, dtype=np.float32)
            results[metric_key] = (fold_values, float(values.mean()), float(values.std(ddof=0)))
        else:
            results[metric_key] = ([], None, None)
    
    return results


def plot_single_metric(
    window_centers: List[float],
    means: List[float],
    stds: List[float],
    window_size: float,
    metric: str,
    output_path: Path,
) -> None:
    """
    Create an error bar plot for a single metric.
    
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
        logging.warning("No valid data points to plot for metric '%s'", metric)
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
    logging.info("Saved plot for %s to %s", metric, output_path)


def plot_multi_metric(
    window_centers: List[float],
    metric_results: Dict[str, Tuple[List[float], List[float]]],  # metric -> (means, stds)
    window_size: float,
    output_path: Path,
) -> None:
    """
    Create a combined error bar plot for multiple metrics.
    
    Args:
        window_centers: Center point of each window (x + k/2)
        metric_results: Dictionary mapping metric name to (means, stds) tuples
        window_size: Width of each window (k)
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    any_valid = False
    for idx, (metric, (means, stds)) in enumerate(metric_results.items()):
        # Filter out None values
        valid_points = [
            (center, mean, std)
            for center, mean, std in zip(window_centers, means, stds)
            if mean is not None
        ]
        
        if not valid_points:
            logging.warning("No valid data points for metric '%s'", metric)
            continue
        
        any_valid = True
        v_centers, v_means, v_stds = zip(*valid_points)
        
        # Create error bar plot
        marker = markers[idx % len(markers)]
        ax.errorbar(
            v_centers,
            v_means,
            yerr=v_stds,
            fmt=f"-{marker}",
            capsize=4,
            linewidth=1.5,
            markersize=7,
            color=colors[idx],
            label=metric.upper(),
            alpha=0.8,
        )
    
    if not any_valid:
        logging.warning("No valid data to plot for any metric")
        plt.close(fig)
        return
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"Sliding Window Analysis: Multiple Metrics\n"
        f"(Train & Test both use [x, x+{window_size}h])",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="best")
    
    # Add window interval annotations on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Window Start Hour (x)", fontsize=11, color="gray")
    ax2.tick_params(axis="x", labelcolor="gray")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved combined multi-metric plot to %s", output_path)


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
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    # Determine stride (step size)
    stride = args.stride if args.stride is not None else window_size
    if stride <= 0:
        raise ValueError("Stride must be positive")
    
    # Generate or use provided window starts
    if args.window_starts is not None:
        window_starts = sorted(args.window_starts)
        logging.info("Using %d user-provided window start positions", len(window_starts))
    else:
        # Auto-generate window starts using stride
        window_starts = []
        current = args.start_hour
        max_end = args.max_hour if args.max_hour is not None else args.end_hour
        
        while current + window_size <= max_end:
            window_starts.append(current)
            current += stride
        
        logging.info(
            "Auto-generated %d windows: start=%.1fh, end=%.1fh, size=%.1fh, stride=%.1fh",
            len(window_starts),
            args.start_hour,
            max_end,
            window_size,
            stride,
        )
        
        if not window_starts:
            raise ValueError(
                f"No valid windows with start={args.start_hour}, "
                f"end={max_end}, size={window_size}"
            )
    
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
    
    # Determine which metrics to evaluate
    if args.metrics:
        metric_keys = args.metrics
        logging.info("Evaluating metrics: %s", ", ".join(metric_keys))
    else:
        metric_keys = [args.metric]
        logging.info("Evaluating single metric: %s", args.metric)
    
    logging.info(
        "Evaluating %d sliding windows with size=%.1fh, stride=%.1fh",
        len(windows),
        window_size,
        stride,
    )
    
    # Evaluate each window
    # Structure: results[metric_key] = {"window_starts": [...], "means": [...], ...}
    results: Dict[str, Dict[str, List]] = {
        metric_key: {
            "window_starts": [],
            "window_ends": [],
            "window_centers": [],
            "fold_metrics": [],
            "means": [],
            "stds": [],
        }
        for metric_key in metric_keys
    }
    
    for start, end, center in windows:
        logging.info("Evaluating window [%.1f, %.1f]h (center=%.1f)", start, end, center)
        
        window_results = evaluate_window(
            cfg,
            base_data_cfg,
            transforms,
            run_dir,
            device,
            fold_count,
            start,
            end,
            args.split,
            metric_keys,
            task_cfg,
            analysis_cfg,
            args.weights_only,
        )
        
        # Store results for each metric
        for metric_key in metric_keys:
            fold_values, mean_value, std_value = window_results[metric_key]
            
            results[metric_key]["window_starts"].append(start)
            results[metric_key]["window_ends"].append(end)
            results[metric_key]["window_centers"].append(center)
            results[metric_key]["fold_metrics"].append(fold_values)
            results[metric_key]["means"].append(mean_value)
            results[metric_key]["stds"].append(std_value)
            
            if mean_value is not None:
                logging.info(
                    "  %s = %.4f ± %.4f",
                    metric_key.upper(),
                    mean_value,
                    std_value,
                )
        
        if all(window_results[mk][1] is None for mk in metric_keys):
            logging.warning("  Window [%.1f, %.1f]h: No valid metrics", start, end)
    
    # Generate output
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename suffix
    stride_suffix = f"_s{stride:.0f}" if stride != window_size else ""
    base_filename = f"sliding_window_w{window_size:.0f}{stride_suffix}"
    
    # If multiple metrics, create combined plot
    if len(metric_keys) > 1:
        logging.info("Creating combined plot for all metrics...")
        combined_plot_path = output_dir / f"{base_filename}_combined.png"
        
        # Prepare data for multi-metric plot
        metric_plot_data = {}
        for metric_key in metric_keys:
            metric_plot_data[metric_key] = (
                results[metric_key]["means"],
                results[metric_key]["stds"],
            )
        
        # Use window_centers from first metric (all should be the same)
        window_centers = results[metric_keys[0]]["window_centers"]
        plot_multi_metric(window_centers, metric_plot_data, window_size, combined_plot_path)
    
    # Create individual plots for each metric
    for metric_key in metric_keys:
        metric_result = results[metric_key]
        plot_path = output_dir / f"{base_filename}_{metric_key}.png"
        
        plot_single_metric(
            metric_result["window_centers"],
            metric_result["means"],
            metric_result["stds"],
            window_size,
            metric_key,
            plot_path,
        )
    
    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / f"{base_filename}_data.json"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "config": args.config,
        "run_dir": str(run_dir),
        "window_size": window_size,
        "stride": stride,
        "window_starts": window_starts,
        "max_hour": args.max_hour,
        "metrics": metric_keys,
        "split": args.split,
        "results": results,
    }
    
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    
    logging.info("Saved sliding window data to %s", save_path)
    
    # Summary statistics for each metric
    logging.info("\n" + "="*60)
    logging.info("Summary Statistics:")
    logging.info("="*60)
    
    for metric_key in metric_keys:
        metric_result = results[metric_key]
        valid_means = [m for m in metric_result["means"] if m is not None]
        
        if valid_means:
            best_idx = metric_result["means"].index(max(valid_means))
            logging.info(f"\n{metric_key.upper()}:")
            logging.info("  Best window: [%.1f, %.1f]h with value=%.4f",
                        metric_result["window_starts"][best_idx],
                        metric_result["window_ends"][best_idx],
                        max(valid_means))
            logging.info("  Overall mean: %.4f ± %.4f",
                        np.mean(valid_means),
                        np.std(valid_means, ddof=1 if len(valid_means) > 1 else 0))
        else:
            logging.warning(f"\n{metric_key.upper()}: No valid data")
    
    logging.info("="*60)


if __name__ == "__main__":
    main()
