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
    parser = argparse.ArgumentParser(description="Analyze performance across infected interval upper bounds")
    parser.add_argument("--config", default="configs/resnet50_baseline.yaml", help="Path to YAML config used for training")
    parser.add_argument("--run-dir", required=True, help="Checkpoint run directory containing fold_* subfolders")
    parser.add_argument("--upper-hours", type=float, nargs="+", required=True, help="Upper bounds X for intervals [start, X]")
    parser.add_argument("--start-hour", type=float, default=1.0, help="Lower bound for infected interval")
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
    parser.add_argument("--metric", default="auc", help="Single metric key to summarize (e.g., auc, accuracy, f1). Use --metrics for multiple metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights-only", action="store_true", help="Load checkpoints with weights_only=True")
    parser.add_argument("--output-dir", default=None, help="Directory to write plots and JSON summaries")
    parser.add_argument("--save-data", default=None, help="Optional path to save raw sweep metrics as JSON")
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


def apply_interval_override(base_frames_cfg: Dict | None, start: float, end: float, mode: str) -> Dict:
    frames_cfg = copy.deepcopy(base_frames_cfg) if base_frames_cfg else {}
    if mode == "train-test":
        frames_cfg["infected_window_hours"] = [start, end]
        for key in _SPLIT_OVERRIDE_KEYS:
            if key in frames_cfg:
                section = _copy_section(frames_cfg.get(key))
                section["infected_window_hours"] = [start, end]
                frames_cfg[key] = section
    else:  # test-only
        base_defaults = _base_section(frames_cfg)
        section = _copy_section(frames_cfg.get("test")) or base_defaults
        if not section:
            section = copy.deepcopy(base_defaults)
        section.update({"infected_window_hours": [start, end]})
        frames_cfg["test"] = section
    return frames_cfg


def evaluate_interval(
    cfg: Dict,
    base_data_cfg: Dict,
    transforms,
    run_dir: Path,
    device,
    fold_count: int,
    hour: float,
    start_hour: float,
    mode: str,
    split: str,
    metric_keys: List[str],
    task_cfg: Dict,
    analysis_cfg: Dict,
    weights_only: bool,
) -> Dict[str, Tuple[List[float], float | None, float | None]]:
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["frames"] = apply_interval_override(base_data_cfg.get("frames"), start_hour, hour, mode)
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

        train_ds, val_ds, test_ds = build_datasets(
            data_cfg,
            transforms,
            fold_index=fold_idx if fold_count > 1 else None,
            num_folds=fold_count,
        )
        dataset = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
        loader = build_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        model = build_model(cfg.get("model", {})).to(device)
        checkpoint = load_checkpoint(ckpt_path, device, weights_only)
        model.load_state_dict(checkpoint["model_state"])

        logger = logging.getLogger(f"sweep_{mode}_{fold_name}")
        metrics = evaluate_loader(model, loader, task_cfg, analysis_cfg, device, logger)
        
        # Collect all requested metrics
        for metric_key in metric_keys:
            metric_value = metrics.get(metric_key)
            if metric_value is not None and not math.isnan(metric_value):
                fold_metrics_dict[metric_key].append(float(metric_value))
        
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


def plot_single_metric_sweep(hours: List[float], stats: Dict[str, Dict[str, List[float]]], metric: str, output_path: Path) -> None:
    """Plot error bars for a single metric across both modes."""
    modes = ["train-test", "test-only"]
    labels = ["Train=[start,x], Test=[start,x]", "Train=full, Test=[start,x]"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode, label in zip(axes, modes, labels):
        mode_stats = stats.get(mode, {})
        means = mode_stats.get("mean", [])
        stds = mode_stats.get("std", [])
        valid_points = [
            (hour, mean, stds[idx] if idx < len(stds) and stds[idx] is not None else 0.0)
            for idx, (hour, mean) in enumerate(zip(hours, means))
            if mean is not None
        ]
        if valid_points:
            v_hours, v_means, v_stds = zip(*valid_points)
            ax.errorbar(v_hours, v_means, yerr=v_stds, fmt="-o", capsize=4, linewidth=2, markersize=6)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Upper bound hour (x)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel(metric.upper(), fontsize=11)
    fig.suptitle(f"Interval Sweep Analysis: {metric.upper()}", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved plot for %s to %s", metric, output_path)


def plot_multi_metric_sweep(
    hours: List[float],
    all_stats: Dict[str, Dict[str, Dict[str, List[float]]]],  # metric -> mode -> stats
    output_path: Path,
) -> None:
    """Plot combined error bars for multiple metrics."""
    modes = ["train-test", "test-only"]
    labels = ["Train=[start,x], Test=[start,x]", "Train=full, Test=[start,x]"]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_stats)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for ax, mode, label in zip(axes, modes, labels):
        for idx, (metric, stats) in enumerate(all_stats.items()):
            mode_stats = stats.get(mode, {})
            means = mode_stats.get("mean", [])
            stds = mode_stats.get("std", [])
            
            valid_points = [
                (hour, mean, stds[j] if j < len(stds) and stds[j] is not None else 0.0)
                for j, (hour, mean) in enumerate(zip(hours, means))
                if mean is not None
            ]
            
            if valid_points:
                v_hours, v_means, v_stds = zip(*valid_points)
                marker = markers[idx % len(markers)]
                ax.errorbar(
                    v_hours,
                    v_means,
                    yerr=v_stds,
                    fmt=f"-{marker}",
                    capsize=4,
                    linewidth=1.5,
                    markersize=6,
                    color=colors[idx],
                    label=metric.upper(),
                    alpha=0.8,
                )
        
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Upper bound hour (x)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="best")
    
    axes[0].set_ylabel("Metric Value", fontsize=11)
    fig.suptitle("Interval Sweep Analysis: Multiple Metrics", fontsize=13, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved combined multi-metric plot to %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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
    hours = sorted(args.upper_hours)
    if any(hour < args.start_hour for hour in hours):
        raise ValueError("All upper bound hours must be >= start-hour")
    
    # Determine which metrics to evaluate
    if args.metrics:
        metric_keys = args.metrics
        logging.info("Evaluating metrics: %s", ", ".join(metric_keys))
    else:
        metric_keys = [args.metric]
        logging.info("Evaluating single metric: %s", args.metric)
    
    modes = ("train-test", "test-only")
    
    # Structure: all_stats[metric][mode] = {"fold_metrics": [...], "mean": [...], "std": [...]}
    all_stats: Dict[str, Dict[str, Dict[str, List]]] = {
        metric_key: {
            mode: {"fold_metrics": [], "mean": [], "std": []} for mode in modes
        }
        for metric_key in metric_keys
    }

    for mode in modes:
        logging.info("Evaluating mode=%s", mode)
        for hour in hours:
            interval_results = evaluate_interval(
                cfg,
                base_data_cfg,
                transforms,
                run_dir,
                device,
                fold_count,
                hour,
                args.start_hour,
                mode,
                args.split,
                metric_keys,
                task_cfg,
                analysis_cfg,
                args.weights_only,
            )
            
            # Store results for each metric
            for metric_key in metric_keys:
                fold_values, mean_value, std_value = interval_results[metric_key]
                all_stats[metric_key][mode]["fold_metrics"].append(fold_values)
                all_stats[metric_key][mode]["mean"].append(mean_value)
                all_stats[metric_key][mode]["std"].append(std_value)
                
                if mean_value is not None:
                    logging.info(
                        "  [%.1f, %.1fh] %s = %.4f ± %.4f",
                        args.start_hour,
                        hour,
                        metric_key.upper(),
                        mean_value,
                        std_value,
                    )

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If multiple metrics, create combined plot
    if len(metric_keys) > 1:
        logging.info("Creating combined plot for all metrics...")
        combined_plot_path = output_dir / "interval_sweep_combined.png"
        plot_multi_metric_sweep(hours, all_stats, combined_plot_path)
    
    # Create individual plots for each metric
    for metric_key in metric_keys:
        plot_path = output_dir / f"interval_sweep_{metric_key}.png"
        plot_single_metric_sweep(hours, all_stats[metric_key], metric_key, plot_path)

    # Save raw data
    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / "interval_sweep_data.json"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "run_dir": str(run_dir),
        "start_hour": args.start_hour,
        "hours": hours,
        "metrics": metric_keys,
        "split": args.split,
        "stats": all_stats,
    }
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved sweep data to %s", save_path)


if __name__ == "__main__":
    main()
