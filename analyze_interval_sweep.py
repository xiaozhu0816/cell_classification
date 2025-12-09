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
    parser.add_argument("--metric", default="auc", help="Metric key to summarize (e.g., auc, accuracy, f1)")
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
    metric_key: str,
    task_cfg: Dict,
    analysis_cfg: Dict,
    weights_only: bool,
) -> Tuple[List[float], float | None, float | None]:
    data_cfg = copy.deepcopy(base_data_cfg)
    data_cfg["frames"] = apply_interval_override(base_data_cfg.get("frames"), start_hour, hour, mode)
    batch_size = data_cfg.get("batch_size", 256)
    num_workers = data_cfg.get("num_workers", 4)

    fold_metrics: List[float] = []
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
        metric_value = metrics.get(metric_key)
        if metric_value is None or math.isnan(metric_value):
            logging.warning("[%s] Metric '%s' unavailable for hour %.2f", fold_name, metric_key, hour)
            continue
        fold_metrics.append(float(metric_value))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not fold_metrics:
        return [], None, None
    values = np.asarray(fold_metrics, dtype=np.float32)
    return fold_metrics, float(values.mean()), float(values.std(ddof=0))


def plot_error_bars(hours: List[float], stats: Dict[str, Dict[str, List[float]]], metric: str, output_path: Path) -> None:
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
            ax.errorbar(v_hours, v_means, yerr=v_stds, fmt="-o", capsize=4)
        ax.set_title(label)
        ax.set_xlabel("Upper bound hour (x)")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_ylabel(metric)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
    modes = ("train-test", "test-only")
    stats: Dict[str, Dict[str, List[float]]] = {
        mode: {"fold_metrics": [], "mean": [], "std": []} for mode in modes
    }

    for mode in modes:
        logging.info("Evaluating mode=%s", mode)
        for hour in hours:
            fold_values, mean_value, std_value = evaluate_interval(
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
                args.metric,
                task_cfg,
                analysis_cfg,
                args.weights_only,
            )
            stats[mode]["fold_metrics"].append(fold_values)
            stats[mode]["mean"].append(mean_value)
            stats[mode]["std"].append(std_value)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    plot_path = output_dir / f"interval_sweep_{args.metric}.png"
    plot_error_bars(hours, stats, args.metric, plot_path)
    logging.info("Saved sweep plot to %s", plot_path)

    if args.save_data:
        save_path = Path(args.save_data)
    else:
        save_path = output_dir / f"interval_sweep_{args.metric}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "run_dir": str(run_dir),
        "start_hour": args.start_hour,
        "hours": hours,
        "metric": args.metric,
        "split": args.split,
        "stats": stats,
    }
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logging.info("Saved sweep data to %s", save_path)


if __name__ == "__main__":
    main()
