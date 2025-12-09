from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

import numpy as np

try:
    from numpy import _core as np_core_module  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback for older numpy
    np_core_module = None

from datasets import build_datasets, resolve_frame_policies, format_policy_summary
from models import build_model
from utils import build_transforms, load_config
from train import (
    build_target_tensor,
    get_analysis_config,
    get_task_config,
    log_threshold_sweep,
    log_time_bin_metrics,
    meta_batch_to_list,
    compute_regression_metrics,
    binary_metrics,
    log_dataset_stack_membership,
)

try:
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - older torch
    add_safe_globals = None


SAFE_GLOBALS_REGISTERED = False
NUMPY_SCALAR_CLASS = None
if np_core_module is not None:
    try:  # pragma: no cover - depends on numpy internals
        NUMPY_SCALAR_CLASS = np_core_module.multiarray.scalar
    except AttributeError:
        NUMPY_SCALAR_CLASS = None


def register_numpy_safe_globals(weights_only: bool) -> None:
    global SAFE_GLOBALS_REGISTERED
    if not weights_only or add_safe_globals is None or SAFE_GLOBALS_REGISTERED:
        return
    safe_types = []
    dtype = getattr(np, "dtype", None)
    if dtype is not None:
        safe_types.append(dtype)
    if NUMPY_SCALAR_CLASS is not None:
        safe_types.append(NUMPY_SCALAR_CLASS)
    if not safe_types:
        return
    try:
        add_safe_globals(safe_types)
        SAFE_GLOBALS_REGISTERED = True
    except Exception as exc:  # pragma: no cover - safety net
        logging.warning("Failed to register numpy safe globals for torch.load: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all fold checkpoints for a run")
    parser.add_argument("--config", default="configs/resnet50_baseline.yaml", help="Path to YAML config")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Checkpoint run directory containing fold_* subfolders (e.g., checkpoints/resnet50_early/20251205-101010)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save aggregated evaluation metrics (JSON). Defaults to <run-dir>/fold_eval.json",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Load checkpoints with weights_only=True (default False to maintain compatibility with earlier torch versions)",
    )
    return parser.parse_args()


def build_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)


def log_frame_policies(logger, data_cfg: Dict) -> None:
    policies = resolve_frame_policies(data_cfg.get("frames"))
    for split_name in ("train", "val", "test"):
        policy = policies[split_name]
        logger.info("frame_policy[%s]: %s", split_name, format_policy_summary(policy))


def evaluate_loader(model, loader, task_cfg: Dict, analysis_cfg: Dict, device, logger) -> Dict[str, float]:
    model.eval()
    logits_list, targets_list, hours_list = [], [], []
    with torch.no_grad():
        for images, labels, meta in loader:
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            targets = build_target_tensor(labels, meta_list, task_cfg, device)
            logits = model(images)
            logits_list.append(logits.detach().cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            hours_list.extend(float(entry.get("hours_since_start", 0.0)) for entry in meta_list)
    if not logits_list:
        return {}
    logits = np.concatenate(logits_list, axis=0).squeeze(-1)
    targets = np.concatenate(targets_list, axis=0).squeeze(-1)
    task_type = task_cfg.get("type", "classification")
    if task_type == "regression":
        metrics = compute_regression_metrics(logits, targets)
    else:
        metrics = binary_metrics(logits, targets)
        probs = 1.0 / (1.0 + np.exp(-logits))
        thresholds = analysis_cfg.get("thresholds", [])
        if thresholds:
            log_threshold_sweep(logger, "eval", probs, targets, thresholds)
        time_bins = analysis_cfg.get("time_bins", [])
        if time_bins:
            log_time_bin_metrics(logger, "eval", time_bins, logits, targets, hours_list)
    return metrics


def load_checkpoint(path: Path, device, weights_only: bool):
    register_numpy_safe_globals(weights_only)
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except Exception as exc:
        if weights_only:
            logging.warning("weights_only load failed for %s (%s); retrying with weights_only=False", path, exc)
            return torch.load(path, map_location=device, weights_only=False)
        raise


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    task_cfg = get_task_config(cfg)
    analysis_cfg = get_analysis_config(cfg)
    device = torch.device(args.device)

    training_cfg = cfg.get("training", {})
    k_folds = max(1, int(training_cfg.get("k_folds", 1)))

    data_cfg = cfg.get("data", {})
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)

    batch_size = data_cfg.get("batch_size", 256)
    num_workers = data_cfg.get("num_workers", 4)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    results = []
    for fold_idx in range(k_folds):
        fold_name = f"fold_{fold_idx + 1:02d}of{k_folds:02d}"
        ckpt_path = run_dir / fold_name / "best.pt"
        if not ckpt_path.exists():
            logging.warning("Checkpoint missing for %s (%s)", fold_name, ckpt_path)
            continue

        train_ds, val_ds, test_ds = build_datasets(
            data_cfg,
            transforms,
            fold_index=fold_idx if k_folds > 1 else None,
            num_folds=k_folds,
        )
        log_frame_policies(logging.getLogger(f"{fold_name}_policy"), data_cfg)
        split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
        dataset = split_map[args.split]
        log_dataset_stack_membership(logger, dataset, args.split, save_dir=run_dir / fold_name)
        loader = build_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        model = build_model(cfg.get("model", {})).to(device)
        checkpoint = load_checkpoint(ckpt_path, device, args.weights_only)
        model.load_state_dict(checkpoint["model_state"])
        logger = logging.getLogger(f"{fold_name}_{args.split}")
        metrics = evaluate_loader(model, loader, task_cfg, analysis_cfg, device, logger)
        logging.info("%s | %s metrics: %s", fold_name, args.split, metrics)
        results.append(
            {
                "fold": fold_idx,
                "split": args.split,
                "checkpoint": str(ckpt_path),
                "metrics": metrics,
            }
        )

    if not results:
        logging.warning("No evaluation results recorded")
        return

    output_path = Path(args.output) if args.output else run_dir / "fold_eval.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved evaluation summary to %s", output_path)


if __name__ == "__main__":
    main()
