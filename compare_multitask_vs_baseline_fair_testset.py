"""Fair temporal comparison: multitask vs baseline on the same test subset.

Why
---
The existing `compare_multitask_vs_baseline.py` overlays:
- multitask temporal metrics computed from multitask CV test predictions
- baseline temporal metrics computed from a separate baseline evaluation run

Those two runs can differ in:
- which timepoints were included for uninfected
- which [start,end] interval was used for infected
- even which exact test samples existed

This script re-evaluates BOTH models on an identical test subset defined by:
- infected time window: [test_start, test_end]
- uninfected time window: [test_start, test_end]   (matched)

Then it computes temporal metrics via a sliding window (size/stride) using the SAME
predictions (same samples) for both models.

Inputs
------
--multitask-dir: CV output dir with fold_*/checkpoints/best.pt
--baseline-ckpt-dir: dir with fold_01_best.pth ... fold_05_best.pth
--multitask-config: config used for multitask training (for dataset paths + transforms)
--baseline-config: config for baseline model architecture (ResNetClassifier)

Outputs
-------
Writes into --output-dir (default: multitask-dir):
- multitask_vs_baseline_temporal_fair_[start-end].png
- fair_temporal_metrics.json

Notes
-----
- This script computes classification metrics only (AUC/accuracy/F1) for both models.
- It uses the DATA SPLITS determined by `multitask-config` (same split_seed/ratios).
  For a perfectly fair comparison, the baseline model should have been trained on the
  same split_seed/ratios too; but at least evaluation will be on the identical test set.

Usage
-----
python compare_multitask_vs_baseline_fair_testset.py \
  --multitask-dir outputs/multitask_resnet50/20260105-155852_5fold \
  --multitask-config configs/multitask_example.yaml \
  --baseline-ckpt-dir outputs/interval_sweep_analysis/20251212-145928/checkpoints/train-test_interval_1-46 \
  --baseline-config configs/resnet50_baseline.yaml \
  --test-start 1 --test-end 48
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.timecourse_dataset import build_datasets
from models.resnet import build_model
from models.multitask_resnet import MultiTaskResNet
from utils.config import load_config
from utils.transforms import build_transforms


def set_test_window(cfg: Dict, start: float, end: float) -> Dict:
    cfg = {**cfg}
    data = {**cfg.get("data", {})}
    frames = {**data.get("frames", {})}
    frames["infected_window_hours"] = [float(start), float(end)]
    frames["uninfected_window_hours"] = [float(start), float(end)]
    frames["uninfected_use_all"] = False

    # also nuke nested split overrides if present
    for k in ["train", "val", "test", "default", "eval", "evaluation"]:
        if isinstance(frames.get(k), dict):
            nested = {**frames[k]}
            nested.pop("infected_window_hours", None)
            nested.pop("uninfected_window_hours", None)
            frames[k] = nested

    data["frames"] = frames
    cfg["data"] = data
    return cfg


def load_multitask_fold_model(ckpt_path: Path, cfg: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    model = MultiTaskResNet(
        backbone=model_cfg.get("backbone", model_cfg.get("name", "resnet50")),
        pretrained=bool(model_cfg.get("pretrained", False)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        train_backbone=bool(model_cfg.get("train_backbone", True)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # tolerate different formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError(f"Unsupported multitask checkpoint type: {type(ckpt)}")
    model.load_state_dict(state)
    return model.to(device).eval()


def load_baseline_fold_model(ckpt_path: Path, cfg: Dict, device: torch.device) -> torch.nn.Module:
    # baseline checkpoints are .pth from interval sweep training; they usually store a state_dict
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(cfg.get("model", {})).to(device)

    # tolerate different checkpoint formats
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    elif isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        # try raw state_dict
        try:
            model.load_state_dict(state)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unrecognized checkpoint format for {ckpt_path}: keys={list(state.keys())[:10]}") from e
    else:
        raise RuntimeError(f"Unsupported baseline checkpoint type: {type(state)}")

    return model.to(device).eval()


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, multitask: bool) -> Tuple[np.ndarray, np.ndarray]:
    probs_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for images, labels, _meta in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        y = labels.numpy().astype(int)

        if multitask:
            logits, _t = model(images)
        else:
            logits = model(images)

        # Both baseline + multitask may output [N,2] logits. If [N,1], treat as sigmoid.
        if logits.ndim == 2 and logits.shape[1] == 2:
            p = torch.softmax(logits, dim=1)[:, 1]
        elif logits.ndim == 2 and logits.shape[1] == 1:
            p = torch.sigmoid(logits[:, 0])
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        probs_all.append(p.detach().cpu().numpy())
        y_all.append(y)

    return np.concatenate(probs_all), np.concatenate(y_all)


def temporal_metrics(probs: np.ndarray, y: np.ndarray, hours: np.ndarray, window_size: float, stride: float, start: float, end: float) -> Dict[str, List[float]]:
    centers: List[float] = []
    aucs: List[float] = []
    accs: List[float] = []
    f1s: List[float] = []

    cur = start
    while cur + window_size <= end:
        w0, w1 = cur, cur + window_size
        m = (hours >= w0) & (hours < w1)
        if m.sum() >= 10 and len(np.unique(y[m])) > 1:
            p = probs[m]
            yt = y[m]
            yp = (p >= 0.5).astype(int)
            centers.append((w0 + w1) / 2.0)
            aucs.append(float(roc_auc_score(yt, p)))
            accs.append(float(accuracy_score(yt, yp)))
            f1s.append(float(f1_score(yt, yp, zero_division=0)))
        cur += stride

    return {"centers": centers, "auc": aucs, "accuracy": accs, "f1": f1s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multitask-dir", required=True)
    ap.add_argument("--multitask-config", required=True)
    ap.add_argument("--baseline-ckpt-dir", required=True)
    ap.add_argument("--baseline-config", required=True)
    ap.add_argument("--test-start", type=float, default=1.0)
    ap.add_argument("--test-end", type=float, default=48.0)
    ap.add_argument("--window-size", type=float, default=6.0)
    ap.add_argument("--stride", type=float, default=3.0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    multitask_dir = Path(args.multitask_dir)
    baseline_ckpt_dir = Path(args.baseline_ckpt_dir)

    out_dir = Path(args.output_dir) if args.output_dir else multitask_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mt_cfg = load_config(args.multitask_config)
    base_cfg = load_config(args.baseline_config)

    # Force identical test window for BOTH classes
    mt_cfg = set_test_window(mt_cfg, args.test_start, args.test_end)

    # Build datasets (we only use test, but need fold split)
    transforms = build_transforms(mt_cfg.get("data", {}).get("transforms", {}))

    device = torch.device(args.device)

    mt_ckpts = [multitask_dir / f"fold_{i}" / "checkpoints" / "best.pt" for i in range(1, 6)]
    base_ckpts = sorted(baseline_ckpt_dir.glob("fold_*_best.pth"))

    if not all(p.exists() for p in mt_ckpts):
        missing = [str(p) for p in mt_ckpts if not p.exists()]
        raise FileNotFoundError(f"Missing multitask fold checkpoints: {missing}")
    if len(base_ckpts) == 0:
        raise FileNotFoundError(f"No baseline fold_*_best.pth found under {baseline_ckpt_dir}")

    num_folds = min(5, len(base_ckpts))

    # Collect per-fold temporal metrics lists to compute mean/std
    per_fold = {
        "multitask": {"auc": [], "accuracy": [], "f1": []},
        "baseline": {"auc": [], "accuracy": [], "f1": []},
        "centers": None,
    }

    for fold_idx in range(num_folds):
        _, _, test_ds = build_datasets(
            data_cfg=mt_cfg["data"],
            transforms=transforms,
            fold_index=fold_idx,
            num_folds=num_folds,
        )

        # extract hours from dataset metadata without loading images
        hours = np.array([test_ds.get_metadata(i)["hours_since_start"] for i in range(len(test_ds))], dtype=float)

        bs = int(mt_cfg.get("data", {}).get("batch_size", 128))
        mult = int(mt_cfg.get("data", {}).get("eval_batch_size_multiplier", 2))
        loader = DataLoader(test_ds, batch_size=bs * mult, shuffle=False, num_workers=0, pin_memory=True)

        mt_model = load_multitask_fold_model(mt_ckpts[fold_idx], mt_cfg, device)
        base_model = load_baseline_fold_model(base_ckpts[fold_idx], base_cfg, device)

        mt_p, y = collect_predictions(mt_model, loader, device, multitask=True)
        base_p, _y2 = collect_predictions(base_model, loader, device, multitask=False)

        mt_tm = temporal_metrics(mt_p, y, hours, args.window_size, args.stride, args.test_start, args.test_end)
        base_tm = temporal_metrics(base_p, y, hours, args.window_size, args.stride, args.test_start, args.test_end)

        # align centers (should match)
        if per_fold["centers"] is None:
            per_fold["centers"] = mt_tm["centers"]
        else:
            # if differing due to class-balance in windows, truncate to intersection
            common = [c for c in per_fold["centers"] if c in mt_tm["centers"] and c in base_tm["centers"]]
            if common != per_fold["centers"]:
                per_fold["centers"] = common

        # helper to align arrays to stored centers
        def _align(tm: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
            idx = {c: i for i, c in enumerate(tm["centers"])}
            out = {}
            for k in ["auc", "accuracy", "f1"]:
                out[k] = np.array([tm[k][idx[c]] for c in per_fold["centers"]], dtype=float)
            return out

        mt_al = _align(mt_tm)
        base_al = _align(base_tm)

        for k in ["auc", "accuracy", "f1"]:
            per_fold["multitask"][k].append(mt_al[k])
            per_fold["baseline"][k].append(base_al[k])

    centers = np.array(per_fold["centers"], dtype=float)

    def _mean_std(arrs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        a = np.stack(arrs, axis=0)
        return a.mean(axis=0), a.std(axis=0)

    out = {
        "test_window": [args.test_start, args.test_end],
        "window_size": args.window_size,
        "stride": args.stride,
        "centers": centers.tolist(),
        "multitask": {},
        "baseline": {},
    }

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {"baseline": "gray", "multitask": "tab:blue"}

    for ax, metric, title in zip(axes, ["auc", "accuracy", "f1"], ["AUC", "Accuracy", "F1 Score"]):
        for name in ["baseline", "multitask"]:
            mean, std = _mean_std(per_fold[name][metric])
            out[name][metric] = {"mean": mean.tolist(), "std": std.tolist()}
            ax.plot(centers, mean, marker="o" if name == "baseline" else "s", color=colors[name], label=name)
            ax.fill_between(centers, mean - std, mean + std, color=colors[name], alpha=0.2)

        ax.set_xlabel("Time Window Center (hours)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}: Multitask vs Baseline (fair test window)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    out_png = out_dir / f"multitask_vs_baseline_temporal_fair_{int(args.test_start)}-{int(args.test_end)}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    out_json = out_dir / "fair_temporal_metrics.json"
    out["plot"] = str(out_png)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✓ Wrote: {out_png}")
    print(f"✓ Wrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
