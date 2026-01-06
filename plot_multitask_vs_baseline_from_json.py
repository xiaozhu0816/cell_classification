"""Plot baseline-vs-multitask temporal curves from JSON artifacts.

You asked to:
- Ignore baseline checkpoint-based evaluation (wrong model chosen previously)
- Use the baseline sliding-window JSON: `final_model_sliding_w6_s3_data.json`
- Use the multitask fair-eval JSON: `fair_temporal_metrics.json`

This script loads both JSONs, aligns them on common window centers, and produces
one 3-panel figure:
- AUC vs time window center
- Accuracy vs time window center
- F1 vs time window center

Each curve is mean ± std (shaded).

Usage (example)
-------------
python plot_multitask_vs_baseline_from_json.py \
  --baseline-json outputs/interval_sweep_analysis/20251212-145928/train-test_interval_1-46_sliding_window_fast_20251231-161811/final_model_sliding_w6_s3_data.json \
  --multitask-json outputs/multitask_resnet50/20260105-155852_5fold/fair_temporal_metrics.json \
  --output-png outputs/multitask_resnet50/20260105-155852_5fold/multitask_vs_baseline_temporal_FROMJSON.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_extract(b: Dict) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Return centers, and metric->(mean,std)."""
    results = b["results"]
    centers = np.array(results["auc"]["window_centers"], dtype=float)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in ["auc", "accuracy", "f1"]:
        out[k] = (
            np.array(results[k]["means"], dtype=float),
            np.array(results[k]["stds"], dtype=float),
        )
    return centers, out


def _multitask_extract(m: Dict) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    centers = np.array(m["centers"], dtype=float)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in ["auc", "accuracy", "f1"]:
        out[k] = (
            np.array(m["multitask"][k]["mean"], dtype=float),
            np.array(m["multitask"][k]["std"], dtype=float),
        )
    return centers, out


def _align(
    base_centers: np.ndarray,
    base_metrics: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mt_centers: np.ndarray,
    mt_metrics: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, Dict, Dict]:
    common = np.array(sorted(set(base_centers.tolist()).intersection(set(mt_centers.tolist()))), dtype=float)
    if common.size == 0:
        raise ValueError("No common window centers between baseline and multitask JSONs")

    b_idx = {float(c): i for i, c in enumerate(base_centers)}
    m_idx = {float(c): i for i, c in enumerate(mt_centers)}

    def _slice(metrics: Dict[str, Tuple[np.ndarray, np.ndarray]], idx_map: Dict[float, int]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        out = {}
        for k, (mean, std) in metrics.items():
            out[k] = (
                np.array([mean[idx_map[float(c)]] for c in common], dtype=float),
                np.array([std[idx_map[float(c)]] for c in common], dtype=float),
            )
        return out

    return common, _slice(base_metrics, b_idx), _slice(mt_metrics, m_idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-json", required=True, help="Path to final_model_sliding_w6_s3_data.json")
    ap.add_argument("--multitask-json", required=True, help="Path to fair_temporal_metrics.json")
    ap.add_argument("--output-png", required=True, help="Where to write the comparison PNG")
    ap.add_argument("--title-suffix", default="(baseline JSON vs multitask fair JSON)")
    args = ap.parse_args()

    baseline_path = Path(args.baseline_json)
    multitask_path = Path(args.multitask_json)
    out_png = Path(args.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    b = _load_json(baseline_path)
    m = _load_json(multitask_path)

    b_centers, b_metrics = _baseline_extract(b)
    mt_centers, mt_metrics = _multitask_extract(m)

    centers, b_aligned, mt_aligned = _align(b_centers, b_metrics, mt_centers, mt_metrics)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = {"baseline": "gray", "multitask": "tab:blue"}

    panels = [
        ("auc", "AUC"),
        ("accuracy", "Accuracy"),
        ("f1", "F1 Score"),
    ]

    for ax, (key, ylab) in zip(axes, panels):
        b_mean, b_std = b_aligned[key]
        m_mean, m_std = mt_aligned[key]

        ax.plot(centers, b_mean, marker="o", color=colors["baseline"], label="baseline")
        ax.fill_between(centers, b_mean - b_std, b_mean + b_std, color=colors["baseline"], alpha=0.2)

        ax.plot(centers, m_mean, marker="s", color=colors["multitask"], label="multitask")
        ax.fill_between(centers, m_mean - m_std, m_mean + m_std, color=colors["multitask"], alpha=0.2)

        ax.set_xlabel("Time Window Center (hours)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}: Multitask vs Baseline {args.title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
