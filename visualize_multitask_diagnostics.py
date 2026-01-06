"""Visual diagnostics for multitask CV results.

Generates a focused set of plots from fold_*/test_predictions.npz files.

Inputs
------
--result-dir: CV results directory with fold_*/test_predictions.npz

Expected NPZ keys (from regenerate_cv_predictions.py):
- cls_probs: [N,2]
- cls_labels: [N]
- reg_preds: [N]
- reg_labels: [N]  (absolute hours_since_start)
- image_paths: [N]

Outputs (written to result_dir)
------------------------------
- diag_confusion_by_time.png
- diag_regression_residual_by_time.png
- diag_pred_vs_true_time.png
- diag_calibration_overall.png
- diag_calibration_early0_6.png

This script is intentionally lightweight and only depends on numpy/pandas/matplotlib.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_POS_RE = re.compile(r"_p(?P<pos>\d+)_")


def _extract_position(path: str) -> str:
    name = Path(path).name
    m = _POS_RE.search(name)
    return m.group("pos") if m else "unknown"


def load_cv_predictions(result_dir: Path) -> pd.DataFrame:
    fold_npzs = sorted(result_dir.glob("fold_*/test_predictions.npz"))
    if not fold_npzs:
        raise FileNotFoundError(f"No fold_*/test_predictions.npz found under {result_dir}")

    parts = []
    for npz in fold_npzs:
        data = np.load(npz, allow_pickle=True)
        cls_probs = data["cls_probs"].astype(float)
        y_true = data["cls_labels"].astype(int)
        p_inf = cls_probs[:, 1]
        y_pred = (p_inf >= 0.5).astype(int)

        reg_pred = data["reg_preds"].astype(float)
        reg_true = data["reg_labels"].astype(float)

        image_paths = data["image_paths"]
        image_paths = [p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths]

        df = pd.DataFrame(
            {
                "fold": npz.parent.name,
                "image_path": image_paths,
                "tiff_name": [Path(p).name for p in image_paths],
                "position": [_extract_position(p) for p in image_paths],
                "hours": reg_true,  # absolute hours_since_start
                "y_true": y_true,
                "p_inf": p_inf,
                "y_pred": y_pred,
                "p_pred": np.where(y_pred == 1, p_inf, 1.0 - p_inf),
                "reg_true": reg_true,
                "reg_pred": reg_pred,
            }
        )
        df["residual"] = df["reg_pred"] - df["reg_true"]
        df["abs_error"] = df["residual"].abs()
        parts.append(df)

    return pd.concat(parts, ignore_index=True)


def plot_confusion_by_time(df: pd.DataFrame, out_path: Path, bin_hours: float = 2.0, max_time: float = 48.0) -> None:
    bins = np.arange(0, max_time + bin_hours, bin_hours)
    labels = [f"{bins[i]:.0f}-{bins[i+1]:.0f}" for i in range(len(bins) - 1)]

    df = df.copy()
    df["time_bin"] = pd.cut(df["hours"], bins=bins, right=False, labels=labels, include_lowest=True)

    def _rate(mask: pd.Series) -> float:
        return float(mask.mean()) if len(mask) else float("nan")

    rows = []
    for b in labels:
        sub = df[df["time_bin"] == b]
        if sub.empty:
            rows.append({"time_bin": b, "n": 0, "fn_rate": np.nan, "fp_rate": np.nan, "err_rate": np.nan})
            continue
        fn = (sub["y_true"] == 1) & (sub["y_pred"] == 0)
        fp = (sub["y_true"] == 0) & (sub["y_pred"] == 1)
        rows.append(
            {
                "time_bin": b,
                "n": int(len(sub)),
                "fn_rate": _rate(fn),
                "fp_rate": _rate(fp),
                "err_rate": _rate(sub["y_true"] != sub["y_pred"]),
            }
        )

    summ = pd.DataFrame(rows)

    x = np.arange(len(summ))
    plt.figure(figsize=(14, 5))
    plt.plot(x, summ["fn_rate"], marker="o", label="FN rate (infected→uninfected)")
    plt.plot(x, summ["fp_rate"], marker="s", label="FP rate (uninfected→infected)")
    plt.plot(x, summ["err_rate"], marker="^", linestyle="--", alpha=0.7, label="Overall error")
    plt.xticks(x, summ["time_bin"], rotation=90)
    plt.ylabel("Rate")
    plt.xlabel(f"Time bin (hours, bin={bin_hours}h)")
    plt.title("Classification error rates over time")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_regression_residual_by_time(df: pd.DataFrame, out_path: Path, bin_hours: float = 2.0, max_time: float = 48.0) -> None:
    bins = np.arange(0, max_time + bin_hours, bin_hours)
    centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(14, 5))
    for label, color in [(0, "#1f77b4"), (1, "#d62728")]:
        sub = df[df["y_true"] == label]
        means, stds = [], []
        for i in range(len(bins) - 1):
            m = (sub["hours"] >= bins[i]) & (sub["hours"] < bins[i + 1])
            if m.sum() < 10:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(sub.loc[m, "residual"].mean()))
                stds.append(float(sub.loc[m, "residual"].std()))

        means = np.array(means, dtype=float)
        stds = np.array(stds, dtype=float)
        plt.plot(centers, means, marker="o", color=color, label=f"{'infected' if label==1 else 'uninfected'} mean residual")
        plt.fill_between(centers, means - stds, means + stds, color=color, alpha=0.15)

    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Time (hours_since_start)")
    plt.ylabel("Residual (pred - true) hours")
    plt.title("Regression residual over time (mean ± std, binned)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pred_vs_true(df: pd.DataFrame, out_path: Path, max_points: int = 20000) -> None:
    sub = df
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)

    plt.figure(figsize=(6, 6))
    for label, color in [(0, "#1f77b4"), (1, "#d62728")]:
        s = sub[sub["y_true"] == label]
        plt.scatter(s["reg_true"], s["reg_pred"], s=6, alpha=0.25, color=color, label=f"{'infected' if label==1 else 'uninfected'}")

    lim = [0, max(float(df["reg_true"].max()), float(df["reg_pred"].max()), 48.0)]
    plt.plot(lim, lim, color="black", linewidth=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel("True time (hours_since_start)")
    plt.ylabel("Predicted time")
    plt.title("Predicted vs true time")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def calibration_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y_true[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_calibration(df: pd.DataFrame, out_path: Path, title: str, mask=None) -> None:
    sub = df if mask is None else df[mask]
    y = sub["y_true"].to_numpy().astype(float)
    p = sub["p_inf"].to_numpy().astype(float)

    xs, ys, ns = calibration_curve(y, p, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.plot(xs, ys, marker="o")
    for x, yv, n in zip(xs, ys, ns):
        plt.text(x, yv, str(n), fontsize=8, ha="left", va="bottom")
    plt.xlabel("Predicted P(infected)")
    plt.ylabel("Empirical infected rate")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    df = load_cv_predictions(result_dir)

    plot_confusion_by_time(df, result_dir / "diag_confusion_by_time.png")
    plot_regression_residual_by_time(df, result_dir / "diag_regression_residual_by_time.png")
    plot_pred_vs_true(df, result_dir / "diag_pred_vs_true_time.png")

    plot_calibration(df, result_dir / "diag_calibration_overall.png", title="Calibration (overall)")
    early = (df["hours"] >= 0.0) & (df["hours"] < 6.0)
    plot_calibration(df, result_dir / "diag_calibration_early0_6.png", title="Calibration (0–6h only)", mask=early)

    print("✓ Wrote diagnostic plots to:")
    for name in [
        "diag_confusion_by_time.png",
        "diag_regression_residual_by_time.png",
        "diag_pred_vs_true_time.png",
        "diag_calibration_overall.png",
        "diag_calibration_early0_6.png",
    ]:
        print(f"  - {result_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
