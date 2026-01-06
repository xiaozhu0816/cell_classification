"""Analyze multitask CV errors by position / TIFF.

This script aggregates classification and regression errors by:
- position (parsed from filename pattern "_p<digits>_")
- optional TIFF file name

It works with multitask CV outputs produced by `train_multitask_cv.py` +
`regenerate_cv_predictions.py`.

Expected folder layout:
  result_dir/
    fold_1/test_predictions.npz
    fold_2/test_predictions.npz
    ...

Each fold's NPZ is expected to contain:
  - cls_probs: [N, 2] softmax probs
  - cls_labels: [N] ground-truth class (0/1)
  - reg_preds: [N] regression predictions
  - reg_labels: [N] regression targets (hours_since_start)
  - image_paths: [N] full paths (strings)

Outputs (written to result_dir):
  - position_error_summary.csv
  - position_error_summary_top20.csv
  - position_error_boxplot.png
  - position_error_scatter.png

Usage:
  python analyze_multitask_position_errors.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold

Notes:
- This uses `reg_labels` as absolute time (hours_since_start) because that's what
  `regenerate_cv_predictions.py` currently saves.
- If you want "time since infection onset" targets, we can recompute those from
  the config onset hour and class labels.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


_POS_RE = re.compile(r"_p(?P<pos>\d+)_")


def parse_position(path: str) -> str:
    """Extract position id from filename; fallback to 'unknown'."""
    name = Path(path).name
    m = _POS_RE.search(name)
    return m.group("pos") if m else "unknown"


def load_fold_npz(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)

    cls_probs = data["cls_probs"]
    cls_labels = data["cls_labels"].astype(int)

    # regression
    reg_preds = data["reg_preds"].astype(float)
    reg_labels = data["reg_labels"].astype(float)

    image_paths = data["image_paths"]
    # allow bytes
    image_paths = [p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in image_paths]

    infected_prob = cls_probs[:, 1].astype(float)
    pred_label = (infected_prob >= 0.5).astype(int)

    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "tiff_name": [Path(p).name for p in image_paths],
            "position": [parse_position(p) for p in image_paths],
            "true_label": cls_labels,
            "pred_label": pred_label,
            "infected_prob": infected_prob,
            "true_time": reg_labels,
            "pred_time": reg_preds,
        }
    )

    df["cls_error"] = (df["pred_label"] != df["true_label"]).astype(int)
    df["abs_time_error"] = (df["pred_time"] - df["true_time"]).abs()

    # Confidence: probability of the *predicted* class
    df["cls_confidence"] = np.where(df["pred_label"] == 1, df["infected_prob"], 1.0 - df["infected_prob"])

    return df


def aggregate_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    g = df.groupby(key, dropna=False)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "cls_error_rate": g["cls_error"].mean(),
            "median_abs_time_error": g["abs_time_error"].median(),
            "mean_abs_time_error": g["abs_time_error"].mean(),
            "p90_abs_time_error": g["abs_time_error"].quantile(0.90),
            "p95_abs_time_error": g["abs_time_error"].quantile(0.95),
            "mean_confidence": g["cls_confidence"].mean(),
        }
    ).reset_index()

    # A simple combined flag: large regression error + low cls errors
    out["score"] = out["mean_abs_time_error"] + 2.0 * out["cls_error_rate"]

    return out.sort_values(["score", "mean_abs_time_error"], ascending=False)


def aggregate_by_key_and_class(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Aggregate metrics by key, with separate columns for infected/uninfected."""

    def _agg(sub: pd.DataFrame) -> Dict[str, float]:
        return {
            "n": float(len(sub)),
            "cls_error_rate": float(sub["cls_error"].mean()) if len(sub) else float("nan"),
            "median_abs_time_error": float(sub["abs_time_error"].median()) if len(sub) else float("nan"),
            "mean_abs_time_error": float(sub["abs_time_error"].mean()) if len(sub) else float("nan"),
            "p90_abs_time_error": float(sub["abs_time_error"].quantile(0.90)) if len(sub) else float("nan"),
            "p95_abs_time_error": float(sub["abs_time_error"].quantile(0.95)) if len(sub) else float("nan"),
            "mean_confidence": float(sub["cls_confidence"].mean()) if len(sub) else float("nan"),
        }

    rows = []
    for group_val, g in df.groupby(key, dropna=False):
        overall = _agg(g)
        inf = _agg(g[g["true_label"] == 1])
        uninf = _agg(g[g["true_label"] == 0])

        row: Dict[str, object] = {key: group_val}
        for prefix, stats in (
            ("all", overall),
            ("infected", inf),
            ("uninfected", uninf),
        ):
            for k, v in stats.items():
                # counts should be int-like
                if k == "n":
                    row[f"{prefix}_{k}"] = int(v) if not np.isnan(v) else 0
                else:
                    row[f"{prefix}_{k}"] = v

        # score based on overall
        row["score"] = float(row.get("all_mean_abs_time_error", 0.0)) + 2.0 * float(row.get("all_cls_error_rate", 0.0))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["score", "all_mean_abs_time_error"], ascending=False)


def plot_position_boxplot(df: pd.DataFrame, out_path: Path, top_k: int = 25) -> None:
    # Select positions with most samples
    counts = df["position"].value_counts()
    keep = counts.head(top_k).index.tolist()
    sub = df[df["position"].isin(keep)].copy()

    # Order by median error
    med = sub.groupby("position")["abs_time_error"].median().sort_values(ascending=False)
    positions = med.index.tolist()

    data = [sub.loc[sub["position"] == p, "abs_time_error"].values for p in positions]

    plt.figure(figsize=(max(12, top_k * 0.5), 6))
    # Matplotlib >=3.9 renamed labels -> tick_labels
    bp = plt.boxplot(data, tick_labels=positions, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set_facecolor("#cfe8f3")
    plt.xticks(rotation=90)
    plt.ylabel("|pred_time - true_time| (hours)")
    plt.title(f"Regression absolute error by position (top {top_k} positions by sample count)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_position_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = summary["cls_error_rate"].values
    y = summary["mean_abs_time_error"].values
    s = np.clip(summary["n"].values, 10, 200)

    plt.scatter(x, y, s=s, alpha=0.7, edgecolors="none")
    plt.xlabel("Classification error rate")
    plt.ylabel("Mean absolute time error (hours)")
    plt.title("Position-level error tradeoff (size ~ sample count)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=str, required=True)
    ap.add_argument("--by", choices=["position", "tiff"], default="position")
    ap.add_argument("--top-k", type=int, default=25)
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(result_dir)

    fold_npzs = sorted(result_dir.glob("fold_*/test_predictions.npz"))
    if not fold_npzs:
        raise FileNotFoundError(f"No fold_*/test_predictions.npz found under {result_dir}")

    dfs = []
    for npz in fold_npzs:
        df = load_fold_npz(npz)
        df["fold"] = npz.parent.name
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    key = "position" if args.by == "position" else "tiff_name"
    summary = aggregate_by_key(all_df, key=key)
    summary_by_class = aggregate_by_key_and_class(all_df, key=key)

    # Always generate a TIFF-level view too (often reveals repeated failure sources)
    tiff_summary = aggregate_by_key(all_df, key="tiff_name")
    tiff_summary_by_class = aggregate_by_key_and_class(all_df, key="tiff_name")

    out_csv = result_dir / f"{args.by}_error_summary.csv"
    out_csv_top = result_dir / f"{args.by}_error_summary_top20.csv"
    out_csv_by_class = result_dir / f"{args.by}_error_summary_by_class.csv"
    out_tiff_csv = result_dir / "tiff_error_summary.csv"
    out_tiff_csv_top = result_dir / "tiff_error_summary_top20.csv"
    out_tiff_csv_by_class = result_dir / "tiff_error_summary_by_class.csv"

    summary.to_csv(out_csv, index=False)
    summary.head(20).to_csv(out_csv_top, index=False)
    summary_by_class.to_csv(out_csv_by_class, index=False)

    tiff_summary.to_csv(out_tiff_csv, index=False)
    tiff_summary.head(20).to_csv(out_tiff_csv_top, index=False)
    tiff_summary_by_class.to_csv(out_tiff_csv_by_class, index=False)

    if args.by == "position":
        plot_position_boxplot(all_df, result_dir / "position_error_boxplot.png", top_k=args.top_k)
        plot_position_scatter(summary, result_dir / "position_error_scatter.png")

    print(f"✓ Wrote: {out_csv}")
    print(f"✓ Wrote: {out_csv_top}")
    print(f"✓ Wrote: {out_csv_by_class}")
    print(f"✓ Wrote: {out_tiff_csv}")
    print(f"✓ Wrote: {out_tiff_csv_top}")
    print(f"✓ Wrote: {out_tiff_csv_by_class}")
    if args.by == "position":
        print(f"✓ Wrote: {result_dir / 'position_error_boxplot.png'}")
        print(f"✓ Wrote: {result_dir / 'position_error_scatter.png'}")

    # quick headline
    worst = summary.head(10)
    print("\nTop 10 groups by score:")
    with pd.option_context('display.max_columns', 50, 'display.width', 160):
        print(worst)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
