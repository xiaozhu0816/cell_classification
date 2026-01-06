"""Valley period (13–19h) regression error analysis for 5-fold CV.

This is the CV/aggregated counterpart of `analyze_regression_by_class.py`.
Instead of loading a single `test_predictions.npz`, it loads all folds:
  <result-dir>/fold_*/test_predictions.npz

It reproduces:
- Printed table of mean/median/95th percentile error by period
- Welch/Student t-test for Valley vs Mid (infected)
- Boxplot figure with infected (left) and uninfected (right)

Outputs
-------
- valley_period_analysis_cv.png
- valley_period_analysis_cv.txt

Usage
-----
python analyze_valley_period_cv.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold

Notes
-----
This expects each fold NPZ to contain the keys used by CV prediction export:
  cls_labels, cls_preds, cls_probs, reg_labels, reg_preds
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_all_fold_predictions(cv_dir: Path, num_folds: int = 5) -> dict:
    all_data = {
        "reg_preds": [],
        "reg_labels": [],
        "cls_labels": [],
    }

    for fold_idx in range(1, num_folds + 1):
        pred_file = cv_dir / f"fold_{fold_idx}" / "test_predictions.npz"
        if not pred_file.exists():
            continue
        data = np.load(pred_file)
        all_data["reg_preds"].append(data["reg_preds"])
        all_data["reg_labels"].append(data["reg_labels"])
        all_data["cls_labels"].append(data["cls_labels"])

    for k in list(all_data.keys()):
        if all_data[k]:
            all_data[k] = np.concatenate(all_data[k])
        else:
            all_data[k] = np.array([])

    return all_data


def summarize_period(errors: np.ndarray, mask: np.ndarray) -> tuple[int, float, float, float]:
    n = int(mask.sum())
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    e = errors[mask]
    return n, float(e.mean()), float(np.median(e)), float(np.percentile(e, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True, help="5-fold CV result directory")
    ap.add_argument("--num-folds", type=int, default=5)
    ap.add_argument("--valley-start", type=float, default=13.0)
    ap.add_argument("--valley-end", type=float, default=19.0)
    ap.add_argument("--early-cut", type=float, default=6.0)
    ap.add_argument("--mid-start", type=float, default=6.0)
    ap.add_argument("--mid-end", type=float, default=13.0)
    ap.add_argument("--output-name", default="valley_period_analysis_cv.png")
    args = ap.parse_args()

    cv_dir = Path(args.result_dir)
    data = load_all_fold_predictions(cv_dir, num_folds=args.num_folds)

    reg_preds = data["reg_preds"]
    reg_labels = data["reg_labels"]
    cls_labels = data["cls_labels"]

    if reg_preds.size == 0:
        raise FileNotFoundError(f"No fold_*/test_predictions.npz found under {cv_dir}")

    errors = np.abs(reg_preds - reg_labels)

    # Masks
    valley_mask_infected = (cls_labels == 1) & (reg_labels >= args.valley_start) & (reg_labels <= args.valley_end)
    valley_mask_uninfected = (cls_labels == 0) & (reg_labels >= args.valley_start) & (reg_labels <= args.valley_end)

    early_mask_infected = (cls_labels == 1) & (reg_labels < args.early_cut)
    mid_mask_infected = (cls_labels == 1) & (reg_labels >= args.mid_start) & (reg_labels < args.mid_end)
    late_mask_infected = (cls_labels == 1) & (reg_labels > args.valley_end)

    early_mask_uninf = (cls_labels == 0) & (reg_labels < args.valley_start)
    late_mask_uninf = (cls_labels == 0) & (reg_labels > args.valley_end)

    lines = []
    lines.append("=" * 80)
    lines.append("CRITICAL TIME WINDOW ANALYSIS (CV Aggregated) - The 13-19h Valley")
    lines.append("=" * 80)

    lines.append("\nINFECTED CELLS - Error by Time Period:")
    lines.append("-" * 80)
    lines.append(f"{'Period':<20} {'N Samples':<12} {'Mean Error':<12} {'Median Error':<12} {'95th %ile'}")
    lines.append("-" * 80)

    inf_periods = [
        ("Early (0-6h)", early_mask_infected),
        ("Mid (6-13h)", mid_mask_infected),
        ("Valley (13-19h)", valley_mask_infected),
        ("Late (>19h)", late_mask_infected),
    ]
    for name, mask in inf_periods:
        n, mean, med, p95 = summarize_period(errors, mask)
        if n > 0:
            lines.append(f"{name:<20} {n:<12d} {mean:<12.3f} {med:<12.3f} {p95:.3f}")

    lines.append("\nUNINFECTED CELLS - Error by Time Period:")
    lines.append("-" * 80)
    uninf_periods = [
        ("Early (<13h)", early_mask_uninf),
        ("Valley (13-19h)", valley_mask_uninfected),
        ("Late (>19h)", late_mask_uninf),
    ]
    for name, mask in uninf_periods:
        n, mean, med, p95 = summarize_period(errors, mask)
        if n > 0:
            lines.append(f"{name:<20} {n:<12d} {mean:<12.3f} {med:<12.3f} {p95:.3f}")

    # Statistical test: valley vs mid (infected)
    if valley_mask_infected.sum() > 1 and mid_mask_infected.sum() > 1:
        valley_errors = errors[valley_mask_infected]
        mid_errors = errors[mid_mask_infected]
        t_stat, p_value = stats.ttest_ind(valley_errors, mid_errors)

        lines.append("\nSTATISTICAL TEST:")
        lines.append("-" * 80)
        lines.append("Valley (13-19h) vs Mid (6-13h) - Infected Cells")
        lines.append(f"  t-statistic: {t_stat:.3f}")
        lines.append(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            if valley_errors.mean() > mid_errors.mean():
                lines.append("  Valley period is SIGNIFICANTLY WORSE (p < 0.05)")
            else:
                lines.append("  Valley period is SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            lines.append("  No significant difference (p >= 0.05)")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Infected
    ax = axes[0]
    periods_inf = []
    labels_inf = []
    colors_inf = []
    for name, mask, color in [
        ("Early\n(0-6h)", early_mask_infected, "lightcoral"),
        ("Mid\n(6-13h)", mid_mask_infected, "coral"),
        ("Valley\n(13-19h)", valley_mask_infected, "darkred"),
        ("Late\n(>19h)", late_mask_infected, "indianred"),
    ]:
        if mask.sum() > 0:
            periods_inf.append(errors[mask])
            labels_inf.append(name)
            colors_inf.append(color)

    if periods_inf:
        bp = ax.boxplot(periods_inf, labels=labels_inf, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_inf):
            patch.set_facecolor(color)

    ax.set_ylabel("Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.set_title("Infected Cells: Error Distribution by Time Period (CV)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvspan(2.5, 3.5, alpha=0.2, color="red")

    # Uninfected
    ax = axes[1]
    periods_uninf = []
    labels_uninf = []
    colors_uninf = []
    for name, mask, color in [
        ("Early\n(<13h)", early_mask_uninf, "lightblue"),
        ("Valley\n(13-19h)", valley_mask_uninfected, "darkblue"),
        ("Late\n(>19h)", late_mask_uninf, "steelblue"),
    ]:
        if mask.sum() > 0:
            periods_uninf.append(errors[mask])
            labels_uninf.append(name)
            colors_uninf.append(color)

    if periods_uninf:
        bp = ax.boxplot(periods_uninf, labels=labels_uninf, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors_uninf):
            patch.set_facecolor(color)

    ax.set_ylabel("Absolute Error (hours)", fontsize=12, fontweight="bold")
    ax.set_title("Uninfected Cells: Error Distribution by Time Period (CV)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvspan(1.5, 2.5, alpha=0.2, color="blue")

    plt.tight_layout()
    out_png = cv_dir / args.output_name
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    out_txt = cv_dir / "valley_period_analysis_cv.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n✓ Saved valley analysis plot to {out_png}")
    print(f"✓ Saved valley analysis report to {out_txt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
