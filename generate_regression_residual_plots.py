"""
Generate Regression Residual Analysis Plots for 5-Fold CV

Generates two figures:
1. Regression residual over time (mean ± std, binned) - separate for infected/uninfected
2. Error distribution by time range (boxplots + trend with std bands) - separate for infected/uninfected

Usage:
    python generate_regression_residual_plots.py --result-dir outputs/multitask_resnet50/20260109-164300_5fold
"""
import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_fold_data(cv_dir: Path):
    """Load predictions and metadata from all folds."""
    all_data = {
        "time_preds": [],
        "time_targets": [],
        "cls_labels": [],
        "hours_since_start": [],
        "fold_id": [],
    }

    for fold_idx in range(1, 6):
        fold_dir = cv_dir / f"fold_{fold_idx}"
        pred_file = fold_dir / "test_predictions.npz"
        meta_file = fold_dir / "test_metadata.jsonl"

        if not pred_file.exists():
            print(f"⚠ Warning: {pred_file} not found, skipping fold {fold_idx}")
            continue

        data = np.load(pred_file)
        all_data["time_preds"].append(data["time_preds"])
        all_data["time_targets"].append(data["time_targets"])
        all_data["cls_labels"].append(data["cls_targets"])

        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = json.loads(line)
                    all_data["hours_since_start"].append(m.get("hours_since_start", float("nan")))
                    all_data["fold_id"].append(fold_idx)
        else:
            fold_n = int(data["cls_targets"].shape[0])
            all_data["hours_since_start"].extend([float("nan")] * fold_n)
            all_data["fold_id"].extend([fold_idx] * fold_n)

    # Concatenate arrays
    for key in ["time_preds", "time_targets", "cls_labels"]:
        if all_data[key]:
            all_data[key] = np.concatenate(all_data[key])
        else:
            all_data[key] = np.array([])

    all_data["hours_since_start"] = np.asarray(all_data["hours_since_start"], dtype=np.float32)
    all_data["fold_id"] = np.asarray(all_data["fold_id"], dtype=np.int64)

    return all_data


def plot_residual_over_time(data, output_dir: Path, bin_size: float = 3.0):
    """
    Plot regression residual (pred - true) over time with mean ± std bands.
    Separate curves for infected and uninfected cells.
    """
    time_preds = data["time_preds"]
    time_targets = data["time_targets"]
    cls_labels = data["cls_labels"]
    hours = data["hours_since_start"]
    fold_id = data["fold_id"]

    residuals = time_preds - time_targets

    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0

    fold_ids = sorted(set(int(x) for x in fold_id.tolist()))
    if not fold_ids:
        print("⚠ No fold data for residual plot")
        return

    # Bin edges
    min_hour = 0.0
    max_hour = float(np.nanmax(hours)) + bin_size
    bins = np.arange(min_hour, max_hour, bin_size)
    bin_centers = bins[:-1] + bin_size / 2.0

    def compute_binned_residuals(mask):
        """Compute per-fold binned residuals, then aggregate."""
        per_fold_means: List[List[float]] = []
        for fid in fold_ids:
            fold_mask = (fold_id == fid) & mask
            means = []
            for i in range(len(bins) - 1):
                bin_mask = fold_mask & (hours >= bins[i]) & (hours < bins[i + 1])
                if bin_mask.sum() > 0:
                    means.append(float(np.mean(residuals[bin_mask])))
                else:
                    means.append(np.nan)
            per_fold_means.append(means)

        mean_arr = np.asarray(per_fold_means, dtype=np.float32)
        mean_vals = np.nanmean(mean_arr, axis=0)
        std_vals = np.nanstd(mean_arr, axis=0)
        return mean_vals, std_vals

    inf_mean, inf_std = compute_binned_residuals(infected_mask)
    uninf_mean, uninf_std = compute_binned_residuals(uninfected_mask)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Uninfected (blue)
    ax.plot(bin_centers, uninf_mean, "o-", color="tab:blue", linewidth=2, label="uninfected mean residual")
    ax.fill_between(
        bin_centers,
        uninf_mean - uninf_std,
        uninf_mean + uninf_std,
        alpha=0.25,
        color="tab:blue",
    )

    # Infected (red)
    ax.plot(bin_centers, inf_mean, "o-", color="tab:red", linewidth=2, label="infected mean residual")
    ax.fill_between(
        bin_centers,
        inf_mean - inf_std,
        inf_mean + inf_std,
        alpha=0.25,
        color="tab:red",
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.7)
    ax.set_xlabel("Time (hours_since_start)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residual (pred - true, hours)", fontsize=12, fontweight="bold")
    ax.set_title(f"Regression residual over time (mean ± std, binned)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "regression_residual_over_time.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_file}")
    plt.close()


def plot_error_distribution_by_time(data, output_dir: Path):
    """
    Plot error distribution by time range.
    Top row: boxplots for infected/uninfected.
    Bottom row: trend over time (mean ± std) for infected/uninfected.
    """
    time_preds = data["time_preds"]
    time_targets = data["time_targets"]
    cls_labels = data["cls_labels"]
    hours = data["hours_since_start"]
    fold_id = data["fold_id"]

    errors = np.abs(time_preds - time_targets)

    infected_mask = cls_labels == 1
    uninfected_mask = cls_labels == 0

    fold_ids = sorted(set(int(x) for x in fold_id.tolist()))

    # Define time ranges for boxplots
    time_ranges = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 36), (36, 48)]
    range_labels = [f"{int(start)}-{int(end)}h" for start, end in time_ranges]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Infected boxplots
    ax = axes[0, 0]
    inf_boxes = []
    inf_medians = []
    for start, end in time_ranges:
        mask = infected_mask & (hours >= start) & (hours < end)
        inf_boxes.append(errors[mask])
        if mask.sum() > 0:
            inf_medians.append(np.median(errors[mask]))
        else:
            inf_medians.append(np.nan)

    bp = ax.boxplot(
        inf_boxes,
        tick_labels=range_labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="darkred", linewidth=2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("salmon")
        patch.set_alpha(0.7)

    # Annotate median values
    for i, med in enumerate(inf_medians):
        if not np.isnan(med):
            ax.text(i + 1, med + 0.1, f"{med:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Absolute Error (hours)", fontsize=11, fontweight="bold")
    ax.set_title("Infected Cells: Error Distribution by Time Range", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # Top-right: Uninfected boxplots
    ax = axes[0, 1]
    uninf_boxes = []
    uninf_medians = []
    for start, end in time_ranges:
        mask = uninfected_mask & (hours >= start) & (hours < end)
        uninf_boxes.append(errors[mask])
        if mask.sum() > 0:
            uninf_medians.append(np.median(errors[mask]))
        else:
            uninf_medians.append(np.nan)

    bp = ax.boxplot(
        uninf_boxes,
        tick_labels=range_labels,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="darkblue", linewidth=2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    for i, med in enumerate(uninf_medians):
        if not np.isnan(med):
            ax.text(i + 1, med + 0.1, f"{med:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Absolute Error (hours)", fontsize=11, fontweight="bold")
    ax.set_title("Uninfected Cells: Error Distribution by Time Range", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # Bottom row: Trend over time with std bands (fine-grained bins)
    def compute_error_trend(mask, bin_size=2.0):
        min_h = 0.0
        max_h = 48.0
        bins = np.arange(min_h, max_h + bin_size, bin_size)
        centers = bins[:-1] + bin_size / 2.0

        per_fold_means: List[List[float]] = []
        for fid in fold_ids:
            fold_mask = (fold_id == fid) & mask
            means = []
            for i in range(len(bins) - 1):
                bin_mask = fold_mask & (hours >= bins[i]) & (hours < bins[i + 1])
                if bin_mask.sum() > 0:
                    means.append(float(np.mean(errors[bin_mask])))
                else:
                    means.append(np.nan)
            per_fold_means.append(means)

        mean_arr = np.asarray(per_fold_means, dtype=np.float32)
        mean_vals = np.nanmean(mean_arr, axis=0)
        std_vals = np.nanstd(mean_arr, axis=0)
        return centers, mean_vals, std_vals

    # Bottom-left: Infected trend
    ax = axes[1, 0]
    centers_inf, mean_inf, std_inf = compute_error_trend(infected_mask)
    ax.plot(centers_inf, mean_inf, "o-", color="tab:red", linewidth=2, label="Mean Error")
    ax.fill_between(centers_inf, mean_inf - std_inf, mean_inf + std_inf, alpha=0.25, color="tab:red", label="±1 Std")
    ax.set_xlabel("Time Since Infection (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Error (hours)", fontsize=11, fontweight="bold")
    ax.set_title("Infected Cells: Error Trend Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Uninfected trend
    ax = axes[1, 1]
    centers_uninf, mean_uninf, std_uninf = compute_error_trend(uninfected_mask)
    ax.plot(centers_uninf, mean_uninf, "o-", color="tab:blue", linewidth=2, label="Mean Error")
    ax.fill_between(
        centers_uninf, mean_uninf - std_uninf, mean_uninf + std_uninf, alpha=0.25, color="tab:blue", label="±1 Std"
    )
    ax.set_xlabel("Experiment Time (hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Absolute Error (hours)", fontsize=11, fontweight="bold")
    ax.set_title("Uninfected Cells: Error Trend Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "error_distribution_by_time_range.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate regression residual analysis plots for 5-fold CV")
    parser.add_argument("--result-dir", type=str, required=True, help="Path to CV results directory")
    parser.add_argument("--bin-size", type=float, default=3.0, help="Bin size in hours for residual plot (default: 3)")
    args = parser.parse_args()

    cv_dir = Path(args.result_dir)
    if not cv_dir.exists():
        print(f"❌ Error: {cv_dir} does not exist")
        return 1

    print("=" * 80)
    print("REGRESSION RESIDUAL ANALYSIS FOR 5-FOLD CV")
    print("=" * 80)
    print(f"CV Directory: {cv_dir}\n")

    print("Loading predictions from all folds...")
    data = load_fold_data(cv_dir)

    n_samples = len(data["time_preds"])
    print(f"✓ Loaded {n_samples} total predictions from all folds\n")

    print("Generating regression residual plots...")
    print("-" * 80)

    plot_residual_over_time(data, cv_dir, bin_size=args.bin_size)
    plot_error_distribution_by_time(data, cv_dir)

    print("\n" + "=" * 80)
    print("✅ ALL DONE!")
    print("=" * 80)
    print(f"\nGenerated plots in {cv_dir}:")
    print("  • regression_residual_over_time.png")
    print("  • error_distribution_by_time_range.png")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
