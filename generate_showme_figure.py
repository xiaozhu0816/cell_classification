"""Generate a representative 2xN panel of real cell images ("Show Me" figure).

Motivation
----------
Biology journals typically expect at least one figure showing representative microscopy
images. For our live-cell HBMVEC time-course dataset, this script produces a
"mock-vs-infected" montage across selected timepoints.

What it does
------------
- Picks representative TIFF stacks from the dataset (one stack for mock/uninfected,
  one stack for infected).
- Extracts frames at requested hours (based on frames_per_hour).
- Applies an optional center-crop that removes a border fraction (e.g., 5%) to match
  the leakage-mitigation preprocessing used in model training.
- Builds a 2xN montage (row 1 = Mock, row 2 = Infected) and saves as a high-res PNG.

Usage (PowerShell)
------------------
python generate_showme_figure.py \
  --config configs/multitask_example_crop5pct.yaml \
  --hours 0 4 8 16 24 36 \
  --out outputs/showme_figure/showme_mock_vs_infected.png

If you want to force specific TIFFs:
python generate_showme_figure.py --mock-tiff <path> --infected-tiff <path> ...

Notes
-----
- The script is "safe" (read-only on the dataset).
- It saves an adjacent JSON manifest recording the chosen files and parameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from utils import load_config


@dataclass
class SelectedStacks:
    mock_tiffs: List[Path]
    infected_tiffs: List[Path]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2xN microscopy montage for the paper")
    p.add_argument("--config", type=str, default=None, help="YAML config used to locate dataset dirs")
    p.add_argument(
        "--mock-tiff",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit mock/uninfected TIFF(s). You can pass multiple to create multiple mock rows.",
    )
    p.add_argument(
        "--infected-tiff",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit infected TIFF(s). You can pass multiple to create multiple infected rows.",
    )
    p.add_argument(
        "--rows-per-condition",
        type=int,
        default=1,
        help="How many stacks (rows) to show per condition (mock + infected). Default=1 (2 rows total).",
    )
    p.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Generate multiple candidate figures by sampling different stacks (default=1).",
    )
    p.add_argument(
        "--candidate-seed",
        type=int,
        default=42,
        help="Random seed used when generating multiple candidate montages.",
    )
    p.add_argument("--hours", type=float, nargs="+", default=[0, 4, 8, 16, 24, 36], help="Hours to visualize")
    p.add_argument("--frames-per-hour", type=float, default=None, help="Override frames_per_hour")
    p.add_argument(
        "--crop-border-fraction",
        type=float,
        default=None,
        help="Center-crop by removing this fraction on each side (e.g., 0.05); overrides config",
    )
    p.add_argument(
        "--center-crop-size",
        type=int,
        default=None,
        help="If set, center-crop each frame to this many pixels (e.g., 512) for a zoomed-in panel.",
    )
    p.add_argument("--out", type=str, required=True, help="Output PNG path")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI")
    p.add_argument("--min-contrast-percentile", type=float, default=1.0, help="Lower percentile for display")
    p.add_argument("--max-contrast-percentile", type=float, default=99.0, help="Upper percentile for display")
    return p.parse_args()


def _choose_stacks_from_dirs(
    infected_dir: Path,
    uninfected_dir: Path,
    rows_per_condition: int,
    rng: Optional[np.random.Generator] = None,
) -> SelectedStacks:
    infected_candidates = sorted(infected_dir.glob("*.tif*"))
    mock_candidates = sorted(uninfected_dir.glob("*.tif*"))
    if not infected_candidates:
        raise FileNotFoundError(f"No infected TIFFs found in {infected_dir}")
    if not mock_candidates:
        raise FileNotFoundError(f"No uninfected TIFFs found in {uninfected_dir}")

    rows_per_condition = max(1, int(rows_per_condition))

    rng = rng or np.random.default_rng(42)

    # Strategy: avoid extreme ends (sometimes odd illumination); sample from the middle 80%.
    def sample_k(cands: List[Path], k: int) -> List[Path]:
        if k >= len(cands):
            return cands
        lo = int(len(cands) * 0.1)
        hi = max(lo + 1, int(len(cands) * 0.9))
        pool = cands[lo:hi]
        if len(pool) < k:
            pool = cands
        idx = rng.choice(len(pool), size=k, replace=False)
        out = [pool[int(i)] for i in idx]
        return sorted(out)

    mock_tiffs = sample_k(mock_candidates, rows_per_condition)
    infected_tiffs = sample_k(infected_candidates, rows_per_condition)
    return SelectedStacks(mock_tiffs=mock_tiffs, infected_tiffs=infected_tiffs)


def _render_montage(
    *,
    out_path: Path,
    stacks: SelectedStacks,
    hours: List[float],
    frames_per_hour: float,
    crop_border_fraction: Optional[float],
    center_crop_size: Optional[int],
    min_contrast_percentile: float,
    max_contrast_percentile: float,
    dpi: int,
    title: str,
) -> None:
    mock_stacks = [(_read_stack(p), p) for p in stacks.mock_tiffs]
    infected_stacks = [(_read_stack(p), p) for p in stacks.infected_tiffs]

    n = len(hours)
    rows = len(mock_stacks) + len(infected_stacks)

    fig, axes = plt.subplots(rows, n, figsize=(2.2 * n, 2.1 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])
    if n == 1:
        axes = axes.reshape(rows, 1)

    row_specs: List[Tuple[str, np.ndarray]] = []
    for i, (stack, _path) in enumerate(mock_stacks):
        row_specs.append((f"Mock (PBS) #{i+1}", stack))
    for i, (stack, _path) in enumerate(infected_stacks):
        row_specs.append((f"VEEV infected (TC-83) #{i+1}", stack))

    for row, (row_title, stack) in enumerate(row_specs):
        for col, h in enumerate(hours):
            idx = _frame_index_for_hour(h, frames_per_hour, stack.shape[0])
            frame = stack[idx]
            frame = _center_crop_border(frame, crop_border_fraction or 0.0)
            frame = _center_crop_pixels(frame, center_crop_size)
            frame = _prepare_display(frame, min_contrast_percentile, max_contrast_percentile)
            ax = axes[row, col]
            ax.imshow(frame, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{int(h)} h", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _read_stack(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    # Normalize stack shape to (T, H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # Some TIFFs store channels first/last; fall back to first channel.
        # Common observation in this project: (C, H, W) already handled; for 4D use first channel.
        return arr[:, 0, ...]
    raise ValueError(f"Unsupported TIFF array ndim={arr.ndim} shape={arr.shape} for {path}")


def _center_crop_border(frame: np.ndarray, crop_border_fraction: float) -> np.ndarray:
    if crop_border_fraction is None or crop_border_fraction <= 0:
        return frame
    h, w = frame.shape[-2], frame.shape[-1]
    dy = int(round(h * crop_border_fraction))
    dx = int(round(w * crop_border_fraction))
    y0, y1 = dy, max(dy + 1, h - dy)
    x0, x1 = dx, max(dx + 1, w - dx)
    return frame[..., y0:y1, x0:x1]


def _center_crop_pixels(frame: np.ndarray, size: Optional[int]) -> np.ndarray:
    """Crop the center to (size x size) pixels."""
    if size is None:
        return frame
    size = int(size)
    if size <= 0:
        return frame
    h, w = frame.shape[-2], frame.shape[-1]
    if h < size or w < size:
        size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frame[..., y0 : y0 + size, x0 : x0 + size]


def _frame_index_for_hour(hour: float, frames_per_hour: float, max_frames: int) -> int:
    idx = int(round(hour * frames_per_hour))
    return max(0, min(max_frames - 1, idx))


def _prepare_display(frame: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo = np.percentile(frame, lo_pct)
    hi = np.percentile(frame, hi_pct)
    if hi <= lo:
        hi = lo + 1.0
    frame = (frame - lo) / (hi - lo)
    frame = np.clip(frame, 0, 1)
    return frame


def main() -> None:
    args = _parse_args()

    cfg = load_config(args.config) if args.config else {}
    data_cfg = cfg.get("data", {})

    if args.frames_per_hour is not None:
        frames_per_hour = float(args.frames_per_hour)
    else:
        frames_per_hour = float((data_cfg.get("frames") or {}).get("frames_per_hour", 2.0))

    if args.crop_border_fraction is not None:
        crop_border_fraction = float(args.crop_border_fraction)
    else:
        crop_border_fraction = None
        transforms_cfg = (data_cfg.get("transforms") or {})
        if "crop_border_fraction" in transforms_cfg:
            crop_border_fraction = transforms_cfg.get("crop_border_fraction")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    explicit = bool(args.mock_tiff and args.infected_tiff)
    if explicit:
        base_stacks = SelectedStacks(
            mock_tiffs=[Path(p) for p in args.mock_tiff],
            infected_tiffs=[Path(p) for p in args.infected_tiff],
        )
        stacks_list = [base_stacks]
    else:
        infected_dir = Path(data_cfg.get("infected_dir", ""))
        uninfected_dir = Path(data_cfg.get("uninfected_dir", ""))
        # Support the cluster-style absolute paths (Linux) by falling back to Windows dataset root
        # when the config paths don't exist locally.
        if not infected_dir.exists() or not uninfected_dir.exists():
            infected_dir = Path(r"..\..\DATA\GMU_cell_1023\HBMVEC\Infected_well\no_labels")
            uninfected_dir = Path(r"..\..\DATA\GMU_cell_1023\HBMVEC\Uninfected_well\no_labels")

        rng = np.random.default_rng(int(args.candidate_seed))
        num_candidates = max(1, int(args.num_candidates))
        stacks_list = [
            _choose_stacks_from_dirs(infected_dir, uninfected_dir, args.rows_per_condition, rng=rng)
            for _ in range(num_candidates)
        ]

    hours = list(args.hours)
    title = "Representative HBMVEC morphology time course (EC channel)"

    # If multiple candidates requested, create numbered outputs next to --out.
    rendered_paths: List[Path] = []
    for i, stacks in enumerate(stacks_list, start=1):
        if len(stacks_list) == 1:
            this_out = out_path
        else:
            this_out = out_path.with_name(f"{out_path.stem}_cand{i:02d}{out_path.suffix}")
        _render_montage(
            out_path=this_out,
            stacks=stacks,
            hours=hours,
            frames_per_hour=frames_per_hour,
            crop_border_fraction=crop_border_fraction,
            center_crop_size=args.center_crop_size,
            min_contrast_percentile=args.min_contrast_percentile,
            max_contrast_percentile=args.max_contrast_percentile,
            dpi=args.dpi,
            title=title,
        )
        rendered_paths.append(this_out)

        manifest = {
            "mock_tiffs": [str(p) for p in stacks.mock_tiffs],
            "infected_tiffs": [str(p) for p in stacks.infected_tiffs],
            "rows_per_condition": int(args.rows_per_condition),
            "hours": hours,
            "frames_per_hour": frames_per_hour,
            "crop_border_fraction": crop_border_fraction,
            "center_crop_size": args.center_crop_size,
            "min_contrast_percentile": args.min_contrast_percentile,
            "max_contrast_percentile": args.max_contrast_percentile,
            "candidate_index": i,
            "candidate_seed": int(args.candidate_seed),
        }
        with open(this_out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    for p in rendered_paths:
        print(f"Saved montage to {p}")


if __name__ == "__main__":
    main()
