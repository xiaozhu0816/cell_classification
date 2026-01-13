"""Extract misclassified samples (top-K) and top regression errors, then render montage figures.

Inputs:
  --run-dir: a training run folder containing at least test_predictions.npz.
  Optionally prefers test_metadata.jsonl for precise mapping.

Outputs (written under --out-dir):
  - cls_mistakes_topK.csv
  - reg_errors_topK.csv
  - cls_mistakes_montage.png
  - reg_errors_montage.png

Notes:
  - If metadata is missing in the run folder (single-run training often doesn't save it),
    you can pass --metadata-jsonl pointing to a file with the same sample ordering.
  - We assume cls_preds is probability for positive class when float; threshold 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Sample:
    index: int
    path: str
    position: str
    frame_index: int
    hours_since_start: float
    label: int
    cls_prob: Optional[float] = None
    cls_pred: Optional[int] = None
    time_target: Optional[float] = None
    time_pred: Optional[float] = None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_predictions(run_dir: Path) -> Dict[str, np.ndarray]:
    npz_path = run_dir / "test_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


def resolve_metadata(run_dir: Path, override: Optional[Path]) -> List[Dict[str, Any]]:
    cand = run_dir / "test_metadata.jsonl"
    if override is not None:
        return read_jsonl(override)
    if cand.exists():
        return read_jsonl(cand)
    return []


def to_int_label(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.int64)
    # handle float targets that are 0/1
    return np.round(arr).astype(np.int64)


def to_prob(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "f":
        return arr.astype(np.float32)
    # if already 0/1 int, treat as probability
    return arr.astype(np.float32)


def build_samples(pred: Dict[str, np.ndarray], meta: List[Dict[str, Any]]) -> List[Sample]:
    n = None
    for k, v in pred.items():
        if v.ndim == 0:
            continue
        n = len(v)
        break
    if n is None:
        raise ValueError("Empty predictions")

    if meta:
        if len(meta) != n:
            raise ValueError(f"Metadata length {len(meta)} doesn't match predictions length {n}")
    else:
        # No metadata: we can't map to paths/frames.
        raise FileNotFoundError(
            "No metadata found. Provide --metadata-jsonl or generate test_metadata.jsonl for this run."
        )

    cls_prob = to_prob(pred.get("cls_preds")) if "cls_preds" in pred else None
    cls_targets = to_int_label(pred.get("cls_targets")) if "cls_targets" in pred else None

    time_pred = pred.get("time_preds")
    time_tgt = pred.get("time_targets")

    samples: List[Sample] = []
    for i in range(n):
        m = meta[i]
        samples.append(
            Sample(
                index=i,
                path=str(m.get("path", "")),
                frame_index=int(m.get("frame_index", -1)),
                    position=str(m.get("position", m.get("cell", ""))),
                hours_since_start=float(m.get("hours_since_start", float("nan"))),
                label=int(m.get("label", -1)),
                cls_prob=float(cls_prob[i]) if cls_prob is not None else None,
                cls_pred=int(cls_prob[i] > 0.5) if cls_prob is not None else None,
                time_pred=float(time_pred[i]) if time_pred is not None else None,
                time_target=float(time_tgt[i]) if time_tgt is not None else None,
            )
        )

    # sanity: if cls_targets exists, override label with cls_targets (they should match)
    if cls_targets is not None:
        for i, s in enumerate(samples):
            s.label = int(cls_targets[i])

    return samples


def pick_cls_mistakes(samples: List[Sample], top_k: int) -> List[Sample]:
    mistakes = [s for s in samples if (s.cls_pred is not None and s.cls_pred != s.label)]
    # sort by confidence (wrong but high confidence)
    def wrong_conf(s: Sample) -> float:
        if s.cls_prob is None:
            return 0.0
        # confidence in predicted class
        p = s.cls_prob
        conf = p if s.cls_pred == 1 else (1.0 - p)
        return float(conf)

    mistakes.sort(key=wrong_conf, reverse=True)
    return mistakes[:top_k]


def pick_top_reg_errors(samples: List[Sample], top_k: int) -> List[Sample]:
    reg = [s for s in samples if (s.time_pred is not None and s.time_target is not None)]
    reg.sort(key=lambda s: abs(float(s.time_pred) - float(s.time_target)), reverse=True)
    return reg[:top_k]


def write_csv(path: Path, rows: List[Sample], kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "path",
        "frame_index",
        "hours_since_start",
        "label",
        "cls_prob",
        "cls_pred",
        "time_target",
        "time_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in rows:
            w.writerow(
                {
                    "index": s.index,
                    "path": s.path,
                    "frame_index": s.frame_index,
                    "hours_since_start": s.hours_since_start,
                    "label": s.label,
                    "cls_prob": s.cls_prob,
                    "cls_pred": s.cls_pred,
                    "time_target": s.time_target,
                    "time_pred": s.time_pred,
                }
            )


def load_frame(path: str, frame_index: int) -> np.ndarray:
    """Load one frame from a multi-page tiff.

    Uses tifffile when available; falls back to imageio.
    Returns HxW (grayscale) array.
    """
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray(key=frame_index)
    except Exception:
        import imageio.v3 as iio

        arr = iio.imread(path, index=frame_index)

    if arr.ndim == 3:
        # take first channel
        arr = arr[..., 0]
    return arr


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.clip(img, 0, 1)
    img = (img - p1) / (p99 - p1)
    return np.clip(img, 0, 1)


def render_montage(samples: List[Sample], out_path: Path, title: str, ncols: int) -> None:
    import matplotlib.pyplot as plt

    if not samples:
        raise ValueError("No samples to render")

    n = len(samples)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))

    fig_w = 4.0 * ncols
    fig_h = 4.0 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for ax in axes[n:]:
        ax.axis("off")

    for i, s in enumerate(samples):
        ax = axes[i]
        img = load_frame(s.path, s.frame_index)
        img = normalize_for_display(img)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        reg_err = None
        if s.time_pred is not None and s.time_target is not None:
            reg_err = abs(float(s.time_pred) - float(s.time_target))

        lines = [
            f"cell={s.position}  true label={s.label}  true time={s.hours_since_start:.1f}h",
        ]
        if s.cls_prob is not None and s.cls_pred is not None:
            lines.append(f"p(inf)={s.cls_prob:.3f}  pred={s.cls_pred}")
        if reg_err is not None:
            lines.append(f"reg: pred={s.time_pred:.2f} tgt={s.time_target:.2f} err={reg_err:.2f}")

        ax.set_title("\n".join(lines), fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--metadata-jsonl", type=str, default=None, help="Optional override metadata file")
    ap.add_argument("--cls-topk", type=int, default=13)
    ap.add_argument("--reg-topk", type=int, default=5)
    ap.add_argument("--montage-cols", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    meta_path = Path(args.metadata_jsonl) if args.metadata_jsonl else None

    pred = load_predictions(run_dir)
    meta = resolve_metadata(run_dir, meta_path)
    samples = build_samples(pred, meta)

    total_cls_mistakes = 0
    if samples and samples[0].cls_pred is not None:
        total_cls_mistakes = sum(1 for s in samples if s.cls_pred != s.label)

    cls_bad = pick_cls_mistakes(samples, args.cls_topk)
    reg_bad = pick_top_reg_errors(samples, args.reg_topk)

    write_csv(out_dir / "cls_mistakes_topK.csv", cls_bad, "cls")
    write_csv(out_dir / "reg_errors_topK.csv", reg_bad, "reg")

    render_montage(
        cls_bad,
        out_dir / "cls_mistakes_montage.png",
        title=f"Misclassified Samples (showing {len(cls_bad)} of {total_cls_mistakes})",
        ncols=args.montage_cols,
    )
    render_montage(
        reg_bad,
        out_dir / "reg_errors_montage.png",
        title=f"Top-{len(reg_bad)} Regression Errors",
        ncols=min(args.montage_cols, max(1, len(reg_bad))),
    )

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
