"""Detect potentially overlapping FOV positions by comparing frame border hashes.

Problem
-------
GMU reports ~5% overlap between neighboring positions (FOVs). When we don't have
stage (x,y) coordinates, we can still *detect* overlap by looking for shared
border content between positions.

Idea
----
If FOV A overlaps with FOV B by ~5%, then a thin border strip of A should look
very similar to the corresponding border strip of B (left/right or top/bottom).

This script:
- Reads a single representative frame from each TIFF stack (default: frame 0)
- Extracts border strips (left/right/top/bottom) with configurable width
- Computes a perceptual hash for each strip
- Compares all pairs within a condition (infected or mock)
- Reports pairs whose strip-hash distance is below threshold (potential overlaps)

This does NOT modify training directly; it helps you audit and then optionally
exclude one of each overlapping pair when building folds.

Usage (PowerShell)
------------------
python scripts\detect_duplicate_positions_by_border_hash.py `
  --dir "\\...\HBMVEC\Infected_well\no_labels" `
  --out overlaps_infected.jsonl

"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tifffile
from PIL import Image


def phash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    img = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    dct = np.fft.fft2(arr)
    dct_low = np.real(dct[:hash_size, :hash_size])
    med = np.median(dct_low[1:, 1:])
    bits = dct_low > med
    return bits.reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def load_frame(path: Path, frame_index: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        pages = len(tif.pages)
        idx = max(0, min(pages - 1, frame_index))
        arr = tif.asarray(key=idx)
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def border_strips(arr: np.ndarray, strip_px: int) -> Dict[str, np.ndarray]:
    if arr.ndim == 3:
        arr = arr[..., 0]
    h, w = arr.shape
    s = max(1, min(strip_px, min(h, w) // 2))
    return {
        "left": arr[:, :s],
        "right": arr[:, w - s :],
        "top": arr[:s, :],
        "bottom": arr[h - s :, :],
    }


@dataclass
class StackInfo:
    path: Path
    key: str
    hashes: Dict[str, np.ndarray]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True, help="Folder containing TIFF stacks")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL file")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--strip-px", type=int, default=64)
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--threshold", type=int, default=8, help="Max Hamming distance for a potential overlap")
    ap.add_argument("--limit", type=int, default=0, help="Debug: only read first N stacks")
    args = ap.parse_args()

    root = Path(args.dir)
    stacks = sorted(root.glob("*.tif*"))
    if args.limit and args.limit > 0:
        stacks = stacks[: args.limit]

    infos: List[StackInfo] = []
    for p in stacks:
        arr = load_frame(p, args.frame)
        strips = border_strips(arr, strip_px=args.strip_px)
        hashes = {k: phash(Image.fromarray(v), hash_size=args.hash_size) for k, v in strips.items()}
        infos.append(StackInfo(path=p, key=p.name, hashes=hashes))

    rows: List[Dict] = []
    # Compare likely neighbor directions: left<->right and top<->bottom
    comparisons = [("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top")]

    for a, b in itertools.combinations(infos, 2):
        best = None
        for sa, sb in comparisons:
            d = hamming(a.hashes[sa], b.hashes[sb])
            if best is None or d < best[0]:
                best = (d, sa, sb)
        assert best is not None
        d, sa, sb = best
        if d <= args.threshold:
            rows.append(
                {
                    "a": a.key,
                    "b": b.key,
                    "best_hamming": int(d),
                    "a_side": sa,
                    "b_side": sb,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total_pairs = len(infos) * (len(infos) - 1) // 2
    print(f"Stacks: {len(infos)}")
    print(f"Pairs:  {total_pairs}")
    print(f"Hits:   {len(rows)}")
    print(f"Out:    {out_path}")


if __name__ == "__main__":
    main()
