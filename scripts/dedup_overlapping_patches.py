"""Deduplicate overlapping GMU patches to reduce near-duplicate leakage.

Why this exists
--------------
Your GMU time-lapse acquisition has ~5% overlap between adjacent fields of view.
Even if you split by position, overlap can still create near-duplicate samples
(or cross-position duplicates if positions overlap physically) and inflate CV.

This script *does not change your existing training settings*.
Instead, it creates a filtered view of an existing patch-based dataset by
removing near-duplicates using perceptual hashing.

Approach
--------
- Compute a perceptual hash (pHash) for each image.
- Use a Hamming-distance threshold to cluster near-identical images.
- Keep only the first image in each cluster.

Important assumptions
---------------------
- Input is a directory tree containing image patches (png/jpg/tif).
- If you train directly from TIFF stacks (as in `datasets/timecourse_dataset.py`),
  then overlap is not handled here; in that case, you likely need a different
  dataset class that loads patch images.

Outputs
-------
- A JSONL manifest listing kept images (and optionally removed ones).
- Optionally, a copied/symlinked (Windows hardlink) directory of deduped images.

Example
-------
python scripts/dedup_overlapping_patches.py \
  --input "/path/to/Image_patches" \
  --output "/path/to/Image_patches_dedup" \
  --phash-threshold 6

"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def phash(image: Image.Image, hash_size: int = 16) -> np.ndarray:
    """Compute a simple perceptual hash (DCT-based pHash).

    Returns a boolean vector of length hash_size*hash_size.
    """
    # Resize + grayscale
    img = image.convert("L").resize((hash_size * 4, hash_size * 4), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)

    # DCT (via FFT trick).
    # We keep this dependency-free; quality is fine for near-duplicate pruning.
    dct = np.fft.fft2(arr)
    dct_low = np.real(dct[:hash_size, :hash_size])

    med = np.median(dct_low[1:, 1:])  # ignore DC component
    bits = dct_low > med
    return bits.reshape(-1)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def load_image(path: Path, max_size: int = 1024) -> Image.Image:
    img = Image.open(path)
    img.load()
    # Downscale for speed, preserving aspect
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img


@dataclass
class Item:
    path: Path
    relpath: str
    hash_bits: np.ndarray


def build_items(paths: List[Path], root: Path, hash_size: int, max_size: int) -> List[Item]:
    items: List[Item] = []
    for p in paths:
        img = load_image(p, max_size=max_size)
        bits = phash(img, hash_size=hash_size)
        items.append(Item(path=p, relpath=str(p.relative_to(root)), hash_bits=bits))
    return items


def dedup(items: List[Item], threshold: int) -> Tuple[List[Item], List[Dict]]:
    kept: List[Item] = []
    removed_meta: List[Dict] = []

    for it in items:
        is_dup = False
        for k in kept:
            if hamming(it.hash_bits, k.hash_bits) <= threshold:
                removed_meta.append({"removed": it.relpath, "kept": k.relpath})
                is_dup = True
                break
        if not is_dup:
            kept.append(it)

    return kept, removed_meta


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Windows hardlink works across same volume; if it fails, fall back to copy
        dst.hardlink_to(src)
    except Exception:
        import shutil

        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Root folder containing image patches")
    ap.add_argument("--output", type=str, required=False, help="Optional output folder for deduped patches")
    ap.add_argument("--manifest", type=str, required=False, default="dedup_manifest.jsonl")
    ap.add_argument("--removed", type=str, required=False, default="dedup_removed.jsonl")
    ap.add_argument("--hash-size", type=int, default=16)
    ap.add_argument("--phash-threshold", type=int, default=6, help="Max Hamming distance to consider duplicate")
    ap.add_argument("--max-image-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="Debug: only process first N images")
    args = ap.parse_args()

    in_root = Path(args.input)
    if not in_root.exists():
        raise FileNotFoundError(in_root)

    paths = sorted(iter_images(in_root))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    items = build_items(paths, in_root, hash_size=args.hash_size, max_size=args.max_image_size)
    kept, removed = dedup(items, threshold=args.phash_threshold)

    # Write manifests
    manifest_rows = [{"path": it.relpath} for it in kept]
    write_jsonl(Path(args.manifest), manifest_rows)
    write_jsonl(Path(args.removed), removed)

    if args.output:
        out_root = Path(args.output)
        for it in kept:
            hardlink_or_copy(it.path, out_root / it.relpath)

    print(f"Input images: {len(items)}")
    print(f"Kept images:  {len(kept)}")
    print(f"Removed:      {len(items) - len(kept)}")
    print(f"Manifest:     {args.manifest}")
    if args.output:
        print(f"Output dir:   {args.output}")


if __name__ == "__main__":
    main()
