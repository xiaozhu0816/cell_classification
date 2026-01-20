# Deduplicate overlapping GMU patches (optional anti-leakage step)

Your GMU acquisition is reported to have ~5% overlap between adjacent fields of view.
Even with position-level splitting, overlap can create near-duplicate images and inflate CV.

This repo’s default data pipeline (`datasets/timecourse_dataset.py`) trains from **TIFF stacks** (frames),
so overlap between fields is baked into the acquisition and not explicitly represented in code.

If you have a **patch image dataset** (e.g. `DATA/Image_patches/` with PNG/JPG/TIF patches),
this script creates a *deduplicated* dataset view by removing near-identical patches.

## What it does

- Computes a perceptual hash (pHash) per patch.
- Considers two patches duplicates if their hash Hamming distance <= `--phash-threshold`.
- Keeps the first one and removes the others.
- Writes manifests (kept + removed) and optionally materializes an output folder.

## Usage (PowerShell)

```powershell
cd "\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$\scratch\WSI\zhengjie\CODE\cell_classification"

python scripts\dedup_overlapping_patches.py `
  --input "\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$\scratch\WSI\zhengjie\DATA\Image_patches" `
  --output "\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$\scratch\WSI\zhengjie\DATA\Image_patches_dedup" `
  --phash-threshold 6
```

### Quick dry-run

If the dataset is huge, start with a limit:

```powershell
python scripts\dedup_overlapping_patches.py --input "...\Image_patches" --limit 2000
```

## Notes / tuning

- `--phash-threshold` is the main knob:
  - smaller (e.g. 4) = stricter (keeps more images)
  - larger (e.g. 8–10) = more aggressive dedup
- The script uses hardlinks when possible (fast, low disk). If hardlink fails, it copies.

## If you only have TIFF stacks

If your training data is only multi-frame TIFF stacks (not patch images), deduplication should be applied
at the **position/FOV-level** (metadata-aware) rather than patch-level, or by adding a dataset mode that
filters overlapping positions based on stage coordinates.

If you have stage-coordinate metadata for each position, we can implement a stronger, principled solution:
- compute adjacency graph of overlapping positions
- select a maximum independent set per fold (or keep one per overlap cluster)

Send me an example filename/metadata mapping for position -> (x,y) (or any coordinate info), and I’ll wire it in.
