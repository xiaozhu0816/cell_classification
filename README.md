# Cell Classification Pipeline

This repository contains a PyTorch training pipeline that classifies live-cell imaging time-course TIFF stacks as **infected** or **uninfected**. Each TIFF file stores 95 time points (t0–t94) acquired every 30 minutes. Empirically, infected cells begin to show cytopathic effects (CPE) between 16–30 hours post-infection, so the default configuration samples frames within that window.

## Project layout

```
cell_classification/
├── configs/
│   └── resnet50_baseline.yaml   # Experiment settings
├── datasets/
│   └── timecourse_dataset.py    # TIFF dataset + split utilities
├── models/
│   └── resnet.py                # ResNet classifier wrapper
├── utils/
│   ├── config.py                # YAML loader
│   ├── logger.py                # Console/file logger
│   ├── metrics.py               # Average meters + binary metrics
│   ├── seed.py                  # Seed all RNGs
│   └── transforms.py            # Build torchvision pipelines
├── requirements.txt             # Python dependencies
└── train.py                     # Main training entrypoint
```

## Data expectations

```
DATA_ROOT/
├── infected/
│   ├── *.tiff   (multi-frame stacks containing infected wells)
├── uninfected/
│   └── *.tiff   (control stacks)
```

Update the paths within `configs/resnet50_baseline.yaml` to point at your infected/uninfected directories. The default config already references the provided UNC paths.

### Frame expansion logic

- **Infected stacks**: every TIFF is expanded into multiple samples covering **all frames between 16 h and 30 h** (based on `frames_per_hour`, default 2). This captures the CPE window instead of just one snapshot.
- **Uninfected stacks**: every frame from t0–t92 is used (configurable via `uninfected_use_all` / `uninfected_stride`).
- Adjust these rules by editing the `data.frames` block in the YAML. For example, change `infected_window_hours` or set a stride > 1 to subsample.

## Quick start

1. **Install dependencies**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Launch training**
   ```powershell
   python train.py --config configs/resnet50_baseline.yaml
   ```

Training artifacts (logs and checkpoints) are written under `outputs/` and `checkpoints/` respectively. The script automatically evaluates on the validation split every epoch and reports accuracy, precision, recall, F1, and AUC. After training, it runs a final evaluation on the held-out test split.

> **Note**: The TIFF stacks use LZW compression, so the Python wheel `imagecodecs` must be installed (already listed in `requirements.txt`). If you install dependencies manually, ensure `pip install imagecodecs` succeeds or tifffile will raise `COMPRESSION.LZW requires the 'imagecodecs' package`.

## Cross-validation & experiment outputs

- Set `training.k_folds` in the YAML to a value > 1 (e.g., 5) to enable K-fold cross-validation. The script will iteratively train on each fold, log per-fold metrics, and print an aggregate summary at the end.
- Every run receives a timestamp-based `run_id`, so logs/checkpoints live under `outputs/<experiment>/<run_id>/fold_XXofYY` and `checkpoints/<experiment>/<run_id>/fold_XXofYY`. This keeps concurrent experiments isolated.
- Within each fold you still get the best checkpoint (`best.pt`) plus detailed logs showing how many frame-level samples were drawn from infected vs uninfected stacks.

## Visualizing Grad-CAM heatmaps

Use `visualize_cam.py` to inspect which regions drive the classifier’s decision:

```powershell
python visualize_cam.py `
   --config configs/resnet50_baseline.yaml `
   --checkpoint checkpoints/resnet50_baseline/20251204-141200/fold_01of05/best.pt `
   --split val `
   --num-samples 8 `
   --num-folds 5 `
   --fold-index 0
```

- The script samples random frames from the requested split, generates Grad-CAM overlays, and writes raw/input/heatmap/overlay PNGs plus `metadata.json` under `cam_outputs/<checkpoint_name>/<split>_foldX/`.
- Filenames now encode the ground-truth label and infection probability (e.g., `cam_003_infected_prob0.92_overlay.png`), and each overlay carries a text banner (`Label / Pred / P(infected)`) so you can inspect ground truth at a glance.
- `--num-folds`/`--fold-index` should mirror the training run if you want CAMs from the same fold partition; leave them at defaults to ignore CV.
- Adjust `--num-samples`, `--split`, or `--cmap` to explore different frames.
- If you intentionally want PyTorch 2.6's safer `weights_only=True` behavior, pass `--weights-only`; by default the script disables it (`weights_only=False`) so legacy checkpoints load without extra allow-list setup.

## Customization tips

- **Frame selection**: Adjust `data.frames.strategy` and `data.frames.window_hours` in the YAML if you want to sample earlier or later time ranges, or switch to a fixed `target_frame`.
- **Model**: Swap to another ResNet depth by changing `model.name` (`resnet18`, `resnet34`, `resnet101`, `resnet152`).
- **Augmentations**: Toggle augmentation flags inside `data.transforms` (flips, rotations, color jitter).
- **Batching**: Increase `batch_size` and `num_workers` once you confirm GPU/CPU memory headroom.

Feel free to extend the pipeline with additional datasets, metrics, or foundation models as your experiments evolve.
