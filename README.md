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

- **Infected stacks (default)**: every TIFF is expanded into multiple samples covering **all frames between 16 h and 30 h** (based on `frames_per_hour`, default 2). This captures the CPE window instead of just one snapshot.
- **Uninfected stacks**: every frame from t0–t92 is used (configurable via `uninfected_use_all` / `uninfected_stride`).
- Adjust these rules by editing the `data.frames` block in the YAML. For example, change `infected_window_hours` or set a stride > 1 to subsample. The early-detection config (`configs/resnet50_early.yaml`) simply switches this window to `[0, 16]` to focus on subtle pre-CPE cues.
- Need split-specific behavior? Provide overrides under `data.frames.train`, `data.frames.val`, or `data.frames.test` (optionally `data.frames.default`). Each override inherits unspecified fields from the base block, letting you train on `[1, 30]` while evaluating on `[1, 12]` without touching the shared config.
- Set `data.balance_sampler: true` when you want a class-balanced `WeightedRandomSampler` during training (useful for early windows where positives are scarce).

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
- After training with `k_folds > 1`, a `cv_summary.json` file is written in the run's output directory with the full per-fold/aggregate metrics so you can revisit the results without rerunning.
- Enable fine-grained diagnostics by adding an `analysis` block to your config:
   ```yaml
   analysis:
      thresholds: [0.05, 0.1, 0.2, 0.3]
      time_bins:
         - [0, 6]
         - [6, 10]
         - [10, 14]
         - [14, 16]
   ```
   During evaluation the trainer prints a threshold sweep table (precision/recall/F1 at each cut-off) plus per-bin metrics so you can see exactly which time windows remain challenging.

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

## Advanced training modes

### Early-infection classifier

Use `configs/resnet50_early.yaml` to restrict infected samples to the 0–16 h window:

```powershell
python train.py --config configs/resnet50_early.yaml
```

This keeps the same model/optimizer but trains on frames where infection cues are subtle. Compare its ROC-AUC against the baseline to understand how early the signal becomes reliable.

### Time-to-infection regression

`train.py` also supports a regression mode. Set `task.type: regression` and describe the target in `task.regression` (see `configs/resnet50_time_regression.yaml`). Example:

```powershell
python train.py --config configs/resnet50_time_regression.yaml
```

Key settings:

- `target`: `time_since_onset` (default) measures how many hours have elapsed since a configurable onset hour; `time_to_onset` counts down instead.
- `infection_onset_hour`: first hour when infected wells truly diverge (e.g., 2.0 for B2 once infection starts at T2).
- `uninfected_target`: regression value assigned to control wells (often 0 for time-since-onset).
- `clamp_range`: optional min/max to keep targets bounded (e.g., `[0, 48]`).

During regression runs the script reports MAE and RMSE instead of classification metrics, and the best checkpoint is picked using RMSE.

## Diagnostics & threshold tuning

- **Threshold sweep:** when `analysis.thresholds` is non-empty, `train.py` logs precision/recall/F1 for each specified probability threshold per split. Use this to select an operating point (e.g., recall ≥ 0.7) before deploying the early-warning model.
- **Time bins:** `analysis.time_bins` accepts `[start_h, end_h]` ranges. The evaluator computes full classification metrics inside every bin, letting you pinpoint whether the model already performs well near 14–16 h but struggles before 8 h.
- **Balanced sampling:** set `data.balance_sampler: true` to construct a `WeightedRandomSampler` so each mini-batch has a more even positive/negative ratio. This is especially helpful for the 0–16 h experiment where infected frames are rare.
- **Re-testing checkpoints:** use `test_folds.py` to evaluate every saved `best.pt` in a run folder on any split:
   ```powershell
   cd CODE/cell_classification
   python test_folds.py `
       --config configs/resnet50_early.yaml `
       --run-dir checkpoints/resnet50_early/20251205-101010 `
       --split test
   ```
   The script rebuilds the datasets with the correct fold partition, loads the checkpoints, prints metrics (including threshold sweeps/time bins), and saves the summary to `fold_eval.json` inside the run directory (or a path you specify via `--output`).
   - PyTorch ≥ 2.6 users can add `--weights-only` for stricter deserialization; the script automatically allowlists `numpy.core.multiarray.scalar` (required by our checkpoints) and falls back to the legacy path if necessary.

### Interval sweep error bars

Use `analyze_interval_sweep.py` to probe how performance evolves as you tighten the infected interval upper bound. The script loads every fold checkpoint, rebuilds the requested split with custom frame policies, and plots mean ± std error bars for two scenarios: (1) both training/test restricted to `[start, x]`, and (2) training keeps the full range while evaluation clips to `[start, x]`.

```powershell
python analyze_interval_sweep.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --upper-hours 8 10 12 14 16 `
   --start-hour 1 `
   --metric auc `
   --split test
```

Outputs:

- `analysis/interval_sweep_<metric>.png`: two-panel error-bar chart.
- `analysis/interval_sweep_<metric>.json`: raw fold metrics + statistics for downstream plotting.

Pass `--weights-only` if your checkpoints require the safer PyTorch deserializer, or `--output-dir`/`--save-data` to customize where the artifacts land.

### Sliding window analysis

Use `analyze_sliding_window.py` to evaluate model performance across different time windows of fixed width. The script trains and evaluates on consecutive windows `[x, x+k]` where `k` is the window width (e.g., 5 or 10 hours) and `x` varies from the start to the end of the infection timeline.

```powershell
python analyze_sliding_window.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --window-width 5 `
   --start-hour 1 `
   --end-hour 30 `
   --metric auc `
   --split test
```

**Parameters:**
- `--window-width`: Size of the sliding window in hours (default: 5)
- `--start-hour`: Starting hour for the first window (default: 1)
- `--end-hour`: Maximum ending hour (default: 30)
- `--metric`: Which metric to plot (choices: accuracy, precision, recall, f1, auc; default: auc)
- `--split`: Dataset split to evaluate on (default: test)

**Outputs:**
- `analysis/sliding_window_<metric>_w<width>.png`: Error bar chart showing mean ± std for each window position
- `analysis/sliding_window_<metric>_w<width>.json`: Raw fold metrics and statistics

This is useful for understanding:
- Which time windows are most predictive for infection classification
- Whether performance is consistent across early/middle/late infection stages
- Optimal window placement for early detection systems

**Example use cases:**
```powershell
# Narrow 5-hour windows to find the most discriminative period
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-width 5 --metric auc

# Wider 10-hour windows for more stable estimates
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-width 10 --metric f1

# Focus on early infection (0-20 hours) with 3-hour windows
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-width 3 --start-hour 0 --end-hour 20
```

Shell scripts are provided for convenience:
- Unix/Linux: `shells/analyze_sliding_window.sh`
- Windows PowerShell: `shells/analyze_sliding_window.ps1`

## Customization tips

- **Frame selection**: Adjust `data.frames.strategy` and `data.frames.window_hours` in the YAML if you want to sample earlier or later time ranges, or switch to a fixed `target_frame`.
- **Model**: Swap to another ResNet depth by changing `model.name` (`resnet18`, `resnet34`, `resnet101`, `resnet152`).
- **Augmentations**: Toggle augmentation flags inside `data.transforms` (flips, rotations, color jitter).
- **Batching**: Increase `batch_size` and `num_workers` once you confirm GPU/CPU memory headroom.

Feel free to extend the pipeline with additional datasets, metrics, or foundation models as your experiments evolve.
