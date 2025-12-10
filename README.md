# Cell Classification Pipeline

This repository contains a PyTorch training pipeline that classifies live-cell imaging time-course TIFF stack   - PyTorch ≥ 2.6 users can add `--weights-only` for stricter deserialization; the script automatically allowlists `numpy.core.multiarray.scalar` (required by our checkpoints) and falls back to the legacy path if necessary.

## Advanced Analysis: Training-Based Window Evaluation

The following analysis scripts **train fresh models** for different time windows to discover which time periods contain the most informative signal for infection classification. These are compute-intensive but provide deeper insights than evaluation-only analysis.

### Interval sweep training analysis

Use `analyze_interval_sweep_train.py` to train models with different infected interval ranges and compare their performance. This reveals how much temporal information is needed for accurate classification.

```powershell
python analyze_interval_sweep_train.py `
   --config configs/resnet50_baseline.yaml `
   --upper-hours 8 10 12 14 16 18 20 `
   --start-hour 1 `
   --metrics auc accuracy f1 `
   --k-folds 5 `
   --epochs 10 `
   --split test
```

**What it does:**
- For each upper bound X, trains models using infected frames from [start, X]
- Runs in two modes:
  - **train-test**: Both train and test use [start, X]
  - **test-only**: Train uses full range, test restricted to [start, X]
- Reports metrics across K-fold cross-validation
- Generates two-panel plots comparing both modes

**Outputs:**
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_combined.png`: All metrics together
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_<metric>.png`: Individual two-panel plots
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_data.json`: Raw results
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_train.log`: Training log

**Shell scripts:**
- Unix/Linux: `bash shells/analyze_interval_sweep_train.sh`
- Windows PowerShell: `.\shells\analyze_interval_sweep_train.ps1`

### Sliding window training analysis

Use `analyze_sliding_window_train.py` to train models on different time windows [x, x+k] and identify which periods are most predictive.

```powershell
python analyze_sliding_window_train.py `
   --config configs/resnet50_baseline.yaml `
   --window-size 5 `
   --stride 5 `
   --start-hour 0 `
   --end-hour 30 `
   --metrics auc accuracy f1 `
   --k-folds 5 `
   --epochs 10 `
   --split test
```

**What it does:**
- For each window [x, x+k], trains a fresh model using only frames from that window
- Train/val/test all use the same time interval [x, x+k]
- Shows which time periods contain the strongest infection signal
- Supports overlapping windows via `--stride` parameter

**Parameters:**
- `--window-size`: Width of each window in hours (default: 5)
- `--stride`: Step between windows (default: window-size for no overlap)
  - `stride < window-size`: Overlapping windows
  - `stride = window-size`: Adjacent windows
  - `stride > window-size`: Gaps between windows
- `--k-folds`: Number of CV folds (default: from config)
- `--epochs`: Training epochs per window (default: from config)

**Outputs:**
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_combined.png`: All metrics
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_<metric>.png`: Individual plots
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_data.json`: Raw results
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_train.log`: Training log

**Example interpretation:**
```
Window [0,5]:   AUC = 0.65 ± 0.03  ← Early period, weak signal
Window [10,15]: AUC = 0.88 ± 0.02  ← Mid-infection, strong signal
Window [20,25]: AUC = 0.95 ± 0.01  ← Late infection, very strong signal
```
This shows that models trained on later time windows (20-25h) achieve the best performance, indicating that cytopathic effects become most discriminative after 20 hours.

**Shell scripts:**
- Unix/Linux: `bash shells/analyze_sliding_window_train.sh`
- Windows PowerShell: `.\shells\analyze_sliding_window_train.ps1`

## Evaluation-Only Analysis (Using Pre-Trained Checkpoints)

For quick analysis using pre-trained checkpoints without re-training, use the following scripts. These are much faster but don't reveal which time windows are optimal for training.

### Interval sweep error bars (evaluation-only)s **infected** or **uninfected**. Each TIFF file stores 95 time points (t0–t94) acquired every 30 minutes. Empirically, infected cells begin to show cytopathic effects (CPE) between 16–30 hours post-infection, so the default configuration samples frames within that window.

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
# Single metric
python analyze_interval_sweep.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --upper-hours 8 10 12 14 16 `
   --start-hour 1 `
   --metric auc `
   --split test

# Multiple metrics (creates combined + individual plots)
python analyze_interval_sweep.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --upper-hours 8 10 12 14 16 18 20 `
   --start-hour 1 `
   --metrics auc accuracy f1 precision recall `
   --split test
```

**Outputs:**

- `analysis/interval_sweep_combined.png`: Combined two-panel plot with all metrics (if `--metrics` used)
- `analysis/interval_sweep_<metric>.png`: Individual two-panel error-bar chart for each metric
- `analysis/interval_sweep_data.json`: Raw fold metrics + statistics for all metrics

The two panels show:
1. **Left panel**: Both train and test restricted to `[start, x]`
2. **Right panel**: Train uses full range, test restricted to `[start, x]`

Pass `--weights-only` if your checkpoints require the safer PyTorch deserializer, or `--output-dir`/`--save-data` to customize where the artifacts land.

### Sliding window analysis (evaluation-only)

Use `analyze_sliding_window.py` to evaluate pre-trained models on different time windows. The script loads existing checkpoints and tests them on data filtered to windows `[x, x+k]`.

```powershell
# Auto-generate windows with stride (supports overlap)
python analyze_sliding_window.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --window-size 5 `
   --start-hour 1 `
   --end-hour 30 `
   --stride 2 `
   --metrics auc accuracy f1 `
   --split test

# Or manually specify window positions
python analyze_sliding_window.py `
   --config configs/resnet50_baseline.yaml `
   --run-dir checkpoints/resnet50_baseline/20251205-101010 `
   --window-size 5 `
   --window-starts 0 5 10 15 20 25 `
   --metrics auc f1 `
   --split test
```

**Note:** This script evaluates pre-trained models. For training-based analysis (recommended for discovering optimal time windows), see [Sliding window training analysis](#sliding-window-training-analysis).

**Parameters:**
- `--run-dir`: Path to checkpoint directory with fold subfolders
- `--window-size`: Size of the sliding window in hours (default: 5)
- `--stride`: Step size between consecutive windows (default: window-size for no overlap)
  - `stride < window-size`: Creates overlapping windows
  - `stride = window-size`: Adjacent windows with no gap or overlap
  - `stride > window-size`: Creates gaps between windows
- `--start-hour`: Starting hour for first window (default: 1, used with auto-generation)
- `--end-hour`: Maximum ending hour (default: 30, used with auto-generation)
- `--window-starts`: Manual list of window start positions (alternative to auto-generation)
- `--metrics`: Multiple metrics to plot (e.g., `auc accuracy f1`). Creates both combined and individual plots
- `--metric`: Single metric (default: auc, for backward compatibility)
- `--split`: Dataset split to evaluate on (default: test)

**Outputs:**
- `analysis/sliding_window_w<width>_s<stride>_combined.png`: Combined plot with all metrics (if multiple metrics)
- `analysis/sliding_window_w<width>_s<stride>_<metric>.png`: Individual plot for each metric
- `analysis/sliding_window_w<width>_s<stride>_data.json`: Raw fold metrics and statistics

This is useful for understanding:
- Which time windows are most predictive for infection classification
- Whether performance is consistent across early/middle/late infection stages
- Optimal window placement and overlap strategies for early detection systems
- Comparative performance across multiple metrics simultaneously

**Example use cases:**
```powershell
# Overlapping 5-hour windows (stride=2) with multiple metrics
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 5 --stride 2 --start-hour 0 --end-hour 30 --metrics auc accuracy f1

# Adjacent 10-hour windows (no overlap) with AUC only
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 10 --stride 10 --metrics auc

# Custom window positions with multiple metrics
python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 6 --window-starts 0 4 8 12 16 20 --metrics auc precision recall
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
