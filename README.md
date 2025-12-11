# Cell Classification Pipeline# Cell Classification Pipeline



A PyTorch training pipeline for classifying live-cell imaging time-course TIFF stacks as **infected** or **uninfected**. Each TIFF stores 95 time points (t0–t94) acquired every 30 minutes. Infected cells show cytopathic effects (CPE) between 16–30 hours post-infection.This repository contains a PyTorch training pipeline that classifies live-cell imaging time-course TIFF stack   - PyTorch ≥ 2.6 users can add `--weights-only` for stricter deserialization; the script automatically allowlists `numpy.core.multiarray.scalar` (required by our checkpoints) and falls back to the legacy path if necessary.



## Table of Contents## Advanced Analysis: Training-Based Window Evaluation



- [Quick Start](#quick-start)The following analysis scripts **train fresh models** for different time windows to discover which time periods contain the most informative signal for infection classification. These are compute-intensive but provide deeper insights than evaluation-only analysis.

- [Project Structure](#project-structure)

- [Basic Training](#basic-training)### Interval sweep training analysis

- [Advanced Analysis](#advanced-analysis)

  - [Training-Based Window Analysis](#training-based-window-analysis)Use `analyze_interval_sweep_train.py` to train models with different infected interval ranges and compare their performance. This reveals how much temporal information is needed for accurate classification.

  - [Evaluation-Only Analysis](#evaluation-only-analysis)

- [Cross-Validation](#cross-validation)```powershell

- [Visualization](#visualization)python analyze_interval_sweep_train.py `

- [Configuration](#configuration)   --config configs/resnet50_baseline.yaml `

   --upper-hours 8 10 12 14 16 18 20 `

---   --start-hour 1 `

   --metrics auc accuracy f1 `

## Quick Start   --k-folds 5 `

   --epochs 10 `

```powershell   --split test

# 1. Install dependencies```

python -m venv .venv

.venv\Scripts\Activate.ps1**What it does:**

pip install -r requirements.txt- For each upper bound X, trains models using infected frames from [start, X]

- Runs in two modes:

# 2. Run basic training  - **train-test**: Both train and test use [start, X]

python train.py --config configs/resnet50_baseline.yaml  - **test-only**: Train uses full range, test restricted to [start, X]

- Reports metrics across K-fold cross-validation

# 3. Or run time window analysis (trains models for each window)- Generates two-panel plots comparing both modes

bash shells/analyze_sliding_window_train.sh

```**Outputs:**

- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_combined.png`: All metrics together

---- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_<metric>.png`: Individual two-panel plots

- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_data.json`: Raw results

## Project Structure- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_train.log`: Training log

- `outputs/interval_sweep_analysis/<timestamp>/checkpoints/*/fold_*_best.pth`: **Trained model weights** for each interval/mode and fold

```

cell_classification/**Shell scripts:**

├── configs/                      # Experiment configurations- Unix/Linux: `bash shells/analyze_interval_sweep_train.sh`

│   ├── resnet50_baseline.yaml   # Default: 16-30h infection window- Windows PowerShell: `.\shells\analyze_interval_sweep_train.ps1`

│   ├── resnet50_early.yaml      # Early detection: 0-16h window

│   └── resnet50_time_regression.yaml### Sliding window training analysis

├── datasets/

│   └── timecourse_dataset.py    # TIFF dataset + split utilitiesUse `analyze_sliding_window_train.py` to train models on different time windows [x, x+k] and identify which periods are most predictive.

├── models/

│   └── resnet.py                # ResNet classifier```powershell

├── utils/                        # Logging, metrics, transformspython analyze_sliding_window_train.py `

├── train.py                      # Main training script   --config configs/resnet50_baseline.yaml `

├── analyze_sliding_window_train.py    # Window analysis (trains models)   --window-size 5 `

├── analyze_interval_sweep_train.py    # Interval analysis (trains models)   --stride 5 `

├── analyze_sliding_window.py          # Eval-only (uses checkpoints)   --start-hour 0 `

├── analyze_interval_sweep.py          # Eval-only (uses checkpoints)   --end-hour 30 `

├── visualize_cam.py              # Grad-CAM heatmaps   --metrics auc accuracy f1 `

└── test_folds.py                 # Re-evaluate saved checkpoints   --k-folds 5 `

```   --epochs 10 `

   --split test

**Data Structure:**```

```

DATA_ROOT/**What it does:**

├── infected/- For each window [x, x+k], trains a fresh model using only frames from that window

│   └── *.tiff     # Multi-frame stacks (infected wells)- Train/val/test all use the same time interval [x, x+k]

└── uninfected/- Shows which time periods contain the strongest infection signal

    └── *.tiff     # Control stacks- Supports overlapping windows via `--stride` parameter

```

**Parameters:**

Update paths in `configs/resnet50_baseline.yaml`:- `--window-size`: Width of each window in hours (default: 5)

```yaml- `--stride`: Step between windows (default: window-size for no overlap)

data:  - `stride < window-size`: Overlapping windows

  infected_dir: "/path/to/infected/"  - `stride = window-size`: Adjacent windows

  uninfected_dir: "/path/to/uninfected/"  - `stride > window-size`: Gaps between windows

```- `--k-folds`: Number of CV folds (default: from config)

- `--epochs`: Training epochs per window (default: from config)

---

**Outputs:**

## Basic Training- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_combined.png`: All metrics

- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_<metric>.png`: Individual plots

### Standard Training- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_data.json`: Raw results

- `outputs/sliding_window_analysis/<timestamp>/sliding_window_train.log`: Training log

```powershell- `outputs/sliding_window_analysis/<timestamp>/checkpoints/window_*/fold_*_best.pth`: **Trained model weights** for each window and fold

python train.py --config configs/resnet50_baseline.yaml

```**Example interpretation:**

```

**Output:**Window [0,5]:   AUC = 0.65 ± 0.03  ← Early period, weak signal

- Checkpoints: `checkpoints/<experiment>/<timestamp>/`Window [10,15]: AUC = 0.88 ± 0.02  ← Mid-infection, strong signal

- Logs: `outputs/<experiment>/<timestamp>/`Window [20,25]: AUC = 0.95 ± 0.01  ← Late infection, very strong signal

```

### Early Detection TrainingThis shows that models trained on later time windows (20-25h) achieve the best performance, indicating that cytopathic effects become most discriminative after 20 hours.



Train on 0-16h window (before visible CPE):**Shell scripts:**

- Unix/Linux: `bash shells/analyze_sliding_window_train.sh`

```powershell- Windows PowerShell: `.\shells\analyze_sliding_window_train.ps1`

python train.py --config configs/resnet50_early.yaml

```## Evaluation-Only Analysis (Using Pre-Trained Checkpoints)



### Time RegressionFor quick analysis using pre-trained checkpoints without re-training, use the following scripts. These are much faster but don't reveal which time windows are optimal for training.



Predict time-to-infection instead of binary classification:### Interval sweep error bars (evaluation-only)s **infected** or **uninfected**. Each TIFF file stores 95 time points (t0–t94) acquired every 30 minutes. Empirically, infected cells begin to show cytopathic effects (CPE) between 16–30 hours post-infection, so the default configuration samples frames within that window.



```powershell## Project layout

python train.py --config configs/resnet50_time_regression.yaml

``````

cell_classification/

---├── configs/

│   └── resnet50_baseline.yaml   # Experiment settings

## Advanced Analysis├── datasets/

│   └── timecourse_dataset.py    # TIFF dataset + split utilities

### Training-Based Window Analysis├── models/

│   └── resnet.py                # ResNet classifier wrapper

**Goal:** Train models on different time windows to discover which periods are most informative.├── utils/

│   ├── config.py                # YAML loader

#### 1. Sliding Window Analysis│   ├── logger.py                # Console/file logger

│   ├── metrics.py               # Average meters + binary metrics

Trains models for each window [x, x+k]:│   ├── seed.py                  # Seed all RNGs

│   └── transforms.py            # Build torchvision pipelines

```powershell├── requirements.txt             # Python dependencies

python analyze_sliding_window_train.py \└── train.py                     # Main training entrypoint

   --config configs/resnet50_baseline.yaml \```

   --window-size 6 \

   --stride 3 \## Data expectations

   --start-hour 1 \

   --end-hour 46 \```

   --metrics auc accuracy f1 \DATA_ROOT/

   --k-folds 5 \├── infected/

   --epochs 10│   ├── *.tiff   (multi-frame stacks containing infected wells)

```├── uninfected/

│   └── *.tiff   (control stacks)

**What it does:**```

- Window [0,5]: Trains model on frames 0-5h only

- Window [5,10]: Trains model on frames 5-10h only  Update the paths within `configs/resnet50_baseline.yaml` to point at your infected/uninfected directories. The default config already references the provided UNC paths.

- Window [10,15]: Trains model on frames 10-15h only

- ...and so on### Frame expansion logic



**Output:**- **Infected stacks (default)**: every TIFF is expanded into multiple samples covering **all frames between 16 h and 30 h** (based on `frames_per_hour`, default 2). This captures the CPE window instead of just one snapshot.

```- **Uninfected stacks**: every frame from t0–t92 is used (configurable via `uninfected_use_all` / `uninfected_stride`).

Window [0,5]:   AUC = 0.65 ± 0.03  ← Weak early signal- Adjust these rules by editing the `data.frames` block in the YAML. For example, change `infected_window_hours` or set a stride > 1 to subsample. The early-detection config (`configs/resnet50_early.yaml`) simply switches this window to `[0, 16]` to focus on subtle pre-CPE cues.

Window [10,15]: AUC = 0.88 ± 0.02  ← Stronger mid-period- Need split-specific behavior? Provide overrides under `data.frames.train`, `data.frames.val`, or `data.frames.test` (optionally `data.frames.default`). Each override inherits unspecified fields from the base block, letting you train on `[1, 30]` while evaluating on `[1, 12]` without touching the shared config.

Window [20,25]: AUC = 0.95 ± 0.01  ← Best late-period- Set `data.balance_sampler: true` when you want a class-balanced `WeightedRandomSampler` during training (useful for early windows where positives are scarce).

```

## Quick start

**Files saved:**

- `outputs/sliding_window_analysis/<timestamp>/`1. **Install dependencies**

  - `sliding_window_w6_s3_combined.png` - All metrics together   ```powershell

  - `sliding_window_w6_s3_auc.png` - AUC plot   python -m venv .venv

  - `sliding_window_w6_s3_data.json` - Raw data   .venv\Scripts\Activate.ps1

  - `checkpoints/window_*/fold_*_best.pth` - Trained models   pip install -r requirements.txt

   ```

**Shell scripts:**

- Bash: `bash shells/analyze_sliding_window_train.sh`2. **Launch training**

- PowerShell: `.\shells\analyze_sliding_window_train.ps1`   ```powershell

   python train.py --config configs/resnet50_baseline.yaml

#### 2. Interval Sweep Analysis   ```



Trains models with different upper bounds [start, X]:Training artifacts (logs and checkpoints) are written under `outputs/` and `checkpoints/` respectively. The script automatically evaluates on the validation split every epoch and reports accuracy, precision, recall, F1, and AUC. After training, it runs a final evaluation on the held-out test split.



```powershell> **Note**: The TIFF stacks use LZW compression, so the Python wheel `imagecodecs` must be installed (already listed in `requirements.txt`). If you install dependencies manually, ensure `pip install imagecodecs` succeeds or tifffile will raise `COMPRESSION.LZW requires the 'imagecodecs' package`.

python analyze_interval_sweep_train.py \

   --config configs/resnet50_baseline.yaml \## Cross-validation & experiment outputs

   --upper-hours 8 10 12 14 16 18 20 \

   --start-hour 1 \- Set `training.k_folds` in the YAML to a value > 1 (e.g., 5) to enable K-fold cross-validation. The script will iteratively train on each fold, log per-fold metrics, and print an aggregate summary at the end.

   --metrics auc accuracy f1 \- Every run receives a timestamp-based `run_id`, so logs/checkpoints live under `outputs/<experiment>/<run_id>/fold_XXofYY` and `checkpoints/<experiment>/<run_id>/fold_XXofYY`. This keeps concurrent experiments isolated.

   --k-folds 5 \- Within each fold you still get the best checkpoint (`best.pt`) plus detailed logs showing how many frame-level samples were drawn from infected vs uninfected stacks.

   --epochs 10- After training with `k_folds > 1`, a `cv_summary.json` file is written in the run's output directory with the full per-fold/aggregate metrics so you can revisit the results without rerunning.

```- Enable fine-grained diagnostics by adding an `analysis` block to your config:

   ```yaml

**What it does:**   analysis:

- Two modes per interval:      thresholds: [0.05, 0.1, 0.2, 0.3]

  - **train-test**: Both use [1, X]h      time_bins:

  - **test-only**: Train uses all data, test uses [1, X]h         - [0, 6]

- Shows how much temporal data is needed         - [6, 10]

         - [10, 14]

**Files saved:**         - [14, 16]

- `outputs/interval_sweep_analysis/<timestamp>/`   ```

  - `interval_sweep_combined.png` - Two-panel comparison   During evaluation the trainer prints a threshold sweep table (precision/recall/F1 at each cut-off) plus per-bin metrics so you can see exactly which time windows remain challenging.

  - `interval_sweep_auc.png` - AUC two-panel plot

  - `checkpoints/*/fold_*_best.pth` - Trained models## Visualizing Grad-CAM heatmaps



**Shell scripts:**Use `visualize_cam.py` to inspect which regions drive the classifier’s decision:

- Bash: `bash shells/analyze_interval_sweep_train.sh`

- PowerShell: `.\shells\analyze_interval_sweep_train.ps1````powershell

python visualize_cam.py `

---   --config configs/resnet50_baseline.yaml `

   --checkpoint checkpoints/resnet50_baseline/20251204-141200/fold_01of05/best.pt `

### Evaluation-Only Analysis   --split val `

   --num-samples 8 `

**Goal:** Quickly test pre-trained models on different time windows (no re-training).   --num-folds 5 `

   --fold-index 0

#### Sliding Window (Eval-Only)```



```powershell- The script samples random frames from the requested split, generates Grad-CAM overlays, and writes raw/input/heatmap/overlay PNGs plus `metadata.json` under `cam_outputs/<checkpoint_name>/<split>_foldX/`.

python analyze_sliding_window.py \- Filenames now encode the ground-truth label and infection probability (e.g., `cam_003_infected_prob0.92_overlay.png`), and each overlay carries a text banner (`Label / Pred / P(infected)`) so you can inspect ground truth at a glance.

   --config configs/resnet50_baseline.yaml \- `--num-folds`/`--fold-index` should mirror the training run if you want CAMs from the same fold partition; leave them at defaults to ignore CV.

   --run-dir checkpoints/resnet50_baseline/20251205-101010 \- Adjust `--num-samples`, `--split`, or `--cmap` to explore different frames.

   --window-size 5 \- If you intentionally want PyTorch 2.6's safer `weights_only=True` behavior, pass `--weights-only`; by default the script disables it (`weights_only=False`) so legacy checkpoints load without extra allow-list setup.

   --stride 2 \

   --metrics auc accuracy f1## Advanced training modes

```

### Early-infection classifier

#### Interval Sweep (Eval-Only)

Use `configs/resnet50_early.yaml` to restrict infected samples to the 0–16 h window:

```powershell

python analyze_interval_sweep.py \```powershell

   --config configs/resnet50_baseline.yaml \python train.py --config configs/resnet50_early.yaml

   --run-dir checkpoints/resnet50_baseline/20251205-101010 \```

   --upper-hours 8 10 12 14 16 18 20 \

   --metrics auc accuracy f1This keeps the same model/optimizer but trains on frames where infection cues are subtle. Compare its ROC-AUC against the baseline to understand how early the signal becomes reliable.

```

### Time-to-infection regression

**When to use:**

- ✅ Have existing checkpoints`train.py` also supports a regression mode. Set `task.type: regression` and describe the target in `task.regression` (see `configs/resnet50_time_regression.yaml`). Example:

- ✅ Want quick analysis

- ❌ Don't need to train fresh models```powershell

python train.py --config configs/resnet50_time_regression.yaml

---```



## Cross-ValidationKey settings:



Enable K-fold CV in config:- `target`: `time_since_onset` (default) measures how many hours have elapsed since a configurable onset hour; `time_to_onset` counts down instead.

- `infection_onset_hour`: first hour when infected wells truly diverge (e.g., 2.0 for B2 once infection starts at T2).

```yaml- `uninfected_target`: regression value assigned to control wells (often 0 for time-since-onset).

training:- `clamp_range`: optional min/max to keep targets bounded (e.g., `[0, 48]`).

  k_folds: 5

```During regression runs the script reports MAE and RMSE instead of classification metrics, and the best checkpoint is picked using RMSE.



Or override from command line:## Diagnostics & threshold tuning

```powershell

python train.py --config configs/resnet50_baseline.yaml --k-folds 5- **Threshold sweep:** when `analysis.thresholds` is non-empty, `train.py` logs precision/recall/F1 for each specified probability threshold per split. Use this to select an operating point (e.g., recall ≥ 0.7) before deploying the early-warning model.

```- **Time bins:** `analysis.time_bins` accepts `[start_h, end_h]` ranges. The evaluator computes full classification metrics inside every bin, letting you pinpoint whether the model already performs well near 14–16 h but struggles before 8 h.

- **Balanced sampling:** set `data.balance_sampler: true` to construct a `WeightedRandomSampler` so each mini-batch has a more even positive/negative ratio. This is especially helpful for the 0–16 h experiment where infected frames are rare.

**Output:**- **Re-testing checkpoints:** use `test_folds.py` to evaluate every saved `best.pt` in a run folder on any split:

- Per-fold checkpoints: `checkpoints/<experiment>/<timestamp>/fold_01of05/best.pt`   ```powershell

- CV summary: `outputs/<experiment>/<timestamp>/cv_summary.json`   cd CODE/cell_classification

   python test_folds.py `

**Re-test all folds:**       --config configs/resnet50_early.yaml `

```powershell       --run-dir checkpoints/resnet50_early/20251205-101010 `

python test_folds.py \       --split test

   --config configs/resnet50_baseline.yaml \   ```

   --run-dir checkpoints/resnet50_baseline/20251205-101010 \   The script rebuilds the datasets with the correct fold partition, loads the checkpoints, prints metrics (including threshold sweeps/time bins), and saves the summary to `fold_eval.json` inside the run directory (or a path you specify via `--output`).

   --split test   - PyTorch ≥ 2.6 users can add `--weights-only` for stricter deserialization; the script automatically allowlists `numpy.core.multiarray.scalar` (required by our checkpoints) and falls back to the legacy path if necessary.

```

### Interval sweep error bars

---

Use `analyze_interval_sweep.py` to probe how performance evolves as you tighten the infected interval upper bound. The script loads every fold checkpoint, rebuilds the requested split with custom frame policies, and plots mean ± std error bars for two scenarios: (1) both training/test restricted to `[start, x]`, and (2) training keeps the full range while evaluation clips to `[start, x]`.

## Visualization

```powershell

### Grad-CAM Heatmaps# Single metric

python analyze_interval_sweep.py `

Visualize which regions the model focuses on:   --config configs/resnet50_baseline.yaml `

   --run-dir checkpoints/resnet50_baseline/20251205-101010 `

```powershell   --upper-hours 8 10 12 14 16 `

python visualize_cam.py \   --start-hour 1 `

   --config configs/resnet50_baseline.yaml \   --metric auc `

   --checkpoint checkpoints/resnet50_baseline/20251205-101010/fold_01of05/best.pt \   --split test

   --split val \

   --num-samples 8# Multiple metrics (creates combined + individual plots)

```python analyze_interval_sweep.py `

   --config configs/resnet50_baseline.yaml `

**Output:** `cam_outputs/<checkpoint>/<split>/`   --run-dir checkpoints/resnet50_baseline/20251205-101010 `

- Raw images   --upper-hours 8 10 12 14 16 18 20 `

- Heatmaps   --start-hour 1 `

- Overlays with predictions   --metrics auc accuracy f1 precision recall `

   --split test

### Load Saved Checkpoints```



Use the example script to load and inspect saved models:**Outputs:**



```powershell- `analysis/interval_sweep_combined.png`: Combined two-panel plot with all metrics (if `--metrics` used)

python load_checkpoint_example.py- `analysis/interval_sweep_<metric>.png`: Individual two-panel error-bar chart for each metric

```- `analysis/interval_sweep_data.json`: Raw fold metrics + statistics for all metrics



This shows how to:The two panels show:

- Load model weights1. **Left panel**: Both train and test restricted to `[start, x]`

- Find best performing window2. **Right panel**: Train uses full range, test restricted to `[start, x]`

- Compare all windows

Pass `--weights-only` if your checkpoints require the safer PyTorch deserializer, or `--output-dir`/`--save-data` to customize where the artifacts land.

---

### Sliding window analysis (evaluation-only)

## Configuration

Use `analyze_sliding_window.py` to evaluate pre-trained models on different time windows. The script loads existing checkpoints and tests them on data filtered to windows `[x, x+k]`.

### Key Config Options

```powershell

```yaml# Auto-generate windows with stride (supports overlap)

data:python analyze_sliding_window.py `

  batch_size: 256                      # Training batch size   --config configs/resnet50_baseline.yaml `

  eval_batch_size_multiplier: 2        # Eval uses 512 (256*2) for speed   --run-dir checkpoints/resnet50_baseline/20251205-101010 `

  num_workers: 4                       # CPU workers for data loading   --window-size 5 `

  balance_sampler: false               # Use weighted sampler for class balance   --start-hour 1 `

     --end-hour 30 `

  frames:   --stride 2 `

    frames_per_hour: 2.0   --metrics auc accuracy f1 `

    infected_window_hours: [16, 30]    # CPE window   --split test

    infected_stride: 1

    uninfected_use_all: true# Or manually specify window positions

python analyze_sliding_window.py `

model:   --config configs/resnet50_baseline.yaml `

  name: resnet50                       # resnet18/34/50/101/152   --run-dir checkpoints/resnet50_baseline/20251205-101010 `

   --window-size 5 `

training:   --window-starts 0 5 10 15 20 25 `

  epochs: 30   --metrics auc f1 `

  k_folds: 1                           # Set >1 for cross-validation   --split test

  amp: true                            # Automatic mixed precision```

  

optimizer:**Note:** This script evaluates pre-trained models. For training-based analysis (recommended for discovering optimal time windows), see [Sliding window training analysis](#sliding-window-training-analysis).

  lr: 1e-4

  weight_decay: 1e-5**Parameters:**

- `--run-dir`: Path to checkpoint directory with fold subfolders

scheduler:- `--window-size`: Size of the sliding window in hours (default: 5)

  type: cosine- `--stride`: Step size between consecutive windows (default: window-size for no overlap)

  t_max: 30  - `stride < window-size`: Creates overlapping windows

  - `stride = window-size`: Adjacent windows with no gap or overlap

task:  - `stride > window-size`: Creates gaps between windows

  type: classification                 # or "regression"- `--start-hour`: Starting hour for first window (default: 1, used with auto-generation)

```- `--end-hour`: Maximum ending hour (default: 30, used with auto-generation)

- `--window-starts`: Manual list of window start positions (alternative to auto-generation)

### Frame Policy Overrides- `--metrics`: Multiple metrics to plot (e.g., `auc accuracy f1`). Creates both combined and individual plots

- `--metric`: Single metric (default: auc, for backward compatibility)

Different time windows for train/val/test:- `--split`: Dataset split to evaluate on (default: test)



```yaml**Outputs:**

data:- `analysis/sliding_window_w<width>_s<stride>_combined.png`: Combined plot with all metrics (if multiple metrics)

  frames:- `analysis/sliding_window_w<width>_s<stride>_<metric>.png`: Individual plot for each metric

    frames_per_hour: 2.0- `analysis/sliding_window_w<width>_s<stride>_data.json`: Raw fold metrics and statistics

    infected_window_hours: [1, 30]     # Default for all splits

    This is useful for understanding:

    # Override for specific splits- Which time windows are most predictive for infection classification

    test:- Whether performance is consistent across early/middle/late infection stages

      infected_window_hours: [1, 12]   # Test early detection only- Optimal window placement and overlap strategies for early detection systems

```- Comparative performance across multiple metrics simultaneously



### Performance Tuning**Example use cases:**

```powershell

**Batch Sizes:**# Overlapping 5-hour windows (stride=2) with multiple metrics

- Training uses `batch_size` (e.g., 256)python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 5 --stride 2 --start-hour 0 --end-hour 30 --metrics auc accuracy f1

- Eval uses `batch_size * eval_batch_size_multiplier` (e.g., 512)

- Eval can be 2-3x larger since no gradients are computed# Adjacent 10-hour windows (no overlap) with AUC only

python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 10 --stride 10 --metrics auc

**Recommended multipliers:**

- 16+ GB GPU: `eval_batch_size_multiplier: 3`# Custom window positions with multiple metrics

- 8-16 GB GPU: `eval_batch_size_multiplier: 2` (default)python analyze_sliding_window.py --config configs/resnet50_baseline.yaml --run-dir checkpoints/resnet50_baseline/20251205-101010 --window-size 6 --window-starts 0 4 8 12 16 20 --metrics auc precision recall

- <8 GB GPU: `eval_batch_size_multiplier: 1````



### Analysis DiagnosticsShell scripts are provided for convenience:

- Unix/Linux: `shells/analyze_sliding_window.sh`

```yaml- Windows PowerShell: `shells/analyze_sliding_window.ps1`

analysis:

  thresholds: [0.1, 0.3, 0.5, 0.7, 0.9]   # Precision/recall at each threshold## Customization tips

  time_bins:                               # Metrics per time window

    - [0, 6]- **Frame selection**: Adjust `data.frames.strategy` and `data.frames.window_hours` in the YAML if you want to sample earlier or later time ranges, or switch to a fixed `target_frame`.

    - [6, 12]- **Model**: Swap to another ResNet depth by changing `model.name` (`resnet18`, `resnet34`, `resnet101`, `resnet152`).

    - [12, 18]- **Augmentations**: Toggle augmentation flags inside `data.transforms` (flips, rotations, color jitter).

    - [18, 24]- **Batching**: 

```  - Increase `batch_size` to utilize GPU/CPU memory headroom during training

  - Set `eval_batch_size_multiplier` (default: 2) to use larger batches during validation/test for faster evaluation

---    - Training batch size: `batch_size` (e.g., 256)

    - Eval batch size: `batch_size * eval_batch_size_multiplier` (e.g., 512)

## Customization Tips    - Since eval doesn't compute gradients, it can use 2-3x larger batches

  - Adjust `num_workers` based on your CPU cores

- **Models**: Change `model.name` to `resnet18/34/101/152`

- **Augmentations**: Toggle in `data.transforms` (flips, rotations, color jitter)Feel free to extend the pipeline with additional datasets, metrics, or foundation models as your experiments evolve.

- **Frame selection**: Adjust `data.frames.infected_window_hours`
- **Early detection**: Use `configs/resnet50_early.yaml` for 0-16h window
- **Batch sizes**: Increase based on GPU memory

---

## Notes

- **TIFF compression**: Requires `imagecodecs` package (included in `requirements.txt`)
- **PyTorch 2.6+**: Use `--weights-only` flag for safer checkpoint loading
- **Frame expansion**: Each TIFF is expanded into multiple samples based on frame policy
- **Checkpoints**: Automatically save best model per fold based on validation metric

---

## Examples

**Quick training run:**
```powershell
python train.py --config configs/resnet50_baseline.yaml
```

**5-fold cross-validation:**
```powershell
python train.py --config configs/resnet50_baseline.yaml --k-folds 5
```

**Find best time window:**
```powershell
bash shells/analyze_sliding_window_train.sh
```

**Visualize decisions:**
```powershell
python visualize_cam.py --checkpoint checkpoints/.../best.pt --split val
```
