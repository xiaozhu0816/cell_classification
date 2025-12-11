# Analysis Scripts: Training vs Evaluation-Only

## Overview

This project now has **TWO types** of analysis scripts:

1. **Training-Based Analysis** (NEW) - Trains fresh models for each time window
2. **Evaluation-Only Analysis** (LEGACY) - Evaluates pre-trained checkpoints

## Key Difference

### ❌ What I Implemented Wrong Initially:
- `analyze_sliding_window.py` - Loads existing checkpoints and evaluates on filtered data
- `analyze_interval_sweep.py` - Loads existing checkpoints and evaluates on filtered data
- **Problem**: Only shows how pre-trained models perform on different time windows, doesn't tell you which windows are best for training

### ✅ What You Actually Needed:
- `analyze_sliding_window_train.py` - **Trains new models** for each time window [x, x+k]
- `analyze_interval_sweep_train.py` - **Trains new models** for each interval [start, x]
- **Benefit**: Shows which time periods contain the most informative signal for learning

## New Files Created

### 1. Training-Based Analysis Scripts

#### `analyze_sliding_window_train.py`
**Purpose**: Train separate models on different time windows to find the most informative periods.

**What it does:**
- For each window [x, x+k]:
  - Filters train/val/test to only use frames from [x, x+k]
  - Trains a brand new model from scratch
  - Tests on the same window
  - Records metrics across K-folds
- Plots performance vs. window position

**Example output:**
```
Window [0,5]:   AUC = 0.65 ± 0.03  ← Early, weak signal
Window [10,15]: AUC = 0.88 ± 0.02  ← Mid-infection, strong
Window [20,25]: AUC = 0.95 ± 0.01  ← Late, very strong
```
**Interpretation**: Models trained on 20-25h windows perform best, so cytopathic effects are most discriminative after 20h.

**Usage:**
```bash
python analyze_sliding_window_train.py \
    --config configs/resnet50_baseline.yaml \
    --window-size 5 \
    --stride 5 \
    --start-hour 0 \
    --end-hour 30 \
    --metrics auc accuracy f1 \
    --k-folds 5 \
    --epochs 10
```

**Shell scripts:**
- `shells/analyze_sliding_window_train.sh`
- `shells/analyze_sliding_window_train.ps1`

#### `analyze_interval_sweep_train.py`
**Purpose**: Train models with different infected interval ranges to understand how much temporal data is needed.

**What it does:**
- For each upper bound X:
  - Mode 1 (train-test): Both train and test use [start, X]
  - Mode 2 (test-only): Train uses full range, test uses [start, X]
  - Trains fresh models for both modes
  - Compares performance across K-folds
- Generates two-panel comparison plots

**Usage:**
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 \
    --start-hour 1 \
    --metrics auc accuracy f1 \
    --k-folds 5 \
    --epochs 10
```

**Shell scripts:**
- `shells/analyze_interval_sweep_train.sh`
- `shells/analyze_interval_sweep_train.ps1`

### 2. Bug Fixes

#### `utils/metrics.py`
**Fixed**: ROC AUC warning when only one class is present

**Before:**
```python
try:
    results["auc"] = metrics.roc_auc_score(labels, probs)
except ValueError:
    results["auc"] = float("nan")
```

**After:**
```python
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")
        results["auc"] = metrics.roc_auc_score(labels, probs)
except (ValueError, RuntimeWarning):
    results["auc"] = float("nan")
```

**Benefit**: Suppresses the sklearn warning about undefined ROC AUC when only one class is present in a batch.

## File Structure

```
cell_classification/
├── analyze_sliding_window.py              # LEGACY: Evaluation-only
├── analyze_sliding_window_train.py        # NEW: Full training
├── analyze_interval_sweep.py              # LEGACY: Evaluation-only  
├── analyze_interval_sweep_train.py        # NEW: Full training
├── shells/
│   ├── analyze_sliding_window.sh          # LEGACY: For evaluation-only
│   ├── analyze_sliding_window.ps1         # LEGACY: For evaluation-only
│   ├── analyze_sliding_window_train.sh    # NEW: For training
│   ├── analyze_sliding_window_train.ps1   # NEW: For training
│   ├── analyze_interval_sweep_train.sh    # NEW: For training
│   └── analyze_interval_sweep_train.ps1   # NEW: For training
└── utils/
    └── metrics.py                         # FIXED: ROC AUC warning suppression
```

## When to Use Which

### Use Training-Based Scripts When:
- ✅ You want to discover which time windows are most informative for **training**
- ✅ You want to understand which periods contain the strongest infection signal
- ✅ You're designing an early detection system and need to know optimal time ranges
- ✅ You have compute resources and time for full training runs
- ✅ You want definitive answers about temporal information content

### Use Evaluation-Only Scripts When:
- ✅ You already have trained checkpoints and want quick analysis
- ✅ You want to test how existing models perform on different time ranges
- ✅ You need fast turnaround without re-training
- ✅ You're debugging or doing exploratory analysis

## Performance Considerations

### Training-Based Analysis:
- **Time**: Hours to days (trains multiple models × K-folds)
- **Compute**: Requires GPU for reasonable speed
- **Example**: 6 windows × 5 folds × 10 epochs ≈ 300 training runs
- **Benefit**: Authoritative results on optimal time windows

### Evaluation-Only Analysis:
- **Time**: Minutes
- **Compute**: Can run on CPU
- **Example**: 6 windows × 5 folds ≈ 30 evaluations
- **Benefit**: Fast iteration for hypothesis testing

## Updated README Sections

The README now has two main analysis sections:

1. **Advanced Analysis: Training-Based Window Evaluation** (NEW)
   - Interval sweep training analysis
   - Sliding window training analysis
   - With interpretation examples

2. **Evaluation-Only Analysis (Using Pre-Trained Checkpoints)** (LEGACY)
   - Interval sweep error bars (evaluation-only)
   - Sliding window analysis (evaluation-only)
   - Clearly marked as "legacy" for existing checkpoints

## Quick Start Examples

### Training-Based Sliding Window Analysis:
```bash
# Edit and run the shell script
bash shells/analyze_sliding_window_train.sh

# Or directly:
python analyze_sliding_window_train.py \
    --window-size 5 --stride 5 \
    --metrics auc accuracy f1 \
    --k-folds 5 --epochs 10
```

### Training-Based Interval Sweep Analysis:
```bash
# Edit and run the shell script
bash shells/analyze_interval_sweep_train.sh

# Or directly:
python analyze_interval_sweep_train.py \
    --upper-hours 8 10 12 14 16 18 20 \
    --metrics auc accuracy f1 \
    --k-folds 5 --epochs 10
```

## Output Locations

### Training-Based Scripts:
```
outputs/
├── sliding_window_analysis/
│   └── <timestamp>/
│       ├── sliding_window_w5_s5_combined.png
│       ├── sliding_window_w5_s5_auc.png
│       ├── sliding_window_w5_s5_data.json
│       ├── sliding_window_train.log
│       └── checkpoints/
│           ├── window_0-5/
│           │   ├── fold_01_best.pth
│           │   ├── fold_02_best.pth
│           │   └── ...
│           ├── window_5-10/
│           │   └── fold_*_best.pth
│           └── ...
└── interval_sweep_analysis/
    └── <timestamp>/
        ├── interval_sweep_combined.png
        ├── interval_sweep_auc.png
        ├── interval_sweep_data.json
        ├── interval_sweep_train.log
        └── checkpoints/
            ├── train-test_interval_1-8/
            │   └── fold_*_best.pth
            ├── train-test_interval_1-10/
            │   └── fold_*_best.pth
            ├── test-only_interval_1-8/
            │   └── fold_*_best.pth
            └── ...
```

**Checkpoint Contents:**
Each `.pth` file contains:
- `model_state_dict`: Trained model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Learning rate scheduler state (if used)
- `epoch`: Best epoch number
- `window_start`, `window_end`: Time window (sliding window only)
- `mode`, `start_hour`, `upper_hour`: Interval configuration (interval sweep only)
- `fold`: Fold number
- `best_val_score`: Best validation metric score
- `best_metrics`: Full metrics dictionary from evaluation
- `config`: Complete configuration used for training

**Loading a checkpoint:**
```python
checkpoint = torch.load('fold_01_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best {checkpoint.get('window_start')}-{checkpoint.get('window_end')}h window")
print(f"Validation AUC: {checkpoint['best_val_score']:.4f}")
print(f"Test metrics: {checkpoint['best_metrics']}")
```

### Evaluation-Only Scripts:
```
checkpoints/<experiment>/<run_id>/
└── analysis/
    ├── sliding_window_w5_s2_combined.png
    ├── interval_sweep_combined.png
    └── *.json
```

## Summary of Changes

### Files Added:
1. `analyze_sliding_window_train.py` - Full training for sliding windows
2. `analyze_interval_sweep_train.py` - Full training for interval sweep
3. `shells/analyze_sliding_window_train.sh` - Bash helper script
4. `shells/analyze_sliding_window_train.ps1` - PowerShell helper script
5. `shells/analyze_interval_sweep_train.sh` - Bash helper script
6. `shells/analyze_interval_sweep_train.ps1` - PowerShell helper script
7. `load_checkpoint_example.py` - Example script for loading and using saved checkpoints

### Files Modified:
1. `utils/metrics.py` - Fixed ROC AUC warning suppression
2. `train.py` - Added `eval_batch_size_multiplier` for faster validation/testing
3. `configs/resnet50_baseline.yaml` - Added eval_batch_size_multiplier=2 (eval uses 512 batch size)
4. `configs/resnet50_early.yaml` - Added eval_batch_size_multiplier=2
5. `configs/resnet50_time_regression.yaml` - Added eval_batch_size_multiplier=3 (eval uses 384 batch size)
6. `README.md` - Major restructure with training vs evaluation sections + batch size optimization docs
7. `ANALYSIS_UPDATES.md` - This file (comprehensive documentation)
8. `analyze_sliding_window_train.py` - Added checkpoint saving for each window/fold
9. `analyze_interval_sweep_train.py` - Added checkpoint saving for each interval/fold

### Files Unchanged (Legacy):
1. `analyze_sliding_window.py` - Kept for quick evaluation
2. `analyze_interval_sweep.py` - Kept for quick evaluation
3. `shells/analyze_sliding_window.sh` - For legacy script
4. `shells/analyze_sliding_window.ps1` - For legacy script

## Performance Optimizations

### Larger Batch Size for Evaluation
Since validation and testing don't require gradient computation, they can use **2-3x larger batch sizes** than training:

**Configuration:**
```yaml
data:
  batch_size: 256                    # Training batch size
  eval_batch_size_multiplier: 2      # Eval uses 512 (256 * 2)
```

**Benefits:**
- ⚡ **Faster validation/testing**: 2-3x speedup on evaluation
- 💾 **Better GPU utilization**: No gradient tensors means more memory for batches
- 🔧 **Automatic**: Works for all scripts (train.py, analysis scripts, etc.)

**Memory usage:**
- Training: `batch_size * (model + gradients + optimizer states)`
- Evaluation: `eval_batch_size * model` only

**Recommended multipliers:**
- GPUs with 16+ GB VRAM: `eval_batch_size_multiplier: 3`
- GPUs with 8-16 GB VRAM: `eval_batch_size_multiplier: 2` (default)
- GPUs with <8 GB VRAM: `eval_batch_size_multiplier: 1` (same as training)

## Recommendations

1. **For research/publication**: Use training-based analysis to get definitive results
2. **For quick checks**: Use evaluation-only analysis with existing checkpoints
3. **For early detection optimization**: Use sliding window training to find optimal periods
4. **For understanding temporal information**: Use interval sweep training to see how much data is needed
5. **For faster experiments**: Increase `eval_batch_size_multiplier` to speed up validation/testing

All scripts support:
- ✅ K-fold cross-validation
- ✅ Multiple metrics simultaneously
- ✅ Combined and individual plots
- ✅ JSON data export for further analysis
- ✅ Detailed logging
- ✅ **Model checkpoint saving** (NEW)
- ✅ **Optimized evaluation batch sizes** (NEW)
