# Interval Sweep Two-Mode Comparison

## What This Experiment Does

The `analyze_interval_sweep_train.py` script automatically runs **TWO experiments in parallel** to compare training strategies.

⚠️ **IMPORTANT:** Both modes **TRAIN fresh models**! The mode names refer to which splits use the restricted interval [1, x].

### Mode 1: "test-only" 
**More accurate name:** "Train on ALL, test on restricted"

```
TRAINING (same for all intervals):
  Infected: [1, FULL] (all available frames, e.g., 1-30h)
  Uninfected: [0, FULL] (all frames)
  ↓ TRAINS A NEW MODEL ↓
  
TESTING (varies by interval):
  Infected: [1, 8]  → [1, 10] → [1, 12] → ... → [1, 30]
  Uninfected: [0, FULL] (all frames)
```

**Question answered:** "If I train on all available data, how well can I detect infection using only early time windows for testing?"

**Why it's called "test-only":** Only the TEST split uses the restricted interval [1, x]. Training always uses full data.

---

### Mode 2: "train-test"
**More accurate name:** "Train and test on same restricted interval"

```
TRAINING (varies by interval):
  Infected: [1, 8]  → [1, 10] → [1, 12] → ... → [1, 30]
  Uninfected: [0, FULL] (all frames)
  ↓ TRAINS A NEW MODEL ↓
  
TESTING (matches training):
  Infected: [1, 8]  → [1, 10] → [1, 12] → ... → [1, 30]
  Uninfected: [0, FULL] (all frames)
```

**Question answered:** "If I know I'll only have data up to hour X at deployment, should I train on that same restricted window?"

**Why it's called "train-test":** Both TRAIN and TEST splits use the same restricted interval [1, x].

---

## Visual Comparison

For each upper bound X (8, 10, 12, 14, ...):

```
Mode 1 (test-only):          Mode 2 (train-test):
┌─────────────────────┐      ┌─────────────────────┐
│ TRAINING            │      │ TRAINING            │
│ Infected: [1, ALL]  │      │ Infected: [1, X]    │
│ Uninfected: ALL     │      │ Uninfected: ALL     │
└─────────────────────┘      └─────────────────────┘
          ↓                            ↓
┌─────────────────────┐      ┌─────────────────────┐
│ TESTING             │      │ TESTING             │
│ Infected: [1, X]    │      │ Infected: [1, X]    │
│ Uninfected: ALL     │      │ Uninfected: ALL     │
└─────────────────────┘      └─────────────────────┘
```

---

## Example Intervals Tested

If you set `--upper-hours 8 10 12 14 16 18 20`:

| Interval | Mode 1 (test-only)              | Mode 2 (train-test)           |
|----------|----------------------------------|-------------------------------|
| [1, 8]   | Train: [1, ALL] → Test: [1, 8]  | Train: [1, 8] → Test: [1, 8]  |
| [1, 10]  | Train: [1, ALL] → Test: [1, 10] | Train: [1, 10] → Test: [1, 10]|
| [1, 12]  | Train: [1, ALL] → Test: [1, 12] | Train: [1, 12] → Test: [1, 12]|
| [1, 14]  | Train: [1, ALL] → Test: [1, 14] | Train: [1, 14] → Test: [1, 14]|
| [1, 16]  | Train: [1, ALL] → Test: [1, 16] | Train: [1, 16] → Test: [1, 16]|
| [1, 18]  | Train: [1, ALL] → Test: [1, 18] | Train: [1, 18] → Test: [1, 18]|
| [1, 20]  | Train: [1, ALL] → Test: [1, 20] | Train: [1, 20] → Test: [1, 20]|

---

## Output Plot Structure

The script generates **two-panel plots**:

```
┌─────────────────────────────────────────────────────────┐
│                 Interval Sweep Comparison               │
├─────────────────────────┬───────────────────────────────┤
│  LEFT: test-only        │  RIGHT: train-test            │
│  (Train ALL, Test X)    │  (Train X, Test X)            │
│                         │                               │
│    AUC                  │    AUC                        │
│    ↑                    │    ↑                          │
│ 1.0│     ●───●───●      │ 1.0│  ●───●───●               │
│    │   ●               │    │●                         │
│ 0.8│ ●                  │ 0.8│                          │
│    │                    │    │                          │
│ 0.6│                    │ 0.6│                          │
│    └─────────────→      │    └─────────────→            │
│      8  12  16  20      │      8  12  16  20            │
│    Upper Hour (X)       │    Upper Hour (X)             │
└─────────────────────────┴───────────────────────────────┘
```

---

## Interpretation Guide

### Scenario 1: Left > Right
```
test-only (left):  ●───●───●───●  (higher AUC)
train-test (right):  ●───●───●    (lower AUC)
```
**Meaning:** Training on MORE data (all frames) helps, even when testing on restricted windows.
**Action:** Use all available data for training, even if deployment only sees early frames.

---

### Scenario 2: Left ≈ Right
```
test-only (left):  ●───●───●───●
train-test (right): ●───●───●───●  (similar AUC)
```
**Meaning:** Restricting training data to match test window doesn't hurt performance.
**Action:** Can safely train on restricted windows if needed (e.g., for faster training).

---

### Scenario 3: Right > Left (unusual)
```
test-only (left):  ●───●───●      (lower AUC)
train-test (right):  ●───●───●───● (higher AUC)
```
**Meaning:** Training on restricted data actually helps (possible overfitting to later frames in test-only mode).
**Action:** Match training and test windows for better generalization.

---

## How to Run

### Bash (Linux/Mac):
```bash
bash shells/interval_sweep_comparison.sh
```

### PowerShell (Windows):
```powershell
.\shells\interval_sweep_comparison.ps1
```

### Direct Python:
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 22 24 26 28 30 \
    --start-hour 1 \
    --metrics auc accuracy f1 \
    --k-folds 5 \
    --epochs 10 \
    --split test
```

---

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--upper-hours` | Upper bounds to test (e.g., 8 12 16 20) | Required |
| `--start-hour` | Start of infected interval | 1 |
| `--metrics` | Metrics to evaluate (auc accuracy f1 ...) | auc |
| `--k-folds` | Number of cross-validation folds | From config |
| `--epochs` | Training epochs per interval | From config |
| `--split` | Evaluation split (test/val) | test |

---

## Total Training Runs

For `--upper-hours 8 10 12 14 16 18 20` with `--k-folds 5`:

```
Total = (# intervals) × (# modes) × (# folds)
      = 7 × 2 × 5
      = 70 training runs
```

Each run trains a fresh model from scratch!

---

## Output Files

```
outputs/interval_sweep_analysis/<timestamp>/
├── interval_sweep_combined.png      # All metrics, two panels
├── interval_sweep_auc.png           # AUC two-panel comparison
├── interval_sweep_accuracy.png      # Accuracy two-panel comparison
├── interval_sweep_f1.png            # F1 two-panel comparison
├── interval_sweep_data.json         # Raw fold metrics + stats
├── interval_sweep_train.log         # Training log
└── checkpoints/
    ├── test-only_interval_1-8/
    │   ├── fold_01_best.pth
    │   ├── fold_02_best.pth
    │   └── ...
    ├── test-only_interval_1-10/
    ├── train-test_interval_1-8/
    ├── train-test_interval_1-10/
    └── ...
```

---

## Summary

✅ **Your Experiment 1** = "test-only" mode (already implemented)  
✅ **Your Experiment 2** = "train-test" mode (already implemented)  
✅ **Both run automatically** in a single script call  
✅ **Side-by-side comparison** in two-panel plots  
✅ **All checkpoints saved** for later analysis  

Just run the shell script and you'll get both experiments! 🚀
