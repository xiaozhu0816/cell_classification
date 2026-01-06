# How to Get the Same Analysis Graphs for 5-Fold CV

## Problem
You have comprehensive analysis graphs in `20260102-163144/` that you want for the 5-fold CV results `20260105-155852_5fold/`.

The CV training didn't save `test_predictions.npz` files, which are needed for the comprehensive analysis.

## Solution

I've created scripts to:
1. **Regenerate predictions** from existing checkpoints
2. **Run all analyses** to generate the same graphs as `20260102-163144/`

---

## Quick Start (One Command)

```bash
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

Or if you used a different config file:

```bash
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold --config configs/your_config.yaml
```

This will automatically:
1. Check if predictions exist
2. Regenerate them if needed (using saved checkpoints)
3. Run all comprehensive analyses
4. Generate all plots

---

## What You'll Get

After running, you'll have these files in `outputs/multitask_resnet50/20260105-155852_5fold/`:

### 📊 Analysis Plots (Same as 20260102-163144)

1. **`prediction_scatter.png`**
   - Regression predictions vs ground truth
   - Red = infected, Blue = uninfected
   - Shows MAE, RMSE, R² statistics
   - Perfect prediction diagonal line

2. **`error_analysis_by_time.png`** (4 subplots)
   - Error histogram with mean/median
   - Error by cell type (infected vs uninfected)
   - Error vs time scatter with trend
   - Error percentiles by time bins (highlights valley 13-19h)

3. **`error_vs_classification_confidence.png`** (2 subplots)
   - Scatter: classification confidence vs regression error
   - Boxplot: error distribution by confidence bins
   - Shows correlation between tasks

4. **`valley_period_analysis.png`** (4 subplots)
   - Mean error by time range and cell type
   - Valley vs non-valley boxplots
   - Statistical tests (t-test, Mann-Whitney U)
   - Error distribution histograms for uninfected cells

5. **`worst_predictions_report.txt`**
   - Top 20 worst regression errors
   - All classification misclassifications
   - Summary statistics

---

## Manual Steps (If Needed)

If you want to run steps individually:

### Step 1: Regenerate Predictions

```bash
python regenerate_cv_predictions.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

This will:
- Load each fold's best checkpoint
- Run inference on test set
- Save `test_predictions.npz` in each `fold_*/` directory

### Step 2: Run Comprehensive Analysis

```bash
python analyze_cv_results_comprehensive.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

This will:
- Load predictions from all 5 folds
- Aggregate them together
- Generate all 5 analysis plots
- Create worst predictions report

---

## What Each Script Does

### `regenerate_cv_predictions.py`
**Purpose:** Create test_predictions.npz files from existing checkpoints

**How it works:**
1. Loads CV summary to get config
2. For each fold (1-5):
   - Loads best.pt checkpoint
   - Creates test dataloader
   - Runs inference
   - Saves predictions to `fold_X/test_predictions.npz`

**Files created:**
- `fold_1/test_predictions.npz`
- `fold_2/test_predictions.npz`
- `fold_3/test_predictions.npz`
- `fold_4/test_predictions.npz`
- `fold_5/test_predictions.npz`

### `analyze_cv_results_comprehensive.py`
**Purpose:** Generate all analysis plots (same as 20260102-163144)

**How it works:**
1. Loads all fold predictions
2. Aggregates them (combines all test data from 5 folds)
3. Runs comprehensive analyses:
   - Prediction scatter
   - Error analysis by time
   - Error vs confidence
   - Valley period analysis
   - Worst predictions report

**Files created:**
- `prediction_scatter.png`
- `error_analysis_by_time.png`
- `error_vs_classification_confidence.png`
- `valley_period_analysis.png`
- `worst_predictions_report.txt`

### `run_full_cv_analysis.py`
**Purpose:** One-command wrapper that runs both steps

**How it works:**
1. Checks if predictions exist
2. Runs `regenerate_cv_predictions.py` if needed
3. Runs `analyze_cv_results_comprehensive.py`
4. Prints summary of what was generated

---

## Expected Runtime

- **Regenerating predictions:** ~2-5 minutes per fold (depends on test set size and GPU)
- **Running analysis:** ~10-30 seconds (just plotting)
- **Total:** ~10-25 minutes for full pipeline

---

## Comparison: Single-Run vs 5-Fold CV

### Single-Run (`20260102-163144/`)
- **test_predictions.npz:** ~2,000 samples (one test fold)
- **Graphs:** Based on single model's predictions

### 5-Fold CV (`20260105-155852_5fold/`)
- **test_predictions.npz:** ~2,000 samples **per fold** = ~10,000 total
- **Graphs:** Based on **aggregated predictions from all 5 folds**
- **More robust:** Every sample predicted exactly once by a model that didn't train on it

**The 5-fold CV graphs will be MORE RELIABLE** because they represent the model's performance across different data splits!

---

## Troubleshooting

### Error: "checkpoint not found"
**Cause:** Missing `fold_X/checkpoints/best.pt`  
**Solution:** Make sure CV training completed successfully for all folds

### Error: "No predictions found"
**Cause:** `regenerate_cv_predictions.py` failed or was skipped  
**Solution:** Run it manually first, check for errors

### Error: "CUDA out of memory"
**Cause:** GPU memory full during prediction regeneration  
**Solution:** Reduce `eval_batch_size_multiplier` in config or use CPU

### Blank/Empty plots
**Cause:** No data loaded from predictions  
**Solution:** Check that `fold_*/test_predictions.npz` files exist and have data

---

## Quick Reference

```bash
# Full pipeline (recommended)
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold

# Just regenerate predictions
python regenerate_cv_predictions.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold

# Just run analysis (if predictions exist)
python analyze_cv_results_comprehensive.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

---

## Summary

**What you need:** The same comprehensive analysis graphs from `20260102-163144/` for your 5-fold CV results

**What was missing:** Test predictions weren't saved during CV training

**Solution:** 
1. Created `regenerate_cv_predictions.py` to generate them from checkpoints
2. Created `analyze_cv_results_comprehensive.py` to run all analyses
3. Created `run_full_cv_analysis.py` as one-command wrapper

**Run this:**
```bash
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

**You'll get:** All 5 analysis plots + worst predictions report, exactly matching `20260102-163144/` but aggregated across 5 folds for more robust results!
