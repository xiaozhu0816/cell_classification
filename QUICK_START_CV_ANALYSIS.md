# Quick Guide: Generate Analysis Plots for 5-Fold CV

## Problem
You have CV results in `outputs/multitask_resnet50/20260105-155852_5fold/` but the comprehensive analysis plots (like those in `20260102-163144/`) are missing or blank.

## Solution - ONE COMMAND

```bash
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

**That's it!** This will:
1. Load each fold's checkpoint
2. Generate predictions on test data
3. Create all analysis plots
4. Save them in the CV results directory

---

## What You'll Get

After running (~10-15 minutes), you'll have:

✅ **`prediction_scatter.png`**
- Regression predictions vs ground truth
- Red = infected, Blue = uninfected  
- Shows MAE, RMSE, R² stats

✅ **`error_analysis_by_time.png`**
- Error histogram
- Error by cell type
- Error vs time with trend
- Error percentiles by time bins (valley highlighted)

✅ **`error_vs_classification_confidence.png`**
- Confidence vs error scatter
- Error distribution by confidence bins
- Shows task coupling

✅ **`valley_period_analysis.png`**
- Mean error by time range
- Valley vs non-valley comparison
- Statistical tests (t-test, Mann-Whitney)
- Error distribution histograms

✅ **`worst_predictions_report.txt`**
- Top 20 worst regression errors
- All classification misclassifications
- Summary statistics

---

## If You Have Issues

### Different Config File?
If you trained with a different config:

```bash
python run_full_cv_analysis.py \
    --result-dir outputs/multitask_resnet50/20260105-155852_5fold \
    --config configs/your_config_file.yaml
```

### Want to Run Steps Separately?

**Step 1: Generate Predictions**
```bash
python regenerate_cv_predictions.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

**Step 2: Run Analysis**
```bash
python analyze_cv_results_comprehensive.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

---

## Why This is Better Than Single-Run Analysis

- **More data**: Aggregates ~10,000 predictions (vs ~2,000 for single run)
- **More robust**: Every sample predicted by model that didn't train on it
- **True generalization**: Shows how model performs on completely unseen data

The plots will show **the same analyses as `20260102-163144/`** but with more reliable statistics!

---

## Troubleshooting

**"CUDA out of memory"**
→ The script uses GPU if available. Should work fine, but if issues arise, it will fallback to CPU

**"Checkpoint not found"**
→ Make sure all 5 folds completed training successfully

**Takes too long?**
→ Expected: 2-3 min per fold = ~10-15 min total (mostly data loading)

---

## Summary

**Run this:**
```bash
python run_full_cv_analysis.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
```

**Get:** All 5 comprehensive analysis plots + worst predictions report

**Same as:** The analysis in `20260102-163144/`, but aggregated across all CV folds!
