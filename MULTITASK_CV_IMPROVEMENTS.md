# Answers to Your Two Questions

## 1. ✅ Fold Splits ARE the Same

**Answer**: Yes, the fold splits are already the same across all experiments!

### How It Works

All scripts use the **same splitting mechanism** from `datasets/timecourse_dataset.py`:

```python
# Line 305 in timecourse_dataset.py
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
```

This means:
- ✅ Uses `split_seed` from your config file (e.g., `split_seed: 123` in `resnet50_baseline.yaml`)
- ✅ Stratified by class (infected/uninfected balanced in each fold)
- ✅ **Same random_state = same splits** across all experiments

### Verification

As long as you use the **same config file** (or same `split_seed` value) for both:
- `train.py` (baseline single-task)
- `train_multitask_cv.py` (multitask CV)

The fold splits will be **identical**. ✅

**Example**:
```yaml
# In resnet50_baseline.yaml and multitask_example.yaml
data:
  split_seed: 123  # ← This ensures same folds
```

If `split_seed` is different, change it to match!

---

## 2. ✅ Temporal Generalization with CV (Fixed!)

**Answer**: Yes, I've made TWO improvements:

### Improvement 1: Auto-Scale Y-Axis (Fixed for All Scripts)

**Before**:
```python
ax.set_ylim(0.0, 1.05)  # Always starts from 0
```

**After**:
```python
# Auto-scale from lowest point (with 10% padding)
y_min = min(all_values)
y_max = max(all_values)
y_range = y_max - y_min
padding = y_range * 0.1
ax.set_ylim(max(0, y_min - padding), min(1.05, y_max + padding))
```

**Result**: Y-axis now starts from the lowest data point (not zero), making differences more visible! 📊

**Applied to**:
- ✅ `generate_temporal_analysis.py` (single run)
- ✅ `analyze_multitask_cv_temporal.py` (CV aggregated)

### Improvement 2: New CV Temporal Analysis Script

Created `analyze_multitask_cv_temporal.py` that:
- ✅ Loads all 5 fold checkpoints
- ✅ Evaluates each fold's temporal generalization
- ✅ **Aggregates** with mean ± std across folds
- ✅ Creates plot with shaded ± std regions

**Usage**:
```bash
# After running train_multitask_cv.py, run:
python analyze_multitask_cv_temporal.py --result-dir outputs/multitask_cv/TIMESTAMP_5fold
```

**Output**:
```
outputs/multitask_cv/TIMESTAMP_5fold/
├── cv_temporal_generalization.png  # 👈 Main plot with mean ± std
├── cv_temporal_metrics.json        # Numerical results
├── fold_1/temporal_metrics.json    # Individual fold results
├── fold_2/temporal_metrics.json
├── fold_3/temporal_metrics.json
├── fold_4/temporal_metrics.json
└── fold_5/temporal_metrics.json
```

**Example Plot Features**:
- Line shows **mean** across 5 folds
- Shaded region shows **± 1 std**
- Y-axis **auto-scaled** from minimum value (not zero!)
- Each metric (AUC, F1, Accuracy, etc.) has its own color

---

## Summary

| Question | Answer | Status |
|----------|--------|--------|
| 1. Are fold splits the same? | ✅ Yes, uses same `split_seed` | Already done |
| 2. Does temporal plot use 5 folds? | ✅ Yes, new script aggregates all folds | **New script created** |
| 2. Is y-axis auto-scaled? | ✅ Yes, starts from lowest point | **Fixed in both scripts** |

---

## Complete Workflow

### For Single Run (Existing Results)
```bash
# Generate temporal plot for existing results
python generate_temporal_analysis.py --result-dir outputs/multitask_resnet50/20260102-163144
```
- ✅ Y-axis auto-scaled
- Output: `temporal_generalization.png`

### For 5-Fold CV
```bash
# Step 1: Run 5-fold training
python train_multitask_cv.py --config configs/multitask_example.yaml --num-folds 5

# Step 2: Generate aggregated temporal analysis
python analyze_multitask_cv_temporal.py --result-dir outputs/multitask_cv/TIMESTAMP_5fold
```
- ✅ Uses all 5 folds
- ✅ Y-axis auto-scaled
- ✅ Shows mean ± std
- Output: `cv_temporal_generalization.png`

---

## Visual Comparison

### Before (Y-axis from 0):
```
1.0 |------------------------
    |     ___---___
0.8 |  __/         \__
    | /               \
0.6 |/                 \
    |
0.4 |
    |
0.2 |
    |
0.0 |_____________________
```
❌ Lots of wasted space, hard to see differences

### After (Y-axis auto-scaled):
```
1.0 |------------------------
    |     ___---___
0.95|  __/         \__
    | /               \
0.90|/                 \
    |
0.85|
    |_____________________
```
✅ Zoomed in on data range, differences clear!

---

## Verification

Both issues are now fixed! You can verify by:

1. **Check fold consistency**: Compare fold splits between `train.py` and `train_multitask_cv.py`
   - Both use same `build_datasets()` with same `split_seed`

2. **Check y-axis scaling**: Run temporal analysis and inspect plots
   - Y-axis should start from ~0.85 or similar (not 0.0)
   - Shaded regions show ± std in CV plot
