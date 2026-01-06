# Temporal Generalization Integration - COMPLETE ✓

## Summary

Successfully integrated **temporal generalization analysis** directly into the multitask training script (`train_multitask.py`). This eliminates the need for a separate analysis script and provides automatic temporal performance evaluation after training.

---

## What Was Added

### 1. New Functions in `train_multitask.py`

#### `evaluate_temporal_generalization()`
- **Purpose**: Compute classification metrics across sliding time windows
- **Parameters**:
  - `predictions`: Dictionary with `cls_preds` and `cls_targets` from test evaluation
  - `metadata_list`: List of metadata dictionaries containing time information
  - `window_size`: Size of each window in hours (default: 6.0)
  - `stride`: Step between windows in hours (default: 3.0)
  - `start_hour`/`end_hour`: Time range to analyze (default: 0-48h)
- **Returns**: 
  - `window_centers`: List of center times for each window
  - `metrics_by_window`: Dictionary with AUC, Accuracy, F1, Precision, Recall per window

#### `plot_temporal_generalization()`
- **Purpose**: Create publication-quality temporal generalization plot
- **Parameters**:
  - `window_centers`: Center times from evaluation function
  - `metrics_by_window`: Metrics dictionary from evaluation function
  - `window_size`: Window size for plot title
  - `output_path`: Where to save the PNG file
  - `model_name`: Model name for plot title
- **Output**: Multi-line plot showing all metrics vs. time window

### 2. Integration in Main Training Loop

After final test evaluation, the script now:
1. **Collects metadata** from test dataset (for time information)
2. **Runs temporal analysis** with 6-hour windows, 3-hour stride
3. **Generates plot** → `temporal_generalization.png`
4. **Saves metrics** → `temporal_metrics.json`

All wrapped in try-except to ensure training results are saved even if temporal analysis fails.

---

## Output Files

After running `python train_multitask.py --config configs/multitask_example.yaml`, you will get:

### Core Training Outputs
- `results.json` - Final test metrics and configuration
- `test_predictions.npz` - Raw predictions (classification + regression)
- `checkpoints/best.pt` - Best model checkpoint

### Automatic Visualizations
- `temporal_generalization.png` - **NEW!** Classification metrics across time windows
- `temporal_metrics.json` - **NEW!** Detailed temporal performance data
- Plus plots from `analyze_multitask_results.py`:
  - `training_curves.png` - Loss and metric curves during training
  - `validation_metrics.png` - Classification and regression validation performance
  - `prediction_scatter_regression.png` - Scatter plot with regression line
  - `training_summary.txt` - Text summary of training

---

## Benefits of Integration

### ✅ **Efficiency**
- Reuses test evaluation - no need to reload model/data
- Automatic execution - no separate script to remember

### ✅ **Consistency**
- Uses exact same predictions as final test metrics
- No risk of version mismatch between training and analysis

### ✅ **Convenience**
- One command does everything: train + analyze
- Immediate temporal insights after training completes

### ✅ **Reproducibility**
- Temporal analysis always uses same parameters
- All outputs saved together in one experiment directory

---

## How to Use

### Standard Training (with automatic temporal analysis)
```bash
python train_multitask.py --config configs/multitask_example.yaml
```

**Expected console output:**
```
================================================================================
Final evaluation on test set
test: total_loss:0.xxxx | cls_loss:0.xxxx | reg_loss:0.xxxx | cls_auc:0.xxxx | ...
Test predictions saved to outputs/multitask_resnet50/20250102_120000_abc123/test_predictions.npz

================================================================================
Temporal Generalization Analysis
================================================================================

Temporal Generalization Analysis:
  Window size: 6.0h, Stride: 3.0h
  Number of windows: XX
  Time range: [0.0, 48.0]h
  Window [0.0, 6.0]h: n=XXX (inf=XX, uninf=XX), AUC=0.XXXX, Acc=0.XXXX, F1=0.XXXX
  Window [3.0, 9.0]h: n=XXX (inf=XX, uninf=XX), AUC=0.XXXX, Acc=0.XXXX, F1=0.XXXX
  ...
✓ Temporal generalization plot saved to temporal_generalization.png
✓ Temporal metrics saved to temporal_metrics.json

Results saved to results.json
Training complete!
```

### Customize Temporal Analysis Parameters

If you want different window sizes or ranges, modify `train_multitask.py` (around line 710):

```python
# Current defaults
window_size = 6.0  # 6 hour windows
stride = 3.0       # 3 hour stride
start_hour = 0.0
end_hour = 48.0

# Example: Smaller windows for finer granularity
window_size = 3.0  # 3 hour windows
stride = 1.5       # 1.5 hour stride

# Example: Focus on early infection period
start_hour = 0.0
end_hour = 24.0
```

---

## Comparison with Single-Task Model

To compare multitask vs single-task temporal generalization:

### 1. Run Multitask Training
```bash
python train_multitask.py --config configs/multitask_example.yaml
```
→ Automatically generates `temporal_generalization.png`

### 2. Run Single-Task Temporal Analysis
```bash
python analyze_final_model_sliding_window_fast.py \
  --checkpoint-dir outputs/interval_sweep/train-test_interval_1-46/resnet50_20250101_123456_xyz789/checkpoints \
  --window-size 6.0 --stride 3.0
```
→ Generates `final_model_sliding_w6_combined.png`

### 3. Visual Comparison
- Open both plots side-by-side
- Compare:
  - **Mean performance**: Which model has higher average AUC/F1?
  - **Stability**: Which model has lower variance across windows?
  - **Early detection**: Which performs better in 0-12h window?
  - **Late stage**: Which maintains performance at 36-48h?

### 4. Quantitative Comparison

Load the JSON files for numerical comparison:

```python
import json
import numpy as np

# Load multitask results
with open("outputs/multitask_resnet50/.../temporal_metrics.json") as f:
    multitask = json.load(f)

# Load single-task results  
with open("outputs/interval_sweep/.../sliding_window_metrics.json") as f:
    singletask = json.load(f)

# Compare AUC
mt_auc = [x for x in multitask["metrics_by_window"]["auc"] if x is not None]
st_auc = [x for x in singletask["metrics_by_window"]["auc"] if x is not None]

print(f"Multitask AUC:   mean={np.mean(mt_auc):.4f}, std={np.std(mt_auc):.4f}")
print(f"Single-task AUC: mean={np.mean(st_auc):.4f}, std={np.std(st_auc):.4f}")
```

See `TEMPORAL_GENERALIZATION_COMPARISON.md` for detailed comparison methodology.

---

## Troubleshooting

### Issue: Temporal analysis fails with error

**Symptoms:**
```
Failed to run temporal generalization analysis: ...
Skipping temporal analysis, but training results are still saved.
```

**Causes & Solutions:**

1. **Missing metadata method**
   - Error: `AttributeError: 'Dataset' object has no attribute 'get_metadata'`
   - Solution: Check if your dataset class has `get_metadata(idx)` method
   - Workaround: Comment out temporal analysis section (line 710-748)

2. **No time field in metadata**
   - Error: `KeyError: 'hours_since_start'`
   - Solution: Check metadata format - should have `hours_since_start` field
   - Alternative: Modify `evaluate_temporal_generalization()` to use different field name

3. **Insufficient samples per window**
   - Warning: Many "No samples, skipping" messages
   - Solution: Use larger window size or smaller stride
   - Example: `window_size = 12.0, stride = 6.0`

4. **All windows have single class**
   - Warning: Many "AUC=N/A" messages
   - Explanation: AUC requires both classes in window
   - Solution: Use larger windows or check data distribution

### Issue: Plot looks incorrect

**Check:**
- Time range: Does `[0.0, 48.0]h` match your experiment duration?
- Window size: Is 6h appropriate for your data density?
- Metrics: Are there enough valid AUC points (need both classes)?

**Adjust:**
```python
# Modify in train_multitask.py around line 710
window_size = 3.0   # Smaller windows
stride = 1.5        # More overlap
end_hour = 24.0     # Shorter range
```

---

## Technical Details

### Sliding Window Algorithm

1. **Window generation**: Start at `start_hour`, create windows of size `window_size`, step by `stride`
2. **Sample filtering**: For each window `[start, end]`, find all test samples where `start ≤ time ≤ end`
3. **Metric computation**: Compute AUC, Accuracy, F1, Precision, Recall for samples in window
4. **Handling edge cases**:
   - Skip windows with 0 samples
   - Skip AUC if only one class present
   - Use `zero_division=0` for precision/recall when undefined

### Why These Default Parameters?

- **Window size = 6 hours**: Balances granularity vs. sample size
  - Too small (e.g., 1h): May have insufficient samples, many windows with single class
  - Too large (e.g., 24h): Loses temporal resolution, can't see early/late differences
  
- **Stride = 3 hours**: 50% overlap for smooth curves
  - Overlapping windows show smoother trends
  - Non-overlapping (stride = window_size) gives independent measurements
  
- **Range [0, 48h]**: Typical experiment duration
  - Adjust based on your actual experiment timeline

### Metrics Interpretation

- **AUC (Area Under ROC Curve)**: Overall classification quality, range [0, 1]
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - <0.7: Poor
  
- **Accuracy**: Correct predictions / Total predictions
  - Affected by class imbalance
  
- **F1 Score**: Harmonic mean of precision and recall
  - Balances false positives and false negatives
  - Good for imbalanced datasets
  
- **Precision**: True Positives / (True Positives + False Positives)
  - "When model says infected, how often is it correct?"
  
- **Recall**: True Positives / (True Positives + False Negatives)
  - "Of all infected samples, how many did model catch?"

### Expected Temporal Patterns

**Good temporal generalization:**
- Metrics relatively stable across time windows
- Small standard deviation (<0.05 for AUC)
- No catastrophic drops at any time range

**Poor temporal generalization:**
- Large variation between windows (std >0.1)
- Specific time ranges with much worse performance
- May indicate:
  - Insufficient training data at those times
  - Dataset bias (e.g., all early samples in training)
  - Temporal shift in infection characteristics

**Multitask advantage hypothesis:**
- Regression task provides temporal awareness
- Should show MORE stable performance across time
- Especially important at time ranges far from training distribution

---

## What's Next?

### 1. Run New Training with Updated Config
```bash
python train_multitask.py --config configs/multitask_example.yaml
```

Current config has:
- `infection_onset_hour: 1.0` (changed from 2.0)
- `epochs: 30` (increased from 20)
- Automatic temporal analysis enabled

### 2. Compare with Single-Task

Run temporal analysis on your best single-task model:
```bash
python analyze_final_model_sliding_window_fast.py \
  --checkpoint-dir <path-to-single-task-checkpoint> \
  --window-size 6.0 --stride 3.0
```

### 3. Analyze Results

- Check if multitask model shows better temporal stability
- Document findings in your research notes
- Consider ablation studies:
  - Different `infection_onset_hour` values
  - Different regression weight ratios
  - Different window sizes for analysis

### 4. Publication-Ready Figures

The generated plots are publication-ready, but you can customize:
- Font sizes in `plot_temporal_generalization()` function
- Color schemes
- Figure dimensions
- Add error bars (compute std across cross-validation folds)

---

## Files Modified

- ✅ `train_multitask.py`: Added temporal analysis functions and integration
- ✅ `configs/multitask_example.yaml`: Updated parameters (onset=1.0, epochs=30)

## Related Documentation

- `REGRESSION_EXPLAINED.md`: Explains regression methodology
- `TEMPORAL_GENERALIZATION_COMPARISON.md`: How to compare models
- `MULTITASK_SUMMARY.md`: Overview of multitask approach
- `RERUN_SUMMARY.md`: Configuration update summary

---

## Quick Reference

### Training Command
```bash
python train_multitask.py --config configs/multitask_example.yaml
```

### Output Location
```
outputs/multitask_resnet50/YYYYMMDD_HHMMSS_<run_id>/
├── checkpoints/
│   └── best.pt
├── results.json
├── test_predictions.npz
├── temporal_generalization.png         # NEW!
├── temporal_metrics.json               # NEW!
├── training_curves.png                 # From analyze_multitask_results.py
├── validation_metrics.png              # From analyze_multitask_results.py
├── prediction_scatter_regression.png   # From analyze_multitask_results.py
└── training_summary.txt                # From analyze_multitask_results.py
```

### Comparison Workflow
```bash
# 1. Train multitask (automatic temporal analysis)
python train_multitask.py --config configs/multitask_example.yaml

# 2. Analyze single-task temporal performance
python analyze_final_model_sliding_window_fast.py \
  --checkpoint-dir <single-task-checkpoint-dir> \
  --window-size 6.0 --stride 3.0

# 3. Compare plots side-by-side
# - temporal_generalization.png (multitask)
# - final_model_sliding_w6_combined.png (single-task)
```

---

## Status: ✓ INTEGRATION COMPLETE

The temporal generalization analysis is now fully integrated into the training pipeline. You can proceed with training and will automatically get temporal performance analysis!
