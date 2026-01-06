# Temporal Generalization Analysis: Single-Task vs Multitask

## Overview

Compare classification performance across different time windows between:
- **Single-task model**: Classification only
- **Multitask model**: Classification + Time regression (joint learning)

**Goal**: Determine if learning time information (regression) helps the model generalize better across different timepoints.

## Scripts

### 1. Single-Task Model Analysis
```bash
python analyze_final_model_sliding_window_fast.py \
  --config configs/resnet50_baseline.yaml \
  --checkpoint-dir outputs/interval_sweep_analysis/.../checkpoints/train-test_interval_1-46 \
  --window-size 6.0 \
  --stride 3.0 \
  --start-hour 0.0 \
  --end-hour 48.0 \
  --metrics auc accuracy f1 precision recall
```

### 2. Multitask Model Analysis
```bash
python analyze_multitask_sliding_window.py \
  --config configs/multitask_example.yaml \
  --checkpoint-dir outputs/multitask_resnet50/YYYYMMDD-HHMMSS/checkpoints \
  --window-size 6.0 \
  --stride 3.0 \
  --start-hour 0.0 \
  --end-hour 48.0 \
  --metrics auc accuracy f1 precision recall
```

## What the Analysis Shows

### Temporal Generalization Curve

The sliding window analysis creates a plot showing classification performance (AUC, Accuracy, F1, etc.) vs. time window.

**X-axis**: Time window center (hours)
**Y-axis**: Classification metric value (0-1)

**Interpretation:**
- **Flat line across all windows**: Model generalizes well to all timepoints ✅
- **Performance drop at early times**: Model struggles with early infection stages
- **Performance drop at late times**: Model struggles with late infection stages  
- **U-shaped curve**: Model best at middle timepoints, worse at extremes

### Expected Results

#### Single-Task Model (Classification Only)
- **Trained on**: Cell images with infected/uninfected labels
- **What it learns**: Visual features for infection detection
- **Temporal knowledge**: Implicit (learns what infected cells "look like")
- **Expected pattern**: May show time-dependent performance if infection visual signatures change over time

#### Multitask Model (Classification + Regression)
- **Trained on**: Cell images with labels + time information
- **What it learns**: Visual features + temporal patterns
- **Temporal knowledge**: Explicit (learns infection progression timeline)
- **Expected pattern**: Should be more robust across timepoints due to temporal awareness

### Hypothesis

**Multitask model should show better temporal generalization because:**
1. **Temporal regularization**: Learning time forces model to understand progression
2. **Richer features**: Features must capture both "what" (infected?) and "when" (what time?)
3. **Explicit time modeling**: Model knows early vs. late infection looks different

## How to Compare

### Step 1: Run Both Analyses

Use same window parameters for fair comparison:
- `--window-size 6.0` (6-hour windows)
- `--stride 3.0` (3-hour steps, 50% overlap)
- `--start-hour 0.0` 
- `--end-hour 48.0`
- Same metrics: `auc accuracy f1`

### Step 2: Compare Plots

Look at the temporal generalization curves:

**Example Comparison:**

```
Single-Task Model:
  Window [0-6h]:   AUC=0.85  (struggles with early infection)
  Window [12-18h]: AUC=0.95  (good at mid-stage)
  Window [42-48h]: AUC=0.88  (some drop at late stage)
  Overall std: 0.05 (moderate variation)

Multitask Model:
  Window [0-6h]:   AUC=0.93  (better at early infection!)
  Window [12-18h]: AUC=0.96  (similar to single-task)
  Window [42-48h]: AUC=0.94  (better at late stage!)
  Overall std: 0.02 (less variation - more robust!)
```

### Step 3: Quantitative Comparison

Compare these metrics across all windows:

1. **Mean Performance**: Average metric across all windows
   - Higher = better overall

2. **Std Deviation**: Variation across windows
   - Lower = more consistent, better generalization

3. **Min Performance**: Worst-case window
   - Higher = less vulnerable to specific timepoints

4. **Performance Range**: Max - Min
   - Smaller = more uniform across time

### Step 4: Statistical Significance

If you have multiple folds, compute:
- Mean ± Std for each window
- Error bars show confidence
- Look for non-overlapping error bars (significant difference)

## Interpretation Guide

### Scenario A: Multitask Wins (Expected)

```
Metric: AUC
Single-task: Mean=0.90, Std=0.06, Min=0.82, Range=0.15
Multitask:   Mean=0.94, Std=0.03, Min=0.91, Range=0.07
```

**Interpretation:**
- ✅ Multitask has higher mean (better overall)
- ✅ Multitask has lower std (more consistent)
- ✅ Multitask has higher min (no "bad" windows)
- ✅ Multitask has smaller range (robust across time)

**Conclusion**: Joint learning of time helps generalization!

### Scenario B: Similar Performance

```
Metric: AUC
Single-task: Mean=0.95, Std=0.02, Min=0.93
Multitask:   Mean=0.95, Std=0.02, Min=0.93
```

**Interpretation:**
- Both models generalize equally well
- Possible reasons:
  - Infection visual signatures consistent over time
  - Dataset already well-balanced across timepoints
  - Models reached ceiling performance

**Conclusion**: Multitask doesn't hurt, but doesn't help much either.

### Scenario C: Single-Task Wins (Unexpected)

```
Metric: AUC  
Single-task: Mean=0.95, Std=0.02
Multitask:   Mean=0.92, Std=0.04
```

**Interpretation:**
- Multitask performs worse (surprising!)
- Possible reasons:
  - Regression task is too hard, interfering with classification
  - Loss weighting suboptimal (classification weight too low)
  - Model capacity insufficient for both tasks

**Conclusion**: Need to tune multitask configuration (loss weights, model size).

## Example Output Files

### Single-Task
```
outputs/interval_sweep_analysis/.../sliding_window_fast_TIMESTAMP/
├── final_model_sliding_w6_combined.png    # All metrics
├── final_model_sliding_w6_auc.png         # AUC only
├── final_model_sliding_w6_data.json       # Raw data
└── final_model_sliding_window_fast.log    # Detailed log
```

### Multitask
```
outputs/multitask_resnet50/.../sliding_window_analysis_TIMESTAMP/
├── multitask_temporal_generalization_w6h.png   # All metrics
├── multitask_sliding_window_results.json       # Raw data
└── multitask_sliding_window.log                # Detailed log
```

## Visualization Tips

### Create Side-by-Side Comparison

You can manually overlay both plots or create a comparison:

```python
import json
import matplotlib.pyplot as plt

# Load both results
with open("single_task_data.json") as f:
    single_data = json.load(f)

with open("multitask_data.json") as f:
    multi_data = json.load(f)

# Plot comparison
fig, ax = plt.subplots(figsize=(14, 7))

# Single-task line
ax.errorbar(
    single_data["results"]["auc"]["window_centers"],
    single_data["results"]["auc"]["means"],
    yerr=single_data["results"]["auc"]["stds"],
    fmt='-o',
    label='Single-Task (Classification Only)',
    linewidth=2,
    markersize=8,
)

# Multitask line
ax.errorbar(
    multi_data["results"]["auc"]["window_centers"],
    multi_data["results"]["auc"]["means"],
    yerr=multi_data["results"]["auc"]["stds"],
    fmt='-s',
    label='Multitask (Classification + Regression)',
    linewidth=2,
    markersize=8,
)

ax.set_xlabel("Time Window Center (hours)", fontsize=13)
ax.set_ylabel("AUC", fontsize=13)
ax.set_title("Temporal Generalization: Single-Task vs Multitask", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig("comparison.png", dpi=200)
```

## Key Research Questions

1. **Does multitask learning improve temporal generalization?**
   - Compare std deviation of performance across windows

2. **Are there specific time ranges where multitask helps most?**
   - Look for windows where multitask >> single-task
   - Often: early infection (time info helps recognize subtle changes)

3. **Is the improvement consistent across metrics?**
   - Check if multitask better for AUC, Accuracy, F1, etc.
   - Or only for specific metrics?

4. **What's the trade-off?**
   - Multitask: Better generalization but more complex training
   - Single-task: Simpler but potentially less robust

## Next Steps After Analysis

### If Multitask Wins
- ✅ Use multitask model for deployment
- ✅ Emphasize temporal generalization in paper/presentation
- ✅ Analyze which time features model learned (via attention/CAM)

### If Results Similar
- Consider computational cost: single-task is simpler
- Multitask still valuable for time prediction capability
- May depend on downstream application needs

### If Single-Task Wins
- Tune multitask hyperparameters:
  - Increase `classification_weight` (e.g., 2.0)
  - Increase `hidden_dim` (e.g., 512)
  - Try different `infection_onset_hour` values
- Verify time labels are correct
- Check if regression task is too noisy

## Publication-Ready Analysis

For your paper/presentation, include:

1. **Temporal generalization plots** (both models side-by-side)
2. **Statistical comparison table**:
   ```
   | Model | Mean AUC | Std AUC | Min AUC | Max AUC | Range |
   |-------|----------|---------|---------|---------|-------|
   | Single-task | 0.90 | 0.06 | 0.82 | 0.97 | 0.15 |
   | Multitask   | 0.94 | 0.03 | 0.91 | 0.97 | 0.06 |
   ```
3. **Key finding statement**: 
   "Multitask learning improved temporal generalization, reducing performance variation across time windows by 50% (std: 0.03 vs 0.06)"

4. **Visual comparison plot** showing both curves with confidence intervals

## Running the Full Comparison

```bash
# Step 1: Analyze single-task model
python analyze_final_model_sliding_window_fast.py \
  --config configs/resnet50_baseline.yaml \
  --checkpoint-dir outputs/interval_sweep_analysis/.../train-test_interval_1-46 \
  --window-size 6.0 --stride 3.0 --end-hour 48.0 \
  --output-dir comparison/single_task

# Step 2: Analyze multitask model
python analyze_multitask_sliding_window.py \
  --config configs/multitask_example.yaml \
  --checkpoint-dir outputs/multitask_resnet50/.../checkpoints \
  --window-size 6.0 --stride 3.0 --end-hour 48.0 \
  --output-dir comparison/multitask

# Step 3: Compare results
ls comparison/single_task/*.png
ls comparison/multitask/*.png
# Open both plots side-by-side for visual comparison
```

Good luck with the comparison! This should give you strong evidence for whether multitask learning helps temporal generalization in your cell classification task! 🔬📊
