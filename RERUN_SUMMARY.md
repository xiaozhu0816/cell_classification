# Multitask Training - Updated Configuration

## Changes Made

### 1. Configuration Updates (`configs/multitask_example.yaml`)

**Infection Onset Hour:**
- **Old:** `infection_onset_hour: 2.0`
- **New:** `infection_onset_hour: 1.0`
- **Impact:** Changes how time targets are calculated for infected cells
  - Infected cell at 10h: Previously target = 10-2 = 8h, Now target = 10-1 = 9h

**Training Epochs:**
- **Old:** `epochs: 20`
- **New:** `epochs: 30`
- **Impact:** More training iterations for better convergence

**Scheduler T_max:**
- **Old:** `t_max: 20`
- **New:** `t_max: 30`
- **Impact:** Matches epoch count for learning rate scheduling

### 2. Automatic Visualization Pipeline

After training completes, the following will be **automatically generated**:

#### Standard Training Visualizations (from `analyze_multitask_results.py`)
1. **`training_curves.png`** - Loss curves over epochs
2. **`validation_metrics.png`** - Validation metrics over epochs
3. **`training_summary.txt`** - Text summary report

#### NEW: Prediction Scatter Plots (from `generate_prediction_plot.py`)
4. **`prediction_scatter_regression.png`** - **3-panel scatter plot with regression lines**
   - Left panel: All samples (infected + uninfected) with overall regression
   - Middle panel: Infected cells only with regression line
   - Right panel: Uninfected cells only with regression line

5. **`test_predictions.npz`** - Raw prediction data for further analysis

### 3. Scatter Plot Features

Each scatter plot includes:
- **Perfect prediction line** (diagonal y=x, black dashed)
- **Regression line** (fitted to data, colored)
- **R² score** and **Pearson R** correlation
- **MAE** and **RMSE** metrics
- **Regression equation**: y = mx + b
- **Infection onset marker** (vertical orange line in combined plot)

### 4. Expected Performance Improvements

With `infection_onset_hour: 1.0` instead of `2.0`:
- **Infected cells**: Time targets will be ~1 hour higher
- **Model should learn**: Time since infection with new reference point
- **Expected metrics**: Similar classification performance, slightly different regression values

From previous run (onset=2.0):
- Classification: AUC=0.9999, Accuracy=99.40%
- Regression: MAE=1.15h, RMSE=1.47h

Expected with onset=1.0:
- Classification: Should remain ~0.999+ (onset doesn't affect this)
- Regression: MAE/RMSE might shift slightly due to different time reference

## How to Run

### Option 1: SLURM Cluster
```bash
sbatch run_multitask_training.sh
```

### Option 2: Interactive/GPU Node
```bash
python train_multitask.py --config configs/multitask_example.yaml
```

### Option 3: Check progress while running
```bash
# Monitor training log
tail -f outputs/multitask_resnet50/<run_id>/train.log/multitask_train.log

# Check SLURM output
tail -f slurm_LOG/multitask_<job_id>.out
```

## Output Directory Structure

After completion, you'll have:
```
outputs/multitask_resnet50/<timestamp>/
├── results.json                           # Final test metrics
├── training_curves.png                    # Loss over epochs
├── validation_metrics.png                 # Validation metrics over epochs
├── prediction_scatter_regression.png      # NEW! Scatter with regression
├── training_summary.txt                   # Text report
├── test_predictions.npz                   # Raw predictions
├── checkpoints/
│   └── best.pt                           # Best model checkpoint
└── train.log/
    └── multitask_train.log               # Training log
```

## What to Look For

### 1. Training Convergence
Check `training_curves.png`:
- Total loss should decrease steadily
- No huge spikes (indicates stability)
- Classification loss << regression loss (normal)

### 2. Validation Performance
Check `validation_metrics.png`:
- AUC should reach >0.99 quickly
- MAE should decrease to <2 hours
- No divergence between train/val (overfitting check)

### 3. Prediction Quality
Check `prediction_scatter_regression.png`:
- **Points near diagonal = good predictions**
- **Regression line slope ~1.0 = unbiased**
- **High R² (>0.9) = strong correlation**
- **Low MAE (<2h) = accurate predictions**

Look for:
- Are infected and uninfected predictions equally good?
- Is there systematic bias (regression line != perfect line)?
- Are there outliers? Which time ranges fail?

### 4. Summary Report
Check `training_summary.txt`:
- Overall performance interpretation
- Regression equation for each class
- Suggested improvements if needed

## Comparison with Previous Run

| Metric | Previous (onset=2.0) | New (onset=1.0) |
|--------|---------------------|-----------------|
| Epochs | 20 | 30 ✅ |
| Infection Onset | 2.0h | 1.0h ✅ |
| Classification AUC | 0.9999 | ? (should be similar) |
| Regression MAE | 1.15h | ? (may change) |
| Regression RMSE | 1.47h | ? (may change) |
| Visualizations | 3 plots | 4 plots ✅ (with regression!) |

## Troubleshooting

### If scatter plot not generated:
```bash
# Manually generate it
python generate_prediction_plot.py --result-dir outputs/multitask_resnet50/<run_id>
```

### If you want to regenerate analysis only:
```bash
# Rerun analysis without retraining
python analyze_multitask_results.py --result-dir outputs/multitask_resnet50/<run_id>
```

### If GPU out of memory:
Edit `configs/multitask_example.yaml`:
```yaml
data:
  batch_size: 64  # Reduce from 128
```

## Key Improvements

✅ **Infection onset updated** to 1.0 hour (more biologically accurate?)
✅ **More training epochs** (30 instead of 20) for better convergence
✅ **Automatic scatter plots** with regression lines (no manual steps!)
✅ **Comprehensive metrics** including R², Pearson R, and regression equations
✅ **Class-specific analysis** (separate plots for infected vs uninfected)
✅ **One-command workflow** - just run and wait for all results

## Timeline Estimate

Based on previous run (20 epochs = ~1 hour):
- **Expected duration:** ~1.5 hours for 30 epochs
- **With analysis:** +5 minutes for visualization generation
- **Total:** ~1 hour 35 minutes

Good luck with the training! 🚀
