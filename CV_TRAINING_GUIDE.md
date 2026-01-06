# 5-Fold Cross-Validation Training Guide

## Quick Start

```bash
# Run 5-fold CV with same settings as other experiments
python train_multitask_cv.py --config configs/multitask_example.yaml --num-folds 5
```

## What It Does

1. **Trains 5 separate models** - one for each fold
2. **Same settings** as your standard training (same config file)
3. **Saves each fold** separately: `outputs/multitask_cv/TIMESTAMP_5fold/fold_1/`, `fold_2/`, etc.
4. **Generates aggregated statistics**: Mean ± Std for all metrics
5. **Enables statistical comparison** with other methods

## Output Structure

```
outputs/multitask_cv/20260105-123456_5fold/
├── train_cv.log                          # Full training log
├── cv_summary.json                        # Aggregated results (mean ± std)
├── fold_1/
│   ├── checkpoints/best.pt               # Best model for fold 1
│   └── results.json                      # Test metrics for fold 1
├── fold_2/
│   ├── checkpoints/best.pt
│   └── results.json
├── fold_3/
│   ├── checkpoints/best.pt
│   └── results.json
├── fold_4/
│   ├── checkpoints/best.pt
│   └── results.json
└── fold_5/
    ├── checkpoints/best.pt
    └── results.json
```

## Expected Results Format

**cv_summary.json**:
```json
{
  "num_folds": 5,
  "aggregated_metrics": {
    "cls_auc": {
      "mean": 0.9856,
      "std": 0.0123,
      "min": 0.9654,
      "max": 0.9987,
      "values": [0.9856, 0.9765, 0.9987, 0.9654, 0.9918]
    },
    "cls_f1": {
      "mean": 0.9234,
      "std": 0.0187,
      ...
    },
    "reg_mae": {
      "mean": 2.456,
      "std": 0.342,
      ...
    },
    "combined": {
      "mean": 0.8567,
      "std": 0.0145,
      ...
    }
  }
}
```

## Training Time

- **Per fold**: ~1 hour (10 epochs, similar to standard training)
- **Total 5-fold**: ~5 hours
- Can run in background or on cluster

## Model Selection

Each fold uses the **combined metric** (0.6*F1 + 0.4*(1-MAE/48)):
- Same metric as your updated standard training
- Fair comparison with single-split results

## Key Features

✅ **Stratified splits** - Each fold has balanced infected/uninfected
✅ **Same settings** - Uses your existing config file
✅ **Robust estimates** - Mean ± Std across 5 folds
✅ **Statistical testing** - Can compare with other methods
✅ **Same metric** - Combined metric for model selection

## Comparison with Standard Training

| Aspect | Standard (`train_multitask.py`) | CV (`train_multitask_cv.py`) |
|--------|--------------------------------|------------------------------|
| Training time | ~1 hour | ~5 hours (5 folds × 1 hour) |
| Results | Single AUC/F1/MAE | Mean ± Std |
| Model saved | 1 best model | 5 best models (1 per fold) |
| Statistical testing | ❌ No | ✅ Yes |
| Hyperparameter tuning | ✅ Good for this | ❌ Too slow |
| Final evaluation | ⚠️ Single estimate | ✅ Robust estimate |

## When to Use CV

- ✅ **Final model evaluation** - Report mean ± std in paper
- ✅ **Comparing methods** - Statistical testing
- ✅ **Small datasets** - More reliable estimates
- ❌ **Hyperparameter tuning** - Too slow, use single split

## Analyzing CV Results

You can create a script to compare CV results with single-task models:

```python
import json
import numpy as np

# Load CV results
with open("outputs/multitask_cv/.../cv_summary.json") as f:
    cv_results = json.load(f)

# Load single-task results (if you have them)
with open("outputs/single_task/.../results.json") as f:
    single_results = json.load(f)

# Compare
mt_auc = cv_results["aggregated_metrics"]["cls_auc"]
print(f"Multitask AUC: {mt_auc['mean']:.4f} ± {mt_auc['std']:.4f}")

# Single task would be just one number:
# print(f"Single Task AUC: {single_results['test_auc']:.4f}")
```

## Log Output Example

```
================================================================================
FOLD 1/5
================================================================================

Building datasets for fold 1...
Fold 1 - Train: 4800, Val: 1200, Test: 1200
Training fold 1 for 10 epochs...
Primary metric: combined (0.6*F1 + 0.4*(1-MAE/48))

Fold 1, Epoch 1/10
F1_E1_train: 100%|████████████| 150/150 [01:23<00:00,  1.80it/s]
F1_E1_val: 100%|██████████████| 38/38 [00:15<00:00,  2.53it/s]
✓ Fold 1: New best model! combined=0.7234

...

Fold 1: Final evaluation on test set
F1_test: 100%|████████████████| 38/38 [00:15<00:00,  2.50it/s]
Fold 1 test results: total_loss:0.3456 | cls_auc:0.9856 | cls_f1:0.9234 | reg_mae:2.456 | combined:0.8567
✓ Fold 1 complete!

================================================================================
CROSS-VALIDATION SUMMARY
================================================================================

Mean ± Std across folds:
  cls_auc: 0.9856 ± 0.0123
  cls_f1: 0.9234 ± 0.0187
  reg_mae: 2.456 ± 0.342
  combined: 0.8567 ± 0.0145

✓ CV summary saved to outputs/multitask_cv/.../cv_summary.json
```

## Next Steps After CV

1. **Generate temporal analysis**: Run sliding window analysis across all folds
   ```bash
   python analyze_multitask_cv_temporal.py --result-dir outputs/multitask_cv/TIMESTAMP_5fold
   ```
   This will create:
   - `cv_temporal_generalization.png` - Plot with mean ± std across folds
   - `cv_temporal_metrics.json` - Numerical results
   - Individual `fold_X/temporal_metrics.json` for each fold

2. **Report results**: Use mean ± std in your paper
3. **Statistical testing**: Compare with single-task using paired t-test
4. **Select best fold**: If needed, choose fold with highest combined score for deployment
5. **Ensemble**: Optionally combine all 5 models for predictions

## Notes

- Uses the **same config file** as standard training
- Each fold's test set is completely independent
- Results are automatically aggregated to `cv_summary.json`
- All individual fold results also saved for debugging
