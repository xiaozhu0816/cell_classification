# Multi-Task Training Guide

## Overview
The multi-task model jointly learns:
1. **Classification**: Infected vs uninfected cells
2. **Time Regression**: Temporal information (time since infection for infected cells, elapsed time for uninfected)

## Quick Start

### 1. Train the model
```bash
# On local machine or interactive node
python train_multitask.py --config configs/multitask_example.yaml

# On SLURM cluster
sbatch shells/train_multitask.sh
```

### 2. View results
After training, automatically generated files:
- `outputs/multitask_resnet50/<run_id>/training_curves.png` - Loss curves
- `outputs/multitask_resnet50/<run_id>/validation_metrics.png` - All metrics over epochs
- `outputs/multitask_resnet50/<run_id>/training_summary.txt` - Text summary
- `outputs/multitask_resnet50/<run_id>/results.json` - Full results JSON
- `outputs/multitask_resnet50/<run_id>/checkpoints/best.pt` - Best model checkpoint

### 3. Manual analysis (if auto-analysis fails)
```bash
python analyze_multitask_results.py \
  --result-dir outputs/multitask_resnet50/20251215-164539
```

## Configuration

Edit `configs/multitask_example.yaml` to customize:

### Key Parameters

**Multi-task settings:**
```yaml
multitask:
  infection_onset_hour: 2.0          # When infection occurs
  classification_weight: 1.0         # Weight for classification loss
  regression_weight: 1.0             # Weight for regression loss
  clamp_range: [0.0, 48.0]          # Clamp time predictions
```

**Loss weight tuning:**
- Start with equal weights (1.0, 1.0)
- If classification is weak: increase `classification_weight` to 2.0-5.0
- If time prediction is weak: increase `regression_weight` to 2.0-5.0
- Monitor validation metrics to find the right balance

**Model architecture:**
```yaml
model:
  name: resnet50                     # resnet18, resnet34, resnet50, etc.
  hidden_dim: 256                    # Hidden layer size (0 for no hidden layer)
  dropout: 0.2                       # Dropout rate
```

**Data filtering:**
```yaml
data:
  frames:
    infected_window_hours: [0, 48]   # Time range for infected samples
    uninfected_window_hours: [0, 48] # Time range for uninfected samples
```

## Understanding Results

### Classification Metrics
- **AUC**: Area under ROC curve (0.95+ is excellent)
- **Accuracy**: Overall correctness
- **Precision**: Of predicted infected, how many are truly infected?
- **Recall**: Of truly infected, how many did we detect?
- **F1**: Harmonic mean of precision and recall

### Regression Metrics
- **MAE** (Mean Absolute Error): Average time prediction error in hours
  - < 1 hour: Excellent
  - 1-2 hours: Good
  - 2-5 hours: Moderate
  - > 5 hours: Poor
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **MSE** (Mean Squared Error): Squared error

### Interpreting Regression Targets
The model predicts **different time references** for each class:
- **Infected cells**: Time since infection onset (e.g., if onset = 2h and current = 10h, target = 8h)
- **Uninfected cells**: Elapsed time from experiment start (e.g., if current = 10h, target = 10h)

This preserves temporal information for BOTH classes!

## Troubleshooting

### Poor Classification Performance
1. Check class balance in dataset
2. Try increasing `classification_weight`
3. Use stronger augmentation (color jitter, etc.)
4. Reduce `hidden_dim` if overfitting

### Poor Regression Performance
1. Verify `infection_onset_hour` is correct
2. Try increasing `regression_weight`
3. Check if time annotations are accurate
4. Adjust `clamp_range` if predictions are outside expected range

### Overfitting
Signs: val_loss >> train_loss, or val metrics decrease after initial improvement
Solutions:
- Increase dropout (0.3-0.5)
- Reduce hidden_dim
- Add more data augmentation
- Early stopping (use best validation model, not final)

### Underfitting
Signs: Both train and val loss are high and not improving
Solutions:
- Increase model capacity (larger hidden_dim or backbone)
- Train for more epochs
- Increase learning rate
- Reduce dropout

## Next Steps

1. **Hyperparameter tuning**: Try different loss weights, architectures
2. **Temporal analysis**: Use trained model for time-series predictions
3. **Cross-validation**: Implement K-fold CV for robust evaluation
4. **Ensemble**: Train multiple models and average predictions

## Files

- `train_multitask.py` - Main training script
- `analyze_multitask_results.py` - Analysis and visualization
- `shells/train_multitask.sh` - SLURM batch script
- `configs/multitask_example.yaml` - Configuration template
- `models/multitask_models.py` - Model architecture (if you need to modify)
