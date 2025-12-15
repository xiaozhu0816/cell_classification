# Multi-Task Learning for Cell Classification and Time Prediction

## Overview

This multi-task learning framework simultaneously trains a model to:

1. **Classify** cells as infected or uninfected (binary classification)
2. **Predict time** for both infected and uninfected cells (regression)
   - **Infected cells**: Time since infection onset (e.g., "infected 5 hours ago")
   - **Uninfected cells**: Elapsed time from experiment start (e.g., "10 hours into experiment")

### Why Multi-Task Learning?

**Problem with previous regression approach:**
- All uninfected samples were assigned target `t=0`, losing temporal information
- Uninfected cells at 0h, 10h, and 48h were treated identically
- Model couldn't learn temporal progression of uninfected cells

**Multi-task solution:**
- ✅ Preserves temporal information for **both** infected and uninfected cells
- ✅ Classification and regression tasks help each other through shared features
- ✅ More biologically meaningful: uninfected cells at different times ARE different

## Architecture

```
Input Image (cell)
       ↓
ResNet Backbone (shared feature extractor)
       ├──> Classification Head → [logit_uninfected, logit_infected]
       └──> Regression Head → time_prediction (hours)
```

**Shared Backbone**: ResNet-18/34/50/101/152 extracts visual features from cell images

**Classification Head**: 
- 2-layer MLP with ReLU and dropout
- Output: Binary logits (infected vs uninfected)
- Loss: Binary Cross-Entropy

**Regression Head**:
- 2-layer MLP with ReLU and dropout  
- Output: Time prediction (scalar)
- Loss: Smooth L1 Loss

**Total Loss**: `α × classification_loss + β × regression_loss`

## Regression Targets

The key innovation is different time references for different classes:

| Cell State | Regression Target | Example |
|------------|------------------|---------|
| **Infected** | `max(hours_since_start - infection_onset, 0)` | If infected at t=2h and current t=10h → target=8h |
| **Uninfected** | `hours_since_start` | If current t=10h → target=10h |

This means:
- Infected cells learn: "How long have I been infected?"
- Uninfected cells learn: "How much time has elapsed in the experiment?"

Both are biologically meaningful and preserve temporal information!

## Files

- **`models/multitask_resnet.py`**: Multi-task model architecture
- **`train_multitask.py`**: Training script for multi-task learning
- **`configs/multitask_example.yaml`**: Example configuration file

## Quick Start

### 1. Prepare Your Config

Copy and modify `configs/multitask_example.yaml`:

```yaml
experiment_name: my_multitask_experiment

data:
  data_dir: /path/to/tiff/files
  label_file: /path/to/labels.csv
  train_stacks: ["stack1", "stack2"]
  val_stacks: ["stack3"]
  test_stacks: ["stack4"]

multitask:
  infection_onset_hour: 2.0  # When infection occurs
  clamp_range: [0.0, 48.0]   # Min/max time bounds
  classification_weight: 1.0  # α in total loss
  regression_weight: 1.0      # β in total loss

model:
  name: resnet50
  pretrained: true
  dropout: 0.2
  hidden_dim: 256

training:
  epochs: 50
  batch_size: 32
  amp: true  # Mixed precision
```

### 2. Train

```bash
python train_multitask.py --config configs/my_config.yaml
```

### 3. Monitor Training

The script logs:
- **Total loss**: Combined classification + regression loss
- **Classification metrics**: AUC, accuracy, precision, recall, F1
- **Regression metrics**: MAE, RMSE, MSE

Example output:
```
Epoch 10/50
train - total_loss: 0.4523 | cls_loss: 0.2145 | reg_loss: 0.2378
val: total_loss:0.3821 | cls_loss:0.1654 | reg_loss:0.2167 | cls_auc:0.9234 | cls_acc:0.8567 | reg_mae:2.345 | reg_rmse:3.456
✓ New best model! cls_auc=0.9234
```

### 4. Results

After training, check `outputs/{experiment_name}/{run_id}/`:
- `train.log`: Full training log
- `checkpoints/best.pt`: Best model checkpoint
- `results.json`: Final test metrics

## Configuration Details

### Multi-Task Settings

```yaml
multitask:
  infection_onset_hour: 2.0
    # When does infection occur (hours from experiment start)?
    # Used to compute "time since infection" for infected cells
  
  clamp_range: [0.0, 48.0]
    # Clamp all time predictions to this range
    # Prevents extreme values
  
  classification_weight: 1.0
    # Weight for classification loss (α)
    # Increase to prioritize classification accuracy
  
  regression_weight: 1.0
    # Weight for regression loss (β)
    # Increase to prioritize time prediction accuracy
```

**Tuning Loss Weights:**

- **Balanced** (α=1, β=1): Equal importance to both tasks
- **Classification-focused** (α=2, β=1): Prioritize infection detection
- **Regression-focused** (α=1, β=2): Prioritize accurate time prediction
- **Auto-balancing**: Try α=1, β=0.1 initially, then adjust based on val metrics

### Model Settings

```yaml
model:
  name: resnet50
    # Backbone architecture
    # Options: resnet18, resnet34, resnet50, resnet101, resnet152
    # Larger = more capacity but slower
  
  pretrained: true
    # Use ImageNet pretrained weights (recommended!)
  
  dropout: 0.2
    # Dropout rate in task heads (prevents overfitting)
  
  hidden_dim: 256
    # Hidden layer size in classification/regression heads
    # Set to 0 for single linear layer (simpler but less capacity)
  
  train_backbone: true
    # Fine-tune ResNet backbone or freeze it
    # true = better performance, false = faster training
```

## Understanding the Output

### During Training

**For each epoch**, you'll see:

1. **Training metrics**:
   - `total_loss`: Combined multi-task loss
   - `cls_loss`: Classification loss component
   - `reg_loss`: Regression loss component

2. **Validation metrics**:
   - **Classification**: `cls_auc`, `cls_acc`, `cls_precision`, `cls_recall`, `cls_f1`
   - **Regression**: `reg_mae`, `reg_rmse`, `reg_mse`

### Model Selection

The best model is saved based on **validation classification AUC** (`cls_auc`).

Rationale: Classification (infected vs uninfected) is typically the primary task, with time prediction as auxiliary information.

You can modify this in `train_multitask.py` line 431:
```python
primary_metric = "cls_auc"  # Change to "reg_mae" or others
```

## Example Use Case

**Scenario**: You have time-lapse microscopy of cells exposed to infection.

**Dataset**:
- Images at t = 0, 2, 4, 6, ..., 48 hours
- Infection occurs at t = 2 hours
- Labels: infected (1) or uninfected (0)

**What the model learns**:

| Image | True Label | True Time | Classification Prediction | Time Prediction |
|-------|------------|-----------|--------------------------|-----------------|
| Cell at t=0h | Uninfected (0) | 0h | P(infected)=0.05 | ~0h |
| Cell at t=10h | Uninfected (0) | 10h | P(infected)=0.08 | ~10h |
| Cell at t=20h | Uninfected (0) | 20h | P(infected)=0.12 | ~20h |
| Cell at t=6h | Infected (1) | 4h since infection | P(infected)=0.92 | ~4h |
| Cell at t=24h | Infected (1) | 22h since infection | P(infected)=0.95 | ~22h |

**Insight**: The model learns:
1. Visual markers of infection (classification)
2. How infected cells change over time (regression for infected)
3. How healthy cells age naturally (regression for uninfected)

## Advanced: Custom Metrics

To add custom metrics, modify the `evaluate()` function in `train_multitask.py`.

Example - Add separate metrics for infected vs uninfected:

```python
# Split by class
infected_mask = all_cls_targets == 1
uninfected_mask = all_cls_targets == 0

# Infected-only regression metrics
if infected_mask.sum() > 0:
    infected_mae = np.abs(all_time_preds[infected_mask] - all_time_targets[infected_mask]).mean()
    metrics["reg_infected_mae"] = infected_mae

# Uninfected-only regression metrics
if uninfected_mask.sum() > 0:
    uninfected_mae = np.abs(all_time_preds[uninfected_mask] - all_time_targets[uninfected_mask]).mean()
    metrics["reg_uninfected_mae"] = uninfected_mae
```

## Troubleshooting

### Loss Imbalance

**Problem**: One loss dominates (e.g., `cls_loss` >> `reg_loss`)

**Solution**: Adjust loss weights in config:
```yaml
multitask:
  classification_weight: 1.0
  regression_weight: 5.0  # Increase to balance
```

### Poor Classification Performance

**Problem**: Low `cls_auc` or `cls_acc`

**Solutions**:
1. Increase `classification_weight`
2. Use larger model (resnet50 → resnet101)
3. Add more training data
4. Check class balance (use class-balanced sampling)

### Poor Regression Performance

**Problem**: High `reg_mae` or `reg_rmse`

**Solutions**:
1. Increase `regression_weight`
2. Check if `clamp_range` is appropriate
3. Verify `infection_onset_hour` is correct
4. Increase `hidden_dim` in model config

### GPU Out of Memory

**Problem**: CUDA OOM error

**Solutions**:
1. Reduce `batch_size` (e.g., 32 → 16)
2. Use smaller model (resnet50 → resnet34 → resnet18)
3. Disable mixed precision: `amp: false`
4. Reduce `hidden_dim`

## Comparison with Single-Task

| Aspect | Single-Task Regression | Multi-Task |
|--------|----------------------|------------|
| Uninfected targets | All = 0 ❌ | Actual elapsed time ✅ |
| Classification | Separate model needed | Built-in ✅ |
| Temporal info | Lost for uninfected ❌ | Preserved for all ✅ |
| Training efficiency | One task at a time | Both tasks jointly ✅ |
| Feature sharing | None | Shared backbone ✅ |

## Citation

If you use this multi-task framework, please cite:

```
[Your paper/project citation here]
```

## Questions?

For issues or questions, please contact [your contact info].
