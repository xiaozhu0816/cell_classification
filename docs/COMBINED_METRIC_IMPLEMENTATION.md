# Combined Metric Implementation - Summary

## Changes Made to `train_multitask.py`

### ✅ Successfully Implemented Combined Metric for Model Selection

---

## What Changed

### 1. Added Combined Metric Computation in `evaluate()` Function

**Location:** After computing all metrics (~line 300)

**Code Added:**
```python
# Compute combined metric for model selection
# Balances classification F1 and regression MAE
cls_f1 = metrics.get("cls_f1", 0.0)
reg_mae = metrics.get("reg_mae", clamp_range[1])
max_time = clamp_range[1]

# Normalize regression: 0 MAE → 1.0, max_time MAE → 0.0
reg_score = max(0.0, 1.0 - (reg_mae / max_time))

# Weighted combination: 60% classification, 40% regression
combined_metric = 0.6 * cls_f1 + 0.4 * reg_score
metrics["combined"] = combined_metric
```

**What it does:**
- Computes a combined score balancing both tasks
- F1 score (classification): 60% weight
- Normalized MAE (regression): 40% weight
- Range: [0, 1], higher is better
- Added to metrics dict automatically

### 2. Changed Primary Metric Selection

**Location:** Main training function (~line 633)

**Before:**
```python
# Primary metric for model selection (use classification AUC)
primary_metric = "cls_auc"
best_score = -math.inf

logger.info(f"Training for {epochs} epochs...")
logger.info(f"Primary metric: {primary_metric}")
```

**After:**
```python
# Primary metric for model selection
# Use combined metric to balance classification and regression performance
# This prevents saturation issues when AUC reaches 1.0
primary_metric = "combined"
best_score = -math.inf

logger.info(f"Training for {epochs} epochs...")
logger.info(f"Primary metric: {primary_metric} (0.6*F1 + 0.4*(1-MAE/{clamp_range[1]}))")
logger.info(f"  - Balances classification F1 and regression MAE")
logger.info(f"  - Prevents AUC saturation at 1.0")
```

**What changed:**
- Now selects best model based on `combined` metric instead of `cls_auc`
- Logs the formula for transparency
- Explains the rationale

---

## How the Combined Metric Works

### Formula

```
combined = 0.6 × F1 + 0.4 × (1 - MAE/max_time)

Where:
  F1: Classification F1 score [0, 1]
  MAE: Regression mean absolute error [0, max_time]
  max_time: Maximum time in hours (from clamp_range, typically 48)
```

### Component Breakdown

#### Classification Component (60% weight)

```python
cls_component = 0.6 * cls_f1

Examples:
  F1 = 1.0   → contribution = 0.600
  F1 = 0.95  → contribution = 0.570
  F1 = 0.90  → contribution = 0.540
```

**Why F1 instead of AUC?**
- More sensitive to improvements at high performance
- Less likely to saturate at 1.0
- Balances precision and recall

#### Regression Component (40% weight)

```python
reg_score = 1.0 - (MAE / max_time)
reg_component = 0.4 * reg_score

Examples (max_time = 48):
  MAE = 0.0h  → reg_score = 1.0  → contribution = 0.400
  MAE = 1.2h  → reg_score = 0.975 → contribution = 0.390
  MAE = 4.8h  → reg_score = 0.9   → contribution = 0.360
  MAE = 24h   → reg_score = 0.5   → contribution = 0.200
```

**Why normalize MAE?**
- Brings it to [0, 1] range to match F1
- Makes weighting meaningful
- 0 MAE = perfect (1.0), max_time MAE = worst (0.0)

### Combined Examples

**Scenario 1: Excellent Classification, Good Regression**
```
F1 = 0.995, MAE = 1.15h (max_time = 48)

cls_component = 0.6 × 0.995 = 0.597
reg_score = 1 - (1.15/48) = 0.976
reg_component = 0.4 × 0.976 = 0.390

combined = 0.597 + 0.390 = 0.987
```

**Scenario 2: Perfect Classification, Moderate Regression**
```
F1 = 1.0, MAE = 2.4h (max_time = 48)

cls_component = 0.6 × 1.0 = 0.600
reg_score = 1 - (2.4/48) = 0.950
reg_component = 0.4 × 0.950 = 0.380

combined = 0.600 + 0.380 = 0.980
```

**Scenario 3: Good Both Tasks**
```
F1 = 0.92, MAE = 1.0h (max_time = 48)

cls_component = 0.6 × 0.92 = 0.552
reg_score = 1 - (1.0/48) = 0.979
reg_component = 0.4 × 0.979 = 0.392

combined = 0.552 + 0.392 = 0.944
```

**Key insight:** Scenario 3 scores lower than 1 & 2 despite similar MAE, because F1 is lower. This shows the metric balances both tasks!

---

## Why This Solves Your Problem

### Problem: AUC Saturation

**Before:**
```
Epoch  Val_AUC  Val_F1   Val_MAE  Selected?
─────────────────────────────────────────────
  5     1.0000   0.9850   2.34    ✓ (first to hit 1.0)
 10     1.0000   0.9920   1.89    ✗ (same AUC)
 15     1.0000   0.9940   1.52    ✗ (same AUC)
 20     1.0000   0.9950   1.15    ✗ (same AUC)
```

Model selection: **Epoch 5** (suboptimal!)
- Best by AUC (1.0) but worst F1 and MAE
- Selected because it hit 1.0 first

**After (with combined metric):**
```
Epoch  Val_AUC  Val_F1   Val_MAE  Combined  Selected?
──────────────────────────────────────────────────────
  5     1.0000   0.9850   2.34    0.9614    ✗
 10     1.0000   0.9920   1.89    0.9743    ✗
 15     1.0000   0.9940   1.52    0.9828    ✗
 20     1.0000   0.9950   1.15    0.9874    ✓ (highest combined)

Calculations:
  Epoch 5:  0.6×0.985 + 0.4×(1-2.34/48) = 0.591 + 0.371 = 0.9614
  Epoch 20: 0.6×0.995 + 0.4×(1-1.15/48) = 0.597 + 0.390 = 0.9874
```

Model selection: **Epoch 20** (optimal!)
- Best overall performance
- Combined metric captures continued improvement

### Problem: Ignoring Regression

**Before:**
- Only looked at classification AUC
- Regression performance irrelevant to model selection
- Could select model with good AUC but terrible MAE

**After:**
- Both tasks contribute to selection
- 60% classification, 40% regression
- True multi-task optimization!

---

## Expected Impact on Training

### Console Output Changes

**New logging during training:**
```
Training for 30 epochs...
Primary metric: combined (0.6*F1 + 0.4*(1-MAE/48.0))
  - Balances classification F1 and regression MAE
  - Prevents AUC saturation at 1.0
```

**Validation metrics will now include:**
```
val: total_loss:1.2345 | cls_loss:0.1234 | reg_loss:1.1111 | 
     cls_accuracy:0.9876 | cls_precision:1.0000 | cls_recall:0.9753 | 
     cls_f1:0.9875 | cls_auc:0.9999 | 
     reg_mae:1.2345 | reg_rmse:1.5678 | reg_mse:2.4567 | 
     combined:0.9845  ← NEW!
```

**Model saving messages:**
```
✓ New best model! combined=0.9845

Instead of:
✓ New best model! cls_auc=1.0000
```

### What to Expect

1. **Different Model Selected**
   - May not be the same epoch as before
   - Likely later epoch (better regression)
   - More balanced performance

2. **Checkpoint Changes**
   - `best.pt` will be saved based on combined metric
   - May have slightly lower AUC but better MAE
   - Overall better multi-task performance

3. **Results JSON**
   - `best_val_metric` will be combined score (0-1 range)
   - All individual metrics still tracked
   - Can compare: "Was combined score 0.98 with AUC=0.999 better than combined=0.96 with AUC=1.0?"

---

## How to Use

### Run Training (Same Command)

```bash
python train_multitask.py --config configs/multitask_example.yaml
```

**No changes needed!** The combined metric is now automatic.

### Interpret Results

**After training, check `results.json`:**
```json
{
  "best_val_metric": 0.9845,  // Combined score, not AUC!
  "test_metrics": {
    "combined": 0.9832,        // Test combined score
    "cls_auc": 0.9999,         // Still tracked
    "cls_f1": 0.9875,          // Used in combined
    "reg_mae": 1.15,           // Used in combined
    ...
  }
}
```

**Interpreting combined score:**
- **0.95-1.0**: Excellent (both tasks performing well)
- **0.90-0.95**: Good (balanced performance)
- **0.85-0.90**: Moderate (room for improvement)
- **<0.85**: Poor (at least one task struggling)

---

## Customization Options

### Adjust Weights

If you want different balance between tasks:

**Edit line ~300 in `train_multitask.py`:**
```python
# Current: 60% classification, 40% regression
combined_metric = 0.6 * cls_f1 + 0.4 * reg_score

# Option 1: Prioritize classification (80/20)
combined_metric = 0.8 * cls_f1 + 0.2 * reg_score

# Option 2: Equal weight (50/50)
combined_metric = 0.5 * cls_f1 + 0.5 * reg_score

# Option 3: Prioritize regression (30/70)
combined_metric = 0.3 * cls_f1 + 0.7 * reg_score
```

### Switch Back to AUC (If Needed)

**Edit line ~633:**
```python
# Revert to AUC
primary_metric = "cls_auc"

# Or use F1 only
primary_metric = "cls_f1"

# Or use regression only (note: lower is better!)
primary_metric = "reg_mae"
best_score = math.inf  # Change to positive infinity!
# And flip comparison in checkpoint saving
```

### Use Different Classification Metric

**Edit line ~300:**
```python
# Use AUC instead of F1
cls_auc = metrics.get("cls_auc", 0.0)
combined_metric = 0.6 * cls_auc + 0.4 * reg_score

# Use accuracy instead
cls_acc = metrics.get("cls_accuracy", 0.0)
combined_metric = 0.6 * cls_acc + 0.4 * reg_score
```

---

## Validation

### Check That It Works

After implementing, verify:

1. **Metric appears in logs:**
   ```bash
   python train_multitask.py --config configs/multitask_example.yaml
   
   # Look for:
   # "Primary metric: combined (0.6*F1 + 0.4*(1-MAE/48.0))"
   # "combined:0.XXXX" in validation outputs
   ```

2. **Checkpoint saves on combined:**
   ```bash
   # Should see:
   # "✓ New best model! combined=0.XXXX"
   # NOT:
   # "✓ New best model! cls_auc=1.0000"
   ```

3. **Results file has combined:**
   ```bash
   cat outputs/.../results.json | grep combined
   
   # Should find:
   # "best_val_metric": 0.9XXX (not 1.0!)
   # "combined": 0.9XXX in test_metrics
   ```

---

## Comparison: Before vs After

### Model Selection Behavior

| Aspect | Before (AUC) | After (Combined) |
|--------|-------------|------------------|
| **Metric** | Classification AUC only | F1 + normalized MAE |
| **Range** | [0, 1] | [0, 1] |
| **Saturation** | ✗ Frequently hits 1.0 | ✓ Less likely to saturate |
| **Regression** | ✗ Ignored | ✓ 40% weight |
| **Sensitivity** | Low at high values | High across range |
| **Selected Model** | First to hit 1.0 | Best overall balance |

### Example Training Run

**Before (AUC=1.0 problem):**
```
Epoch 8: AUC=0.9989 → not saved
Epoch 9: AUC=1.0000 → SAVED (first perfect)
Epoch 10: AUC=1.0000, F1=0.995, MAE=2.1 → not saved (same AUC)
Epoch 15: AUC=1.0000, F1=0.997, MAE=1.5 → not saved (same AUC)
Epoch 20: AUC=1.0000, F1=0.998, MAE=1.1 → not saved (same AUC)

Result: Selected epoch 9 (suboptimal regression!)
```

**After (Combined metric):**
```
Epoch 8: Combined=0.9725 → not saved
Epoch 9: Combined=0.9800 → SAVED
Epoch 10: Combined=0.9815 → SAVED (better!)
Epoch 15: Combined=0.9858 → SAVED (better!)
Epoch 20: Combined=0.9882 → SAVED (best!)

Result: Selected epoch 20 (optimal for both tasks!)
```

---

## Troubleshooting

### Issue: Combined metric looks weird

**Check normalization:**
```python
# In evaluate() function, verify:
max_time = clamp_range[1]  # Should be 48.0 or your max time
print(f"Max time: {max_time}")
print(f"MAE: {reg_mae:.2f}")
print(f"Normalized: {1 - reg_mae/max_time:.4f}")
```

### Issue: All combined scores very low

**Possible cause:** Weights might be off or MAE is very high

**Check:**
- Is MAE reasonable? (should be < 5 hours typically)
- Are weights summing to 1.0? (0.6 + 0.4 = 1.0 ✓)
- Is F1 computed correctly?

### Issue: Want to see breakdown

**Add debug logging:**
```python
# In evaluate() after computing combined
logger.info(f"  Combined breakdown: F1={cls_f1:.4f} (→{0.6*cls_f1:.4f}), "
           f"MAE={reg_mae:.2f}h (→{0.4*reg_score:.4f})")
```

---

## Summary

### ✅ Implementation Complete

- Combined metric added to `evaluate()` function
- Primary metric changed from `cls_auc` to `combined`
- Logging updated to show formula and rationale
- No syntax errors, ready to use!

### 🎯 Benefits

1. **Solves AUC saturation**: Won't get stuck at 1.0
2. **Optimizes both tasks**: 60% classification, 40% regression
3. **Better model selection**: Selects truly best model, not just first perfect AUC
4. **More meaningful**: Reflects multi-task learning goals
5. **Still tracks everything**: All individual metrics still available

### 📝 Next Steps

1. **Run new training:**
   ```bash
   python train_multitask.py --config configs/multitask_example.yaml
   ```

2. **Compare results:**
   - Old model (AUC selection): Check previous `results.json`
   - New model (combined selection): Check new `results.json`
   - Which has better balance?

3. **Analyze improvement:**
   - Did combined metric select a different epoch?
   - Is regression MAE better?
   - Is classification still excellent?

The combined metric will help you select models that excel at **both** infection detection and time prediction! 🚀
