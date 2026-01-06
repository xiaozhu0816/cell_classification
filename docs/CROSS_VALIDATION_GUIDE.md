# Adding Cross-Validation to Multitask Training

## Question: Should we add 5-fold cross-validation?

**Short Answer: YES, it would be very valuable!** ✓

---

## Why Cross-Validation is Important

### Current Setup (Single Train/Val/Test Split)
```
Data → [70% Train | 15% Val | 15% Test]
         ↓          ↓         ↓
       Train     Select    Evaluate
                 Best       Final
                 Model     Performance
```

**Issues:**
- Single test set might not be representative
- Model selection based on one validation set
- Can't assess performance variance
- One "lucky" or "unlucky" split affects everything

### With 5-Fold Cross-Validation
```
Fold 1: [Train Train Train Train | Val ] → Model 1
Fold 2: [Train Train Train | Val  Train] → Model 2
Fold 3: [Train Train | Val  Train Train] → Model 3
Fold 4: [Train | Val  Train Train Train] → Model 4
Fold 5: [Val | Train Train Train  Train] → Model 5

Final: Average performance across all 5 folds
       Report: mean ± std
```

**Benefits:**
- ✅ Every sample used for both training and testing
- ✅ More robust performance estimate
- ✅ Can report confidence intervals (mean ± std)
- ✅ Detect overfitting to specific splits
- ✅ Better for publication (more rigorous)

---

## Recommended Approach for Your Research

### Option 1: Stratified 5-Fold CV (RECOMMENDED)

**Best for:**
- Final model evaluation
- Publication-quality results
- Comparing single-task vs multitask
- Statistical significance testing

**Implementation:**
```python
# The dataset already supports this!
for fold in range(5):
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=fold,      # ← Use this!
        num_folds=5,          # ← And this!
    )
    
    # Train model on this fold
    # Evaluate on test fold
    # Save results
```

**Output:**
```
Fold 1: AUC=0.9234, MAE=1.23h
Fold 2: AUC=0.9456, MAE=1.15h
Fold 3: AUC=0.9312, MAE=1.34h
Fold 4: AUC=0.9389, MAE=1.28h
Fold 5: AUC=0.9401, MAE=1.19h
───────────────────────────────
Mean:   AUC=0.9358 ± 0.0084
        MAE=1.24 ± 0.07h
```

### Option 2: Train-Val-Test Split with Multiple Runs

**Best for:**
- Quick experiments
- Hyperparameter tuning
- Initial exploration

**Current approach (what you're doing):**
```
Single run: AUC=0.9999, MAE=1.09h
```

**Problem:** Can't tell if this is typical or just a lucky split!

---

## How Cross-Validation Helps Your Research

### 1. Model Comparison

**Without CV:**
```
Single-task: AUC=0.9823
Multitask:   AUC=0.9999

Conclusion: Multitask is better?
Question: Is this difference significant or just luck?
```

**With CV:**
```
Single-task: AUC=0.9356 ± 0.0123
Multitask:   AUC=0.9401 ± 0.0089

Statistical test:
  t-test: p=0.042 → Significant! ✓
  
Conclusion: Multitask is significantly better
```

### 2. Temporal Generalization

**Without CV:**
```
One sliding window curve
→ Might be lucky/unlucky split
```

**With CV:**
```
5 sliding window curves
→ Average curve with confidence bands
→ "Model performs 0.92±0.03 AUC across all time windows"
```

### 3. Regression Performance

**Without CV:**
```
MAE=1.09h
→ Is this good? Hard to know variance
```

**With CV:**
```
MAE=1.24 ± 0.07h
→ 95% confidence: [1.17h, 1.31h]
→ Much more informative!
```

---

## Implementation Guide

### Step 1: Create CV Training Script

I'll create this for you based on `train_multitask.py`:

```python
# train_multitask_cv.py
for fold_idx in range(num_folds):
    logger.info(f"\n{'='*80}")
    logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
    logger.info(f"{'='*80}\n")
    
    # Build fold-specific datasets
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=fold_idx,
        num_folds=num_folds,
    )
    
    # Train model (same as before)
    # Save fold-specific results
    
    fold_results[fold_idx] = {
        "test_metrics": test_metrics,
        "temporal_metrics": temporal_metrics,
    }

# Aggregate results across folds
aggregate_cv_results(fold_results)
```

### Step 2: Aggregate Results

```python
def aggregate_cv_results(fold_results):
    """Compute mean and std across folds."""
    
    metrics = ["cls_auc", "cls_f1", "reg_mae", "combined"]
    
    for metric in metrics:
        values = [fold["test_metrics"][metric] for fold in fold_results.values()]
        mean = np.mean(values)
        std = np.std(values)
        
        print(f"{metric}: {mean:.4f} ± {std:.4f}")
```

### Step 3: Visualize CV Results

```python
# Plot metrics across folds
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# AUC across folds
axes[0,0].bar(range(5), auc_per_fold)
axes[0,0].errorbar(2, mean_auc, yerr=std_auc, fmt='r*', markersize=15)

# MAE across folds
axes[0,1].bar(range(5), mae_per_fold)
axes[0,1].errorbar(2, mean_mae, yerr=std_mae, fmt='r*', markersize=15)

# Temporal generalization (all folds)
for fold_idx, curve in enumerate(temporal_curves):
    axes[1,0].plot(curve, alpha=0.3, label=f'Fold {fold_idx+1}')
axes[1,0].plot(mean_curve, 'k-', linewidth=3, label='Mean')

# Combined metric distribution
axes[1,1].boxplot(combined_per_fold)
```

---

## Recommended Workflow

### Phase 1: Development (No CV)
```bash
# Fast iteration for hyperparameter tuning
python train_multitask.py --config configs/multitask_example.yaml

# Experiment with:
# - Different infection_onset_hour values
# - Different loss weights
# - Different architectures
```

### Phase 2: Evaluation (With CV)
```bash
# Once you have good hyperparameters, run full CV
python train_multitask_cv.py --config configs/multitask_final.yaml --num-folds 5

# This takes 5x longer but gives robust results
```

### Phase 3: Publication
```
Report:
  "We trained the multitask model using 5-fold cross-validation.
   Average test AUC: 0.9358 ± 0.0084 (mean ± std)
   Average MAE: 1.24 ± 0.07 hours
   
   Compared to single-task model:
   - AUC: +0.0145 (p=0.042, paired t-test)
   - Temporal stability: improved (std=0.03 vs 0.08)"
```

---

## Dataset Already Supports CV!

Good news - your dataset code already has CV support:

```python
# In datasets/timecourse_dataset.py
def build_datasets(
    data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
    fold_index: Optional[int] = None,  # ← Already there!
    num_folds: int = 1,                 # ← Already there!
):
    # ... creates stratified folds
```

**How it works:**
1. `num_folds=1`: Standard train/val/test split (current)
2. `num_folds=5, fold_index=0-4`: 5-fold CV

---

## Time Considerations

### Single Run
```
Training: ~30-60 minutes (30 epochs)
Total: 1 hour
```

### 5-Fold CV
```
Training: ~30-60 minutes × 5 folds
Total: 5-6 hours
```

**Strategy:**
- Use single runs for exploration
- Use CV for final evaluation
- Run CV overnight or on weekends

---

## What I'll Create for You

### 1. train_multitask_cv.py
```python
# Full cross-validation training script
# - Trains on each fold
# - Saves per-fold results
# - Aggregates metrics
# - Creates CV-specific visualizations
```

### 2. analyze_cv_results.py
```python
# Analyzes CV results
# - Plots metrics across folds
# - Computes mean ± std
# - Statistical tests
# - Temporal generalization with confidence bands
```

### 3. Documentation
```
- How to run CV
- How to interpret results
- Statistical testing guide
- Publication-ready plots
```

---

## Recommendation

### For Your Current Work:

**Immediate (Now):**
1. ✅ Generate temporal analysis for current run (I'll fix the script)
2. ✅ Verify all visualizations work
3. ✅ Test with new training run

**Next Steps (This Week):**
1. Create CV training script
2. Run 5-fold CV with current best config
3. Compare results with single run

**For Publication (Later):**
1. Full 5-fold CV on final model
2. Statistical comparison with single-task
3. Report mean ± std for all metrics
4. Include confidence bands on temporal plots

---

## Quick Decision Guide

**Use Single Split If:**
- ✗ Just exploring
- ✗ Tuning hyperparameters
- ✗ Quick experiments
- ✗ Limited time

**Use Cross-Validation If:**
- ✅ Final model evaluation
- ✅ Comparing models
- ✅ Publishing results
- ✅ Want robust estimates
- ✅ Statistical significance needed

---

## Summary

### Should you add 5-fold CV?

**Answer: YES, for final evaluation!**

**Benefits:**
- More robust performance estimates
- Can report mean ± std
- Better for publication
- Statistical comparison possible
- Temporal generalization with confidence

**Implementation:**
- Dataset already supports it
- I'll create the CV training script
- Run it for final results

**Workflow:**
1. Single runs for development ← You're here
2. CV for final evaluation ← Do this next
3. Report CV results in paper ← For publication

Would you like me to create the CV training script now?
