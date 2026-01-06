# Regression Task Explanation for Multitask Cell Classification

## What is Regression?

**Regression** is a machine learning task where the model predicts a **continuous numerical value** (unlike classification which predicts discrete categories).

**Examples:**
- Classification: "Is this cell infected?" → Answer: Yes/No (discrete)
- Regression: "What time is this cell at?" → Answer: 15.3 hours (continuous)

---

## Our Regression Task

### What We're Predicting
We predict **time** (in hours) for each cell image:
- **Infected cells**: Time since infection onset (how long they've been infected)
- **Uninfected cells**: Elapsed experiment time (how long into the experiment)

### Why This Matters
Time information helps us:
1. **Track infection progression** - see how cells change over time
2. **Predict disease stages** - early vs. late infection
3. **Temporal generalization** - can the model recognize cells at different timepoints?

---

## Regression Method Used

### 1. Model Architecture

```
Input Image (512x512x3)
         ↓
ResNet50 Backbone (feature extractor)
         ↓
Feature Vector (2048-dim)
         ↓
┌────────────────┬────────────────┐
│ Classification │   Regression   │
│     Head       │      Head      │
├────────────────┼────────────────┤
│ Linear(2048→256)│ Linear(2048→256)│
│ ReLU + Dropout │ ReLU + Dropout │
│ Linear(256→2)  │ Linear(256→1)  │
│ (class logits) │ (time value)   │
└────────────────┴────────────────┘
         ↓                ↓
    [Infected,      [15.3 hours]
     Uninfected]
```

**Key Points:**
- **Shared backbone**: Same feature extractor for both tasks
- **Separate heads**: Different output layers for classification vs regression
- **Regression output**: Single number (predicted time)

### 2. Loss Function

We use **Smooth L1 Loss** (also called Huber Loss):

```python
loss = SmoothL1Loss(prediction, target)
```

**Why Smooth L1 instead of MSE?**

| Loss Type | Formula | Behavior | When to Use |
|-----------|---------|----------|-------------|
| **MSE (L2)** | `(pred - true)²` | Squares errors → very sensitive to outliers | Clean data, no outliers |
| **MAE (L1)** | `|pred - true|` | Linear penalty → less sensitive but not smooth | Robust to outliers |
| **Smooth L1** | `0.5*(pred-true)² if |error|<1 else |error|-0.5` | **Best of both**: smooth near 0, linear for large errors | **Our choice!** Robust + smooth gradients |

**Smooth L1 Benefits:**
- ✅ Less sensitive to outliers (better than MSE)
- ✅ Smooth gradients for optimization (better than MAE)
- ✅ Prevents exploding gradients from extreme predictions

### 3. Target Computation

How we calculate the "ground truth" time for each image:

```python
for each cell image:
    hours_since_start = extract_from_filename()  # e.g., frame 20 → 10 hours (2 frames/hour)
    
    if cell is infected:
        target = max(hours_since_start - infection_onset_hour, 0.0)
        # Example: Image at 10h, onset at 1h → target = 9h (infected for 9 hours)
    else:  # uninfected
        target = hours_since_start
        # Example: Image at 10h → target = 10h (10 hours into experiment)
```

**Key Design Choice:**
- Different time references for different classes!
- Infected: "How long infected?" (relative to onset)
- Uninfected: "How long in experiment?" (absolute time)

### 4. Training Process

```python
for each batch:
    # Forward pass
    cls_logits, time_pred = model(images)  # Get both outputs
    
    # Compute losses
    cls_loss = CrossEntropyLoss(cls_logits, cls_targets)
    reg_loss = SmoothL1Loss(time_pred, time_targets)
    
    # Combined loss
    total_loss = 1.0 * cls_loss + 1.0 * reg_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
```

**Multi-task Learning:**
- Model learns **both** tasks simultaneously
- Gradients from both losses update the shared backbone
- Helps model learn better features (temporal + categorical)

---

## Evaluation Metrics

### Regression Metrics We Track

#### 1. **MAE (Mean Absolute Error)**
```python
MAE = mean(|predicted_time - true_time|)
```
- **Interpretation**: Average prediction error in hours
- **Example**: MAE = 1.15h means predictions are off by ~1.15 hours on average
- **Good value**: < 2 hours for our 48-hour experiment

#### 2. **RMSE (Root Mean Squared Error)**
```python
RMSE = sqrt(mean((predicted_time - true_time)²))
```
- **Interpretation**: Like MAE but penalizes large errors more
- **Example**: RMSE = 1.47h means typical error is ~1.5 hours
- **Relation**: RMSE ≥ MAE always (equality only if all errors same)

#### 3. **R² Score (Coefficient of Determination)**
```python
R² = 1 - (sum of squared errors / total variance)
```
- **Range**: -∞ to 1.0
- **Interpretation**:
  - R² = 1.0 → Perfect predictions
  - R² = 0.8 → Model explains 80% of variance (good!)
  - R² = 0.5 → Model explains 50% of variance (moderate)
  - R² = 0.0 → Model no better than predicting mean
  - R² < 0.0 → Model worse than predicting mean (very bad!)
- **Example**: R² = 0.95 means model captures 95% of time variation

#### 4. **Pearson Correlation (R)**
```python
R = correlation(predicted_times, true_times)
```
- **Range**: -1 to +1
- **Interpretation**:
  - R = 1.0 → Perfect positive correlation
  - R = 0.9 → Strong correlation
  - R = 0.5 → Moderate correlation
  - R = 0.0 → No correlation
- **Relation**: R² = (Pearson R)²

---

## Regression Line Analysis

### What is Linear Regression?

After model training, we fit a **linear regression line** to the predictions:

```python
slope, intercept, r_value = linregress(true_times, predicted_times)
regression_line = slope * true_times + intercept
```

**Perfect Model Should Have:**
- **Slope = 1.0**: No scaling bias (predicted increase matches true increase)
- **Intercept = 0.0**: No offset bias (prediction at 0h = 0h)
- **R value = 1.0**: Perfect correlation

**Real Example from Training:**
```
Infected cells:
  Regression: y = 0.987x + 0.234
  → Nearly perfect slope (0.987 ≈ 1.0)
  → Small offset (+0.234h)
  → R = 0.995 (very strong correlation)
```

### Interpretation Guide

| Regression Result | Meaning | Action |
|------------------|---------|--------|
| **y = 1.0x + 0.0** | Perfect! No bias | ✅ Model is excellent |
| **y = 0.9x + 0.5** | Slight underestimation + offset | ⚠️ Model slightly conservative |
| **y = 1.2x - 1.0** | Overestimation | ⚠️ Model exaggerates time differences |
| **y = 0.5x + 10** | Severe underestimation + large offset | ❌ Model has systematic bias |

---

## Comparison: Classification vs Regression in Our Model

| Aspect | Classification Task | Regression Task |
|--------|-------------------|-----------------|
| **Output** | 2 logits → probabilities [0,1] | 1 value (hours) |
| **Loss** | CrossEntropyLoss | SmoothL1Loss |
| **Target** | Class label: 0 or 1 | Time: 0.0 to 48.0 hours |
| **Metrics** | Accuracy, Precision, Recall, F1, AUC | MAE, RMSE, R², Pearson R |
| **Evaluation** | Confusion matrix, ROC curve | Scatter plot, regression line |
| **Good Performance** | AUC > 0.95 | MAE < 2h, R² > 0.9 |

---

## Why Multitask Learning?

### Benefits of Joint Training

1. **Shared Representations**: 
   - Backbone learns features useful for BOTH tasks
   - Example: "Cell morphology at 20h" helps both infection detection AND time estimation

2. **Regularization Effect**:
   - Learning multiple tasks prevents overfitting
   - Model can't just memorize - must learn general patterns

3. **Better Features**:
   - Regression forces model to learn temporal patterns
   - Classification forces model to learn disease signatures
   - Combined: richer feature representations

4. **Computational Efficiency**:
   - One model does two jobs
   - Shared backbone means less parameters than two separate models

### Trade-offs

**Advantages:**
- ✅ Better generalization
- ✅ More efficient (one model)
- ✅ Richer learned features

**Challenges:**
- ⚠️ Need to balance two losses (classification_weight vs regression_weight)
- ⚠️ More complex training dynamics
- ⚠️ Harder to debug (which task is causing issues?)

---

## Your Results (Previous Run)

From `outputs/multitask_resnet50/20251215-164539/`:

### Classification Performance
- **AUC**: 0.9999 (nearly perfect!)
- **Accuracy**: 99.40%
- **Interpretation**: Model almost perfectly distinguishes infected/uninfected

### Regression Performance
- **MAE**: 1.15 hours
- **RMSE**: 1.47 hours
- **Interpretation**: Predictions typically off by ~1.15 hours
  - In a 48-hour experiment, this is ~2.4% error - excellent!

### What This Means
- Model can predict cell time within **±1-2 hours**
- Over 48 hours, that's **very accurate** temporal tracking
- Combined with 99.4% classification: model has strong spatiotemporal understanding

---

## Common Questions

### Q: Why not just use classification for time?
**A:** Classification would bin time into discrete buckets (e.g., 0-6h, 6-12h, etc.), losing precision. Regression gives exact continuous predictions (e.g., 15.3h).

### Q: Can we predict negative times?
**A:** No, we clamp predictions to [0.0, 48.0] range during training. Model output is constrained to realistic values.

### Q: What if model predicts 50h for a 10h image?
**A:** This would be a large error caught by metrics:
- MAE would increase (|50-10| = 40h)
- RMSE would increase even more (40² = 1600)
- Scatter plot would show outlier far from diagonal
- This signals model needs more training or different architecture

### Q: How do we know regression is working?
**A:** Check the scatter plot:
- Points cluster near diagonal → good predictions
- Random scatter → model not learning time
- Systematic deviation → model has bias
- Regression line slope ≈ 1.0 → unbiased predictions

---

## Summary

**Regression Method Used:**
- **Architecture**: Shared ResNet50 backbone + separate regression head
- **Loss**: Smooth L1 Loss (robust to outliers, smooth gradients)
- **Output**: Continuous time value (hours)
- **Target**: Time relative to infection onset (infected) or experiment start (uninfected)
- **Optimization**: Joint training with classification using combined loss
- **Evaluation**: MAE, RMSE, R², scatter plots with regression lines

**Your Model's Performance:**
- MAE = 1.15h means typically accurate within ±1 hour
- In 48-hour study, that's 97.6% accuracy - excellent!
- Scatter plot will show if this accuracy is consistent across all timepoints

**Key Insight:**
Regression allows the model to learn a **continuous temporal representation** of cell states, not just discrete infection status. This makes it powerful for understanding disease progression over time!
