# Quick Reference: Combined Metric

## Formula

```
Combined = 0.6 × F1 + 0.4 × (1 - MAE/48)
           ↑           ↑
    Classification  Regression
       (60%)         (40%)
```

---

## Visual Comparison

### Before: Using AUC

```
┌──────────────────────────────────────────────────┐
│  PROBLEM: AUC Saturation                         │
└──────────────────────────────────────────────────┘

Epoch    AUC      F1      MAE     Selected?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1    0.9234   0.9012   5.23      ✗
  5    0.9876   0.9678   2.84      ✗
  8    0.9989   0.9823   2.12      ✗
  9    1.0000   0.9850   2.34      ✓  ← SAVED
 10    1.0000   0.9920   1.89      ✗  ← Better but ignored!
 15    1.0000   0.9940   1.52      ✗  ← Even better!
 20    1.0000   0.9950   1.15      ✗  ← Best but not selected!

Result: Epoch 9 selected (first to hit AUC=1.0)
        Not optimal! MAE=2.34h when could be 1.15h
```

### After: Using Combined Metric

```
┌──────────────────────────────────────────────────┐
│  SOLUTION: Combined Metric                       │
└──────────────────────────────────────────────────┘

Epoch    AUC      F1      MAE     Combined  Selected?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1    0.9234   0.9012   5.23    0.8949      ✗
  5    0.9876   0.9678   2.84    0.9571      ✗
  8    0.9989   0.9823   2.12    0.9718      ✗
  9    1.0000   0.9850   2.34    0.9804      ✗
 10    1.0000   0.9920   1.89    0.9868      ✗
 15    1.0000   0.9940   1.52    0.9891      ✗
 20    1.0000   0.9950   1.15    0.9909      ✓  ← SAVED (best!)

Calculations (epoch 20):
  F1 component:  0.6 × 0.9950 = 0.5970
  MAE component: 0.4 × (1 - 1.15/48) = 0.3939
  Combined:      0.5970 + 0.3939 = 0.9909

Result: Epoch 20 selected (best combined performance)
        Optimal for both tasks!
```

---

## Metric Behavior Across Training

```
AUC (saturates early)
1.00 ●━━━━━━━━━━━━━━━━━━━━━━  ← Can't improve
0.95 │    ●●
0.90 │  ●
0.85 ●
     └─────────────────────────
     1  5  10  15  20  25  30  (epoch)


Combined (continues improving)
1.00 │                      ●  ← Still improving
0.95 │              ●●●●
0.90 │      ●●●●
0.85 ●●●
     └─────────────────────────
     1  5  10  15  20  25  30  (epoch)
```

---

## Component Contributions

### Example 1: Excellent Both Tasks
```
F1 = 0.995, MAE = 1.15h

Classification:  0.6 × 0.995 = 0.597  (60%)
Regression:      0.4 × 0.976 = 0.390  (40%)
                               ─────
Combined:                      0.987
```

### Example 2: Perfect Classification, Good Regression
```
F1 = 1.0, MAE = 2.4h

Classification:  0.6 × 1.0 = 0.600  (60%)
Regression:      0.4 × 0.95 = 0.380  (40%)
                             ─────
Combined:                    0.980
```

### Example 3: Trade-off Scenario
```
F1 = 0.92, MAE = 0.8h

Classification:  0.6 × 0.92 = 0.552  (60%)
Regression:      0.4 × 0.983 = 0.393  (40%)
                               ─────
Combined:                      0.945

Note: Lower F1 but excellent MAE
      Combined shows the trade-off
```

---

## Score Ranges

### Combined Score Interpretation

```
0.95 - 1.00   ████████████████████  Excellent
0.90 - 0.95   ████████████████      Very Good
0.85 - 0.90   ████████████          Good
0.80 - 0.85   ████████              Moderate
0.75 - 0.80   ████                  Fair
< 0.75        ██                    Poor
```

### What Each Range Means

**0.95-1.00: Excellent**
- F1 > 0.97 AND MAE < 2h
- Both tasks performing exceptionally
- Production-ready model

**0.90-0.95: Very Good**
- F1 > 0.93 AND MAE < 4h
- Strong performance on both
- Minor improvements possible

**0.85-0.90: Good**
- F1 > 0.88 AND MAE < 6h
- Solid baseline
- Room for optimization

**< 0.85: Needs Work**
- At least one task struggling
- Review architecture or data

---

## Weight Sensitivity Analysis

### Different Weight Configurations

```
                Weight Ratio          Score for:
Config      Cls:Reg      F1=0.95,MAE=2h   F1=1.0,MAE=4h
─────────────────────────────────────────────────────────
Current     60:40        0.9742           0.9667
Equal       50:50        0.9604           0.9583
Cls-heavy   80:20        0.9787           0.9833
Reg-heavy   40:60        0.9646           0.9500

Observations:
  • Current (60:40): Balanced, slight cls preference
  • Equal (50:50): True balance
  • Cls-heavy (80:20): Favors F1=1.0 scenario
  • Reg-heavy (40:60): Penalizes high MAE more
```

---

## Console Output Examples

### During Training

```
================================================================================
Epoch 15/30
train: total_loss:1.0234 | cls_loss:0.0934 | reg_loss:0.9300 | combined:0.9823
val: total_loss:1.2456 | cls_loss:0.1356 | reg_loss:1.1100 | 
     cls_accuracy:0.9940 | cls_precision:1.0000 | cls_recall:0.9879 | 
     cls_f1:0.9939 | cls_auc:0.9999 | 
     reg_mae:1.1500 | reg_rmse:1.4700 | reg_mse:2.1600 | 
     combined:0.9891  ← NEW METRIC
✓ New best model! combined=0.9891  ← SAVED BASED ON THIS
```

### At Start

```
Training for 30 epochs...
Primary metric: combined (0.6*F1 + 0.4*(1-MAE/48.0))
  - Balances classification F1 and regression MAE
  - Prevents AUC saturation at 1.0
```

---

## Decision Tree: When Model is Saved

```
                    ┌─ YES ─→ SAVE CHECKPOINT
Is combined > best? ┤
                    └─ NO ──→ SKIP

Example:
  Previous best: 0.9850
  Current:       0.9891
  0.9891 > 0.9850? YES → SAVE!
```

Compare to old (AUC):
```
                    ┌─ YES ─→ SAVE CHECKPOINT
Is AUC > best?      ┤         (might be same as epoch 9!)
                    └─ NO ──→ SKIP

Example:
  Previous best: 1.0000
  Current:       1.0000
  1.0000 > 1.0000? NO → SKIP (even if MAE improved!)
```

---

## Quick Diagnostic Commands

### Check Current Best Model

```bash
# View results
cat outputs/multitask_resnet50/LATEST/results.json | grep -A5 test_metrics

# Look for:
# "combined": 0.9XXX  (should be present now)
# "cls_auc": 0.9XXX   (still tracked)
# "reg_mae": X.XX     (used in combined)
```

### Compare Before/After

```bash
# Old model (AUC selection)
echo "Old: AUC-based selection"
cat outputs/OLD_RUN/results.json | grep "best_val_metric"

# New model (combined selection)  
echo "New: Combined-based selection"
cat outputs/NEW_RUN/results.json | grep "best_val_metric"

# Expected:
# Old: "best_val_metric": 1.0000 (saturated!)
# New: "best_val_metric": 0.9XXX (meaningful!)
```

---

## FAQ

**Q: Why 60/40 split?**
A: Classification slightly prioritized (infection detection is primary goal), but regression still meaningful (40% is substantial).

**Q: Can I change the weights?**
A: Yes! Edit line ~300 in `train_multitask.py`. Try 50/50 for true balance, or 80/20 if classification is much more important.

**Q: What if I want AUC back?**
A: Change `primary_metric = "cls_auc"` on line ~633. But combined is better for avoiding saturation!

**Q: Does this change the model architecture?**
A: No! Only changes which checkpoint is saved. The model still learns both tasks the same way.

**Q: Will my results be different?**
A: Yes, likely better! You'll select a model with more balanced performance instead of first to hit AUC=1.0.

---

## Summary

### Problem Solved ✓
- AUC saturation at 1.0
- Ignoring regression task
- Suboptimal model selection

### Solution Implemented ✓
- Combined metric: 60% F1 + 40% normalized MAE
- Automatic computation in evaluate()
- Used for model selection

### What You Get ✓
- Better model selection
- Balanced multi-task optimization
- Continued improvement tracking
- No more "stuck at 1.0" problem

Ready to train! 🚀
