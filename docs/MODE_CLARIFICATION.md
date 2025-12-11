# CLARIFICATION: "test-only" vs "train-test" Modes

## ⚠️ BOTH MODES TRAIN MODELS!

The confusing names refer to **which splits use the restricted interval**, NOT whether training happens.

---

## Mode Name Explanation

### "test-only" mode
- **Better name:** "Train on ALL, test on [1, x]"
- **What happens:**
  1. ✅ **TRAINS** a model on infected [1, FULL] + all uninfected
  2. ✅ **TESTS** that model on infected [1, x] + all uninfected
- **Why "test-only"?** Only the **test split** gets the restricted interval
- **Training data:** Always uses ALL available frames (e.g., [1, 30]h)
- **Test data:** Uses restricted interval [1, x] where x varies (8, 10, 12, ...)

### "train-test" mode  
- **Better name:** "Train on [1, x], test on [1, x]"
- **What happens:**
  1. ✅ **TRAINS** a model on infected [1, x] + all uninfected
  2. ✅ **TESTS** that model on infected [1, x] + all uninfected
- **Why "train-test"?** **Both train AND test** get the same restricted interval
- **Training data:** Uses restricted interval [1, x]
- **Test data:** Uses same restricted interval [1, x]

---

## Visual Comparison for Interval [1, 12]

```
┌────────────────────────────────────────────────────────────────┐
│                   MODE: "test-only"                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  TRAINING PHASE:                                               │
│  ┌──────────────────────────────────────────┐                 │
│  │ Infected frames:   [1, 2, 3, ... 28, 30] │ ← ALL frames   │
│  │ Uninfected frames: [0, 1, 2, ... 92, 94] │ ← ALL frames   │
│  └──────────────────────────────────────────┘                 │
│                ↓                                               │
│         TRAIN NEW MODEL                                        │
│                ↓                                               │
│  TESTING PHASE:                                                │
│  ┌──────────────────────────────────────────┐                 │
│  │ Infected frames:   [1, 2, 3, ... 10, 12] │ ← Restricted!  │
│  │ Uninfected frames: [0, 1, 2, ... 92, 94] │ ← ALL frames   │
│  └──────────────────────────────────────────┘                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                   MODE: "train-test"                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  TRAINING PHASE:                                               │
│  ┌──────────────────────────────────────────┐                 │
│  │ Infected frames:   [1, 2, 3, ... 10, 12] │ ← Restricted!  │
│  │ Uninfected frames: [0, 1, 2, ... 92, 94] │ ← ALL frames   │
│  └──────────────────────────────────────────┘                 │
│                ↓                                               │
│         TRAIN NEW MODEL                                        │
│                ↓                                               │
│  TESTING PHASE:                                                │
│  ┌──────────────────────────────────────────┐                 │
│  │ Infected frames:   [1, 2, 3, ... 10, 12] │ ← Restricted!  │
│  │ Uninfected frames: [0, 1, 2, ... 92, 94] │ ← ALL frames   │
│  └──────────────────────────────────────────┘                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Total Training Runs

For `--upper-hours 8 10 12 14 16 18 20` with `--k-folds 5`:

### "test-only" mode:
- For [1, 8]:  Trains 5 models (one per fold) on [1, FULL], tests on [1, 8]
- For [1, 10]: Trains 5 models (one per fold) on [1, FULL], tests on [1, 10]
- For [1, 12]: Trains 5 models (one per fold) on [1, FULL], tests on [1, 12]
- ... and so on
- **Total:** 7 intervals × 5 folds = **35 training runs**

### "train-test" mode:
- For [1, 8]:  Trains 5 models (one per fold) on [1, 8], tests on [1, 8]
- For [1, 10]: Trains 5 models (one per fold) on [1, 10], tests on [1, 10]
- For [1, 12]: Trains 5 models (one per fold) on [1, 12], tests on [1, 12]
- ... and so on
- **Total:** 7 intervals × 5 folds = **35 training runs**

### Grand Total: 35 + 35 = **70 training runs**

⚠️ **Every run trains a completely new model from scratch!**

---

## Example: What Actually Happens

When you run:
```bash
python analyze_interval_sweep_train.py \
    --upper-hours 8 12 \
    --start-hour 1 \
    --k-folds 2
```

### "test-only" mode (4 training runs):

1. **Interval [1, 8], Fold 1:**
   - Train on: infected [1, 30] + all uninfected
   - Test on: infected [1, 8] + all uninfected
   - Result: AUC = 0.72

2. **Interval [1, 8], Fold 2:**
   - Train on: infected [1, 30] + all uninfected
   - Test on: infected [1, 8] + all uninfected
   - Result: AUC = 0.68

3. **Interval [1, 12], Fold 1:**
   - Train on: infected [1, 30] + all uninfected
   - Test on: infected [1, 12] + all uninfected
   - Result: AUC = 0.85

4. **Interval [1, 12], Fold 2:**
   - Train on: infected [1, 30] + all uninfected
   - Test on: infected [1, 12] + all uninfected
   - Result: AUC = 0.83

**Mean for [1, 8]:** 0.70 ± 0.02
**Mean for [1, 12]:** 0.84 ± 0.01

### "train-test" mode (4 training runs):

1. **Interval [1, 8], Fold 1:**
   - Train on: infected [1, 8] + all uninfected
   - Test on: infected [1, 8] + all uninfected
   - Result: AUC = 0.65

2. **Interval [1, 8], Fold 2:**
   - Train on: infected [1, 8] + all uninfected
   - Test on: infected [1, 8] + all uninfected
   - Result: AUC = 0.63

3. **Interval [1, 12], Fold 1:**
   - Train on: infected [1, 12] + all uninfected
   - Test on: infected [1, 12] + all uninfected
   - Result: AUC = 0.82

4. **Interval [1, 12], Fold 2:**
   - Train on: infected [1, 12] + all uninfected
   - Test on: infected [1, 12] + all uninfected
   - Result: AUC = 0.80

**Mean for [1, 8]:** 0.64 ± 0.01
**Mean for [1, 12]:** 0.81 ± 0.01

---

## Interpretation

In this example:
- **"test-only"** ([1, 8]): 0.70 ← Training on more data helped!
- **"train-test"** ([1, 8]): 0.64 ← Less training data = lower performance

**Conclusion:** For early detection at [1, 8]h, it's better to train on ALL available data even though we'll only see early frames at test time.

---

## Summary

✅ **"test-only" = TRAINS on ALL, tests on restricted [1, x]**
✅ **"train-test" = TRAINS on [1, x], tests on [1, x]**  
✅ **Both modes train fresh models** - no pre-trained checkpoints used
✅ **Total training runs = (# intervals) × 2 modes × (# folds)**

The confusing names come from describing **which splits use the restriction**, not whether training happens!
