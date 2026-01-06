# Updated Interval Sweep Logic - Test-Only Mode

## What Changed

### OLD Behavior (Before):
**Test-Only Mode:**
- Train on [1-46h] infected + [0-48h] uninfected
- Test this SAME model on different intervals: [1-7h], [1-10h], ..., [1-46h]
- **Problem:** At x=46h, blue and orange lines were DIFFERENT due to uninfected distribution mismatch

### NEW Behavior (After):
**Test-Only Mode:**
- Uses the FINAL model from train-test mode (trained on [1-MAX])
- Tests this model on different intervals: [1-7h], [1-10h], ..., [1-MAX]
- **Result:** At x=MAX, blue and orange lines are now IDENTICAL! ✓

---

## What This Means

### Left Panel (Train-Test):
```
x=7h:  Train on [1-7h],  Test on [1-7h]  → Model optimized for early infection
x=10h: Train on [1-10h], Test on [1-10h] → Model optimized for 10h window
x=46h: Train on [1-46h], Test on [1-46h] → Model optimized for full window
```

**Question:** How well does a model perform when trained and tested on the SAME limited data?

### Right Panel (Test-Only):
```
x=7h:  Train on [1-46h], Test on [1-7h]  → Best model tested on limited data
x=10h: Train on [1-46h], Test on [1-10h] → Best model tested on limited data
x=46h: Train on [1-46h], Test on [1-46h] → Best model tested on full data ← SAME AS LEFT!
```

**Question:** How well does the BEST model (trained on full data) perform when tested on limited data?

---

## At x=MAX (Final Point)

**Both panels now use:**
- Same model: Trained on [1-MAX]
- Same test data: [1-MAX]
- **Same performance** ✓

This makes logical sense!

---

## What The Comparison Shows

**Gap between blue (left) and orange (right) lines reveals:**

- **At early timepoints (x < MAX):**
  - Left (blue): Lower performance because model trained on LIMITED data
  - Right (orange): Higher performance because model trained on FULL data
  - **Gap = Benefit of having more training data**

- **At final timepoint (x = MAX):**
  - Left (blue): Performance with full data
  - Right (orange): Performance with full data (SAME model!)
  - **Gap = 0** ✓

---

## Example Interpretation

If you see:
- x=10h: Blue AUC = 0.95, Orange AUC = 0.98
- x=46h: Blue AUC = 0.99, Orange AUC = 0.99

**This means:**
1. Training on [1-10h] → 0.95 AUC on [1-10h] test
2. Training on [1-46h] → 0.98 AUC on [1-10h] test
3. **Conclusion:** Having more training data (up to 46h) improves performance even when testing on early timepoints (10h) by 3%!

---

## Implementation Details

### Code Changes:

1. **Test-only mode now reuses train-test checkpoints:**
   ```python
   if mode == "test-only":
       max_hour = max(hours)
       # Use final train-test checkpoint instead of training new model
       test_only_checkpoint_dir = output_dir / "checkpoints" / f"train-test_interval_{int(args.start_hour)}-{int(max_hour)}"
   ```

2. **Plot labels updated:**
   - Old: "Train=full, Test=[start,x]"
   - New: "Train=[start,MAX], Test=[start,x]"

3. **No redundant training:**
   - If running `--mode both`, train-test trains the [1-MAX] model
   - Test-only simply reuses that checkpoint for all test intervals

---

## Usage

### Normal run (trains both modes):
```bash
python analyze_interval_sweep_train.py \
    --upper-hours 7 10 13 16 19 22 25 28 31 34 37 40 43 46 \
    --mode both \
    --match-uninfected-window
```

**What happens:**
1. Train-test: Trains 14 models (one per interval) × 5 folds = 70 models
2. Test-only: Reuses the final [1-46h] model from step 1, tests it on all 14 intervals
3. Final point (x=46h): Both modes show IDENTICAL performance

### Eval-only mode (reuses existing checkpoints):
```bash
python analyze_interval_sweep_train.py \
    --upper-hours 7 10 13 16 19 22 25 28 31 34 37 40 43 46 \
    --mode both \
    --eval-only \
    --checkpoint-dir outputs/interval_sweep_analysis/20251210-170101
```

**What happens:**
1. Train-test: Loads checkpoints from existing directory
2. Test-only: Loads the final [1-46h] checkpoint from existing directory
3. Both modes just run evaluation + visualization

---

## Benefits

1. ✅ **Logical consistency:** Final point converges to same value
2. ✅ **Meaningful comparison:** Gap shows benefit of more training data
3. ✅ **No redundant training:** Reuses checkpoints efficiently
4. ✅ **Clear interpretation:** Easy to explain what each panel means

---

## Summary

**Old design:** Test-only used different uninfected distribution → confusing results at x=MAX  
**New design:** Test-only reuses final train-test model → clear, interpretable comparison

The new design makes the interval sweep experiment much more intuitive and scientifically sound!
