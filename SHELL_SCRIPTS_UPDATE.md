# ✅ Shell Scripts Updated - Match Uninfected Window

## Summary

All interval sweep shell scripts have been updated to support the new `--match-uninfected-window` feature!

---

## 📝 Updated Files

### 1. `shells/analyze_sliding_window_train.sh`
✅ Added `MATCH_UNINFECTED=false` variable  
✅ Conditional flag passing to Python script  
✅ Updated help text

### 2. `shells/exp1_train_all_test_restricted.sh`
✅ Added `MATCH_UNINFECTED=false` variable  
✅ Conditional flag passing to Python script  
✅ Shows setting in output

### 3. `shells/exp2_train_test_restricted.sh`
✅ Added `MATCH_UNINFECTED=false` variable  
✅ Conditional flag passing to Python script  
✅ Shows setting in output

### 4. `shells/run_both_experiments.sh`
✅ Added `MATCH_UNINFECTED=false` variable  
✅ Conditional flag passing to Python script  
✅ Shows setting in output

---

## 🎯 How It Works

Each script now has this pattern:

```bash
# Configuration section
MATCH_UNINFECTED=false  # <-- SET THIS TO true TO ENABLE

# Build command
CMD="python analyze_..._train.py \
    --config $CONFIG \
    --other-flags ..."

# Add flag if enabled
if [ "$MATCH_UNINFECTED" = true ]; then
    CMD="$CMD --match-uninfected-window"
fi

# Execute
eval $CMD
```

---

## 🚀 Usage Examples

### Example 1: Quick Toggle
```bash
# Edit the script
nano shells/run_both_experiments.sh

# Find this line:
MATCH_UNINFECTED=false

# Change to:
MATCH_UNINFECTED=true

# Save and run:
bash shells/run_both_experiments.sh
```

### Example 2: All Scripts
```bash
# Enable in all interval sweep scripts
sed -i 's/MATCH_UNINFECTED=false/MATCH_UNINFECTED=true/g' shells/exp*.sh
sed -i 's/MATCH_UNINFECTED=false/MATCH_UNINFECTED=true/g' shells/run_both_experiments.sh

# Run experiments
bash shells/run_both_experiments.sh
```

### Example 3: One-time Override
You can also run with the flag directly:
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 \
    --k-folds 5 \
    --epochs 10 \
    --match-uninfected-window
```

---

## 📊 What You'll See

When you run a script with `MATCH_UNINFECTED=true`, the output will show:

```
Settings:
  Intervals: [1, x] where x = 8 10 12 14 16 18 20 22 24 26 28 30
  K-folds: 5
  Epochs: 10
  Metrics: auc accuracy f1
  Match uninfected interval: true  ← YOU'LL SEE THIS
==================================================

Device: cuda
K-folds: 5
✓ Match uninfected window: ENABLED (infected and uninfected use same time window)  ← AND THIS
```

---

## ⚙️ Default Behavior

**By default, all scripts have `MATCH_UNINFECTED=false`**

This means:
- ✅ Backward compatible - works like before
- ✅ No breaking changes to existing workflows
- ✅ Opt-in feature - you choose when to enable it

---

## 🎓 Recommendation

**For new experiments, set `MATCH_UNINFECTED=true`**

This gives you:
- Fair temporal comparison
- Scientific rigor
- Publication-quality results
- No data leakage

**For reproducing old results, keep `MATCH_UNINFECTED=false`**

This ensures:
- Consistency with previous experiments
- Direct comparison with older runs

---

## ✅ Verification Checklist

- [x] `analyze_sliding_window_train.sh` - Updated
- [x] `exp1_train_all_test_restricted.sh` - Updated
- [x] `exp2_train_test_restricted.sh` - Updated  
- [x] `run_both_experiments.sh` - Updated
- [x] Python scripts support `--match-uninfected-window` flag
- [x] Backward compatible (default = false)
- [x] Documentation created

---

**All shell scripts are ready to use! 🎉**

Set `MATCH_UNINFECTED=true` in any script to enable strict temporal matching.

---

**Updated:** December 12, 2025  
**Status:** ✅ Complete
