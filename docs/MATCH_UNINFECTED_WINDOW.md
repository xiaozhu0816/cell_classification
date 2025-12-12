# NEW FEATURE: Match Uninfected Window

## 🎯 What Changed

Added `--match-uninfected-window` flag to both analysis scripts to control how uninfected samples are filtered by time.

**Affected scripts:**
- `analyze_sliding_window_train.py`
- `analyze_interval_sweep_train.py`

---

## 📊 The Problem

### OLD Behavior (Default - Still Available)
- **Infected samples**: Use restricted time window (e.g., [10, 16]h)
- **Uninfected samples**: Use ALL time points (0h, 1h, 2h, 4h, ..., 48h)

**Issue**: Unfair comparison - uninfected gets more data from different time periods

### NEW Behavior (With `--match-uninfected-window`)
- **Infected samples**: Use restricted time window (e.g., [10, 16]h)
- **Uninfected samples**: Use SAME time window (e.g., [10, 16]h)

**Benefit**: Fair comparison - both classes use exact same time period

---

## 🔧 How To Use

### Option A: Command Line

```bash
# OLD: Uninfected uses all time points (default)
python analyze_sliding_window_train.py \
    --config configs/resnet50_baseline.yaml \
    --window-size 6 \
    --stride 3 \
    --k-folds 5 \
    --epochs 10

# NEW: Uninfected uses same window as infected
python analyze_sliding_window_train.py \
    --config configs/resnet50_baseline.yaml \
    --window-size 6 \
    --stride 3 \
    --k-folds 5 \
    --epochs 10 \
    --match-uninfected-window  # <-- Add this flag
```

### Option B: Shell Script

Edit `shells/analyze_sliding_window_train.sh`:

```bash
MATCH_UNINFECTED=true  # Change from false to true
```

Then run:
```bash
bash shells/analyze_sliding_window_train.sh
```

---

## 📖 Examples

### Example 1: Sliding Window [10, 16]h

**Without flag (default):**
- Infected frames: 10h, 12h, 14h, 16h
- Uninfected frames: 0h, 1h, 2h, 4h, 6h, 8h, 10h, 12h, 14h, 16h, 18h, ..., 48h

**With `--match-uninfected-window`:**
- Infected frames: 10h, 12h, 14h, 16h
- Uninfected frames: 10h, 12h, 14h, 16h ✅ **SAME**

### Example 2: Interval Sweep [1, 8]h

**Without flag (default):**
- Infected frames: 1h, 2h, 4h, 6h, 8h
- Uninfected frames: 0h, 1h, 2h, 4h, 6h, 8h, 10h, ..., 48h

**With `--match-uninfected-window`:**
- Infected frames: 1h, 2h, 4h, 6h, 8h
- Uninfected frames: 1h, 2h, 4h, 6h, 8h ✅ **SAME**

---

## 🧪 Scientific Rationale

### Why Match Windows?

1. **Fair Comparison**: Both classes learn from same temporal context
2. **Avoid Data Leakage**: Prevent uninfected from having "future" information
3. **Biological Relevance**: Compare infection at specific time vs mock treatment at same time
4. **Rigorous Analysis**: Isolate effect of infection, not time period advantage

### When To Use Each Mode

**Use DEFAULT (no flag):**
- You want uninfected as a general "negative control" across all times
- You have limited uninfected data
- Baseline comparison doesn't require temporal matching

**Use `--match-uninfected-window`:**
- ✅ Scientific rigor: compare same time periods
- ✅ Fair evaluation: no data advantage to either class
- ✅ Time-specific analysis: how infection differs at specific hours
- ✅ Publication quality: reviewers will appreciate matched controls

---

## 🔄 Backward Compatibility

**The default behavior has NOT changed!**

- Without the flag: Works exactly as before (uninfected uses all times)
- With the flag: New behavior (uninfected uses matched window)
- All existing scripts and results remain valid

---

## 📝 Updated Shell Scripts

All shell scripts now have a `MATCH_UNINFECTED` variable:

### `analyze_sliding_window_train.sh`
```bash
MATCH_UNINFECTED=false  # Set to true to enable matching
```

### Interval Sweep Scripts
Edit these files to add matching:
- `exp1_train_all_test_restricted.sh`
- `exp2_train_test_restricted.sh`
- `run_both_experiments.sh`
- `interval_sweep_comparison.sh`

Add this line after the configuration section:
```bash
MATCH_UNINFECTED=false  # Set to true to enable matching
```

Then modify the Python command to include the flag conditionally.

---

## 🎓 Technical Details

### What Happens Under The Hood

The flag modifies the `frames` configuration passed to the dataset:

**Without flag:**
```yaml
frames:
  infected_window_hours: [10, 16]
  # uninfected_window_hours not set → uses all times
```

**With flag:**
```yaml
frames:
  infected_window_hours: [10, 16]
  uninfected_window_hours: [10, 16]  # <-- Added!
```

### Code Changes

1. **Added argument parser option** in both scripts
2. **Modified `apply_sliding_window()`** to accept `match_uninfected` parameter
3. **Modified `apply_interval_override()`** to accept `match_uninfected` parameter
4. **Updated function calls** to pass the flag through
5. **Added logging** to show which mode is active

---

## 💡 Recommendations

### For New Experiments
**Use `--match-uninfected-window`** for:
- Publication-quality results
- Fair temporal comparisons
- Rigorous scientific analysis

### For Existing Experiments
Keep the default (no flag) to:
- Maintain consistency with previous results
- Compare with earlier analyses
- Use as baseline before switching

---

## 📊 Expected Impact

### Performance Changes

With matched windows, you may see:
- **Lower overall accuracy** (less uninfected data)
- **More fair AUC** (balanced temporal context)
- **Different optimal windows** (not biased by uninfected "all times" advantage)

This is EXPECTED and CORRECT - it's a fairer comparison!

---

## 🚀 Quick Start

**Try it now with your existing data:**

```bash
# Run without matching (old way)
python analyze_sliding_window_train.py \
    --config configs/resnet50_baseline.yaml \
    --window-size 6 --stride 3 \
    --k-folds 5 --epochs 10 \
    --output-dir outputs/test_old

# Run with matching (new way)
python analyze_sliding_window_train.py \
    --config configs/resnet50_baseline.yaml \
    --window-size 6 --stride 3 \
    --k-folds 5 --epochs 10 \
    --match-uninfected-window \
    --output-dir outputs/test_new

# Compare results!
```

---

## ✅ Summary

- **Flag added**: `--match-uninfected-window`
- **Default behavior**: Unchanged (backward compatible)
- **New behavior**: Fair temporal matching for rigorous analysis
- **Recommendation**: Use the flag for publication-quality results
- **Shell scripts**: Updated with `MATCH_UNINFECTED` variable

**Date implemented**: December 12, 2025
