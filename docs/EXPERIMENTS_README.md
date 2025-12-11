# Understanding the Experiment Scripts

## Important: One Command Runs BOTH Experiments!

The Python script `analyze_interval_sweep_train.py` is designed to run **BOTH experiments automatically** in a single execution. This is by design for efficiency and easy comparison.

---

## Available Bash Scripts

### Option 1: Run Both (Recommended) ⭐
```bash
bash shells/run_both_experiments.sh
```
- Most efficient
- Runs both experiments in one go
- Creates side-by-side comparison plots
- **This is what you want most of the time**

### Option 2: Individual Experiment Scripts
```bash
bash shells/exp1_train_all_test_restricted.sh
bash shells/exp2_train_test_restricted.sh
```
- These are **conceptually separate** but run the same Python command
- Both will execute BOTH experiments (it's how the Python script works)
- The only difference is the documentation/comments about which panel to look at

---

## Why Does One Command Run Both?

The Python script was designed this way because:

1. **Efficiency:** Shares data loading, config parsing, fold splitting
2. **Fair Comparison:** Same data splits, same randomization, same everything
3. **Visualization:** Creates perfect side-by-side comparison plots
4. **Convenience:** One run gives you both answers

---

## What Each Script Actually Does

All three bash scripts run this **same Python command**:

```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 22 24 26 28 30 \
    --start-hour 1 \
    --metrics auc accuracy f1 \
    --k-folds 5 \
    --epochs 10 \
    --split test
```

The Python script automatically:
1. Runs **"test-only" mode** for all intervals (Experiment 1)
2. Runs **"train-test" mode** for all intervals (Experiment 2)  
3. Creates two-panel plots comparing them

---

## If You Really Want Separate Runs

If you want to run experiments separately (e.g., to save time or debug), you would need to modify the Python script to accept a `--mode` parameter. Currently, it always runs both.

### To modify for separate runs:

You could add this to `analyze_interval_sweep_train.py`:

```python
parser.add_argument(
    "--mode",
    type=str,
    default="both",
    choices=["test-only", "train-test", "both"],
    help="Which mode to run"
)
```

Then in the main loop:
```python
if args.mode == "both":
    modes = ("test-only", "train-test")
elif args.mode == "test-only":
    modes = ("test-only",)
else:
    modes = ("train-test",)
```

**But this is not recommended** because you lose the side-by-side comparison!

---

## Recommended Workflow

1. **Run both experiments together:**
   ```bash
   bash shells/run_both_experiments.sh
   ```

2. **View the two-panel plots:**
   - Left panel = Experiment 1 (train on ALL)
   - Right panel = Experiment 2 (train on restricted)

3. **Extract specific results from JSON:**
   ```bash
   # Get Experiment 1 results only
   cat outputs/interval_sweep_analysis/<timestamp>/interval_sweep_data.json | \
       jq '.stats.auc."test-only"'
   
   # Get Experiment 2 results only
   cat outputs/interval_sweep_analysis/<timestamp>/interval_sweep_data.json | \
       jq '.stats.auc."train-test"'
   ```

4. **Use specific checkpoints:**
   - Experiment 1: `checkpoints/test-only_interval_*/`
   - Experiment 2: `checkpoints/train-test_interval_*/`

---

## Summary

✅ **Use:** `bash shells/run_both_experiments.sh`  
✅ **This runs BOTH experiments** in one efficient call  
✅ **Results separated** in plots (left vs right panel)  
✅ **Checkpoints separated** in different folders  
✅ **JSON data separated** by mode name  

The "two experiments" are really two modes of the same analysis, designed to be compared directly!
