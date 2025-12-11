# NEW: --mode Parameter Added!

## What Changed

I added a `--mode` parameter to `analyze_interval_sweep_train.py` so you can now run experiments **separately** or **together**.

---

## The NEW Parameter

```bash
--mode {both,test-only,train-test}
```

### Options:

| Mode | What It Does | When To Use |
|------|-------------|-------------|
| `--mode both` | Runs BOTH experiments, creates two-panel comparison plots | **Default** - Best for comparing strategies |
| `--mode test-only` | Runs ONLY Experiment 1 (train on ALL, test on [1,x]) | When you only care about Exp 1 |
| `--mode train-test` | Runs ONLY Experiment 2 (train on [1,x], test on [1,x]) | When you only care about Exp 2 |

---

## Updated Bash Scripts

Now the three bash scripts are **actually different**:

### 1. Run BOTH Experiments (Recommended)
```bash
bash shells/run_both_experiments.sh
```
Uses: `--mode both`

**Output:**
- Two-panel plots comparing both experiments
- Checkpoints for both modes
- Faster than running separately (shared setup)

---

### 2. Run ONLY Experiment 1
```bash
bash shells/exp1_train_all_test_restricted.sh
```
Uses: `--mode test-only`

**What it does:**
- Train: infected [1, FULL] + all uninfected
- Test: infected [1, x] + all uninfected

**Output:**
- Single-panel plots (just Exp 1)
- Only test-only checkpoints
- Half the training time

---

### 3. Run ONLY Experiment 2
```bash
bash shells/exp2_train_test_restricted.sh
```
Uses: `--mode train-test`

**What it does:**
- Train: infected [1, x] + all uninfected
- Test: infected [1, x] + all uninfected

**Output:**
- Single-panel plots (just Exp 2)
- Only train-test checkpoints
- Half the training time

---

## Examples

### Run both for comparison:
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 \
    --mode both \
    --k-folds 5 \
    --epochs 10
```
**Training runs:** 7 intervals × 2 modes × 5 folds = **70 runs**

### Run only Experiment 1:
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 \
    --mode test-only \
    --k-folds 5 \
    --epochs 10
```
**Training runs:** 7 intervals × 1 mode × 5 folds = **35 runs** (50% faster!)

### Run only Experiment 2:
```bash
python analyze_interval_sweep_train.py \
    --config configs/resnet50_baseline.yaml \
    --upper-hours 8 10 12 14 16 18 20 \
    --mode train-test \
    --k-folds 5 \
    --epochs 10
```
**Training runs:** 7 intervals × 1 mode × 5 folds = **35 runs** (50% faster!)

---

## Answer to Your Question

**Q: Which params make them different for 2 exps?**

**A:** The `--mode` parameter!

- `--mode test-only` → Experiment 1 (train on ALL)
- `--mode train-test` → Experiment 2 (train on restricted)
- `--mode both` → Both experiments (default)

Before this change, there was **no parameter** - it always ran both. Now you have control! ✅
