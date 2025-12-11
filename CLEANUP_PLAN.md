# Repository Cleanup Plan

## Current Status
Your repository is fairly well organized, but here are some suggestions to make it even cleaner.

---

## 📁 Proposed Directory Structure

```
cell_classification/
├── README.md                          # Main documentation ✅ KEEP
├── requirements.txt                   # Dependencies ✅ KEEP
├── .gitignore                         # Git ignore rules ✅ KEEP
│
├── docs/                              # 📝 NEW: Consolidate documentation
│   ├── ANALYSIS_UPDATES.md           # Move here
│   ├── MODE_CLARIFICATION.md         # Move here
│   ├── MODE_PARAMETER_UPDATE.md      # Move here
│   ├── INTERVAL_SWEEP_GUIDE.md       # Move here
│   └── EXPERIMENTS_README.md         # Move from shells/
│
├── scripts/                           # 🔧 Core training & analysis scripts
│   ├── train.py                       # ✅ KEEP
│   ├── test_folds.py                  # ✅ KEEP
│   ├── analyze_sliding_window_train.py    # ✅ KEEP
│   ├── analyze_interval_sweep_train.py    # ✅ KEEP
│   ├── visualize_sliding_window.py        # ✅ KEEP
│   ├── visualize_cam.py                   # ✅ KEEP
│   └── load_checkpoint_example.py         # ✅ KEEP
│
├── scripts/legacy/                    # 📦 OLD: Evaluation-only scripts (deprecated)
│   ├── analyze_sliding_window.py      # MOVE HERE (old eval-only version)
│   └── analyze_interval_sweep.py      # MOVE HERE (old eval-only version)
│
├── shells/                            # 🐚 Bash scripts ✅ KEEP AS IS
│   ├── train_baseline.sh
│   ├── train_early.sh
│   ├── exp1_train_all_test_restricted.sh
│   ├── exp2_train_test_restricted.sh
│   ├── run_both_experiments.sh
│   ├── interval_sweep_comparison.sh
│   ├── analyze_sliding_window_train.sh
│   ├── analyze_sliding_window.sh
│   ├── CAM.sh
│   ├── draw_chart.sh
│   └── test.sh
│
├── configs/                           # ⚙️ Configuration files ✅ KEEP
│   ├── resnet50_baseline.yaml
│   ├── resnet50_early.yaml
│   └── resnet50_time_regression.yaml
│
├── models/                            # 🧠 Model definitions ✅ KEEP
│   ├── __init__.py
│   └── resnet.py
│
├── datasets/                          # 📊 Dataset classes ✅ KEEP
│   ├── __init__.py
│   └── timecourse_dataset.py
│
├── utils/                             # 🛠️ Utility functions ✅ KEEP
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   ├── seed.py
│   └── transforms.py
│
├── outputs/                           # 📈 Training outputs ✅ KEEP (in .gitignore)
├── checkpoints/                       # 💾 Model checkpoints ✅ KEEP (in .gitignore)
├── cam_outputs/                       # 🎨 CAM visualizations ✅ KEEP
├── trys/                              # 🧪 Experimental/scratch work
└── __pycache__/                       # 🗑️ Python cache (in .gitignore)
```

---

## 🔄 Proposed Actions

### 1. Create New Directories
```bash
mkdir docs
mkdir scripts
mkdir scripts/legacy
```

### 2. Move Documentation Files
```bash
# Consolidate all markdown docs into docs/
mv ANALYSIS_UPDATES.md docs/
mv MODE_CLARIFICATION.md docs/
mv MODE_PARAMETER_UPDATE.md docs/
mv INTERVAL_SWEEP_GUIDE.md docs/
mv shells/EXPERIMENTS_README.md docs/
```

### 3. Move Script Files
```bash
# Move core scripts to scripts/
mv train.py scripts/
mv test_folds.py scripts/
mv analyze_sliding_window_train.py scripts/
mv analyze_interval_sweep_train.py scripts/
mv visualize_sliding_window.py scripts/
mv visualize_cam.py scripts/
mv load_checkpoint_example.py scripts/

# Move deprecated evaluation-only scripts to legacy/
mv analyze_sliding_window.py scripts/legacy/
mv analyze_interval_sweep.py scripts/legacy/
```

### 4. Clean Up Backup Files
```bash
# Option 1: Delete backup (if you're confident)
rm README.md.backup

# Option 2: Move to archive
mkdir archive
mv README.md.backup archive/
```

### 5. Update .gitignore
Add these lines if not already present:
```
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
.DS_Store
.idea/
.vscode/
*.swp
*.swo
*~
```

### 6. What to Do with trys/ ?
**Question for you:** What's in `trys/`? 
- If it's old experiments → Keep as `experiments/` or `archive/`
- If it's junk → Delete
- If it's active testing → Rename to `dev/` or `sandbox/`

---

## 📋 After Cleanup, Update Import Paths

If you move scripts to `scripts/`, you'll need to update how you run them:

**Before:**
```bash
python analyze_sliding_window_train.py --config configs/resnet50_baseline.yaml
```

**After:**
```bash
python scripts/analyze_sliding_window_train.py --config configs/resnet50_baseline.yaml
```

**OR** add scripts to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts"
python -m analyze_sliding_window_train --config configs/resnet50_baseline.yaml
```

**OR** keep scripts in root (simpler, current approach works fine!)

---

## 🎯 Alternative: Minimal Cleanup (Recommended)

If you want to keep it simple and minimize disruption:

### Just do these:
1. **Create docs/ folder** and move all .md files there (except README.md)
2. **Delete or archive** README.md.backup
3. **Add a note** in trys/ explaining what it is
4. **Update .gitignore** to exclude __pycache__

```bash
# Minimal cleanup commands
mkdir docs
mv ANALYSIS_UPDATES.md MODE_CLARIFICATION.md MODE_PARAMETER_UPDATE.md INTERVAL_SWEEP_GUIDE.md docs/
mv shells/EXPERIMENTS_README.md docs/
rm README.md.backup  # or: mkdir archive && mv README.md.backup archive/
echo "# Experimental/Draft Code" > trys/README.md
```

---

## ✅ What to Keep As-Is

- `configs/` - well organized
- `models/` - clean structure
- `datasets/` - clean structure
- `utils/` - clean structure
- `shells/` - all scripts are actively used
- `outputs/`, `checkpoints/`, `cam_outputs/` - runtime outputs

---

## 🤔 Questions for You

1. **Do you want to move scripts to `scripts/` folder?** 
   - Pros: Cleaner root directory
   - Cons: Need to update all shell scripts and paths

2. **What should I do with `trys/`?**
   - Keep as-is?
   - Rename to something clearer?
   - Archive or delete?

3. **Keep deprecated scripts?**
   - `analyze_sliding_window.py` (old eval-only version)
   - `analyze_interval_sweep.py` (old eval-only version)
   - Option: Move to `scripts/legacy/` or delete?

4. **README.md.backup** - Delete or archive?

---

## 🚀 My Recommendation

**Start with minimal cleanup:**
```bash
cd CODE/cell_classification

# 1. Create docs folder and consolidate documentation
mkdir -p docs
mv ANALYSIS_UPDATES.md MODE_CLARIFICATION.md MODE_PARAMETER_UPDATE.md INTERVAL_SWEEP_GUIDE.md docs/
mv shells/EXPERIMENTS_README.md docs/

# 2. Handle backup
rm README.md.backup  # or move to archive/

# 3. Document trys/
echo "# Experimental/Draft Code\n\nTemporary testing and draft implementations." > trys/README.md

# 4. Update .gitignore if needed
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore

# 5. Create legacy folder for deprecated scripts
mkdir -p scripts/legacy
mv analyze_sliding_window.py scripts/legacy/
mv analyze_interval_sweep.py scripts/legacy/
echo "# Legacy Scripts\n\nDeprecated evaluation-only versions. Use *_train.py versions instead." > scripts/legacy/README.md
```

This keeps everything functional while making it cleaner!

---

**Let me know which approach you prefer and I'll execute it!**
