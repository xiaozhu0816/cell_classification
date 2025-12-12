#!/bin/bash

##############################################################################
# MASTER SCRIPT: Run Both Experiments
#
# This is a convenience wrapper. The actual Python script runs BOTH 
# experiments in a single call, which is more efficient.
#
# You can either:
#   1. Run this script (runs both experiments once)
#   2. Run exp1_train_all_test_restricted.sh (runs both, focus on left panel)
#   3. Run exp2_train_test_restricted.sh (runs both, focus on right panel)
#
# All three approaches run the SAME Python command. The Python script
# automatically runs both "test-only" and "train-test" modes and creates
# side-by-side comparison plots.
##############################################################################

echo "=================================================="
echo "Running BOTH Experiments (Most Efficient)"
echo "=================================================="
echo ""
echo "Experiment 1 (test-only):"
echo "  Train: infected [1, FULL] + all uninfected"
echo "  Test:  infected [1, x] + all uninfected"
echo "  → Results in LEFT panel of plots"
echo ""
echo "Experiment 2 (train-test):"
echo "  Train: infected [1, x] + all uninfected"
echo "  Test:  infected [1, x] + all uninfected"
echo "  → Results in RIGHT panel of plots"
echo ""
echo "=================================================="

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
UPPER_HOURS=(7 10 13 16 19 22 25 28 31 34 37 40 43 46)
START_HOUR=1
K_FOLDS=5
EPOCHS=10
METRICS="auc accuracy f1"
MATCH_UNINFECTED=true  # Set to true to apply same interval to uninfected samples

echo ""
echo "Settings:"
echo "  Intervals: [1, x] where x = ${UPPER_HOURS[@]}"
echo "  K-folds: $K_FOLDS"
echo "  Epochs: $EPOCHS"
echo "  Metrics: $METRICS"
echo "  Match uninfected interval: $MATCH_UNINFECTED"
echo "=================================================="
echo ""

# Build command with optional flag
CMD="python analyze_interval_sweep_train.py \
    --config $CONFIG \
    --upper-hours ${UPPER_HOURS[@]} \
    --start-hour $START_HOUR \
    --metrics $METRICS \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS \
    --split test \
    --mode both"

# Add flag if enabled
if [ "$MATCH_UNINFECTED" = true ]; then
    CMD="$CMD --match-uninfected-window"
fi

# Execute
eval $CMD

echo ""
echo "=================================================="
echo "BOTH Experiments Complete!"
echo "=================================================="
echo ""
echo "Output directory: outputs/interval_sweep_analysis/<timestamp>/"
echo ""
echo "Generated plots (two panels each):"
echo "  - interval_sweep_combined.png"
echo "    └─ Left:  Experiment 1 (test-only)"
echo "    └─ Right: Experiment 2 (train-test)"
echo ""
echo "  - interval_sweep_auc.png"
echo "  - interval_sweep_accuracy.png"
echo "  - interval_sweep_f1.png"
echo ""
echo "Model checkpoints:"
echo "  - checkpoints/test-only_interval_*/  (Experiment 1 models)"
echo "  - checkpoints/train-test_interval_*/ (Experiment 2 models)"
echo ""
echo "Data file:"
echo "  - interval_sweep_data.json (both experiments)"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Check the two-panel plots to compare experiments"
echo "  2. Left panel higher? → Train on ALL data is better"
echo "  3. Panels similar? → Can restrict training data safely"
echo "=================================================="
