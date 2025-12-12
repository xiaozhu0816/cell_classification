#!/bin/bash

##############################################################################
# Experiment 2: Train and test on same restricted intervals
#
# For each interval [1, x]:
#   TRAIN: infected [1, x] + all uninfected
#   TEST:  infected [1, x] + all uninfected
#
# This shows: "Should I train on the same restricted window that I'll 
#              encounter at deployment?"
##############################################################################

echo "=================================================="
echo "Experiment 2: Train on [1, x], Test on [1, x]"
echo "=================================================="

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
UPPER_HOURS=(8 10 12 14 16 18 20 22 24 26 28 30)
START_HOUR=1
K_FOLDS=5
EPOCHS=10
METRICS="auc accuracy f1"
MATCH_UNINFECTED=false  # Set to true to apply same interval to uninfected samples

echo "For each interval [1, x]:"
echo "  Training:"
echo "    Infected frames:   [1, x]"
echo "    Uninfected frames: [0, FULL]"
echo ""
echo "  Testing:"
echo "    Infected frames:   [1, x] (same as training)"
echo "    Uninfected frames: [0, FULL]"
echo ""
echo "Testing intervals: [1, x] where x = ${UPPER_HOURS[@]}"
echo "K-folds: $K_FOLDS"
echo "Epochs per interval: $EPOCHS"
echo "Metrics: $METRICS"
echo "Match uninfected interval: $MATCH_UNINFECTED"
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
    --mode train-test"

# Add flag if enabled
if [ "$MATCH_UNINFECTED" = true ]; then
    CMD="$CMD --match-uninfected-window"
fi

# Execute
eval $CMD

echo ""
echo "=================================================="
echo "Experiment 2 Complete!"
echo "=================================================="
echo ""
echo "Results location: outputs/interval_sweep_analysis/<timestamp>/"
echo ""
echo "Output files:"
echo "  - interval_sweep_auc.png (single panel - train-test mode)"
echo "  - interval_sweep_accuracy.png"
echo "  - interval_sweep_f1.png"
echo "  - interval_sweep_data.json"
echo ""
echo "Checkpoints saved in:"
echo "  checkpoints/train-test_interval_*/"
echo "=================================================="
