#!/bin/bash

##############################################################################
# Experiment 1: Train on ALL data, test on restricted intervals
#
# For each interval [1, x]:
#   TRAIN: infected [1, FULL] + all uninfected
#   TEST:  infected [1, x] + all uninfected
#
# This shows: "How well can a model trained on ALL data perform when 
#              testing on early time windows?"
##############################################################################

echo "=================================================="
echo "Experiment 1: Train on ALL, Test on [1, x]"
echo "=================================================="

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
UPPER_HOURS=(8 10 12 14 16 18 20 22 24 26 28 30)
START_HOUR=1
K_FOLDS=5
EPOCHS=10
METRICS="auc accuracy f1"

echo "Training setup:"
echo "  Infected frames:   [1, FULL] (all available)"
echo "  Uninfected frames: [0, FULL] (all available)"
echo ""
echo "Testing intervals: [1, x] where x = ${UPPER_HOURS[@]}"
echo "K-folds: $K_FOLDS"
echo "Epochs per interval: $EPOCHS"
echo "Metrics: $METRICS"
echo "=================================================="
echo ""

# Run the analysis (ONLY experiment 1)
python analyze_interval_sweep_train.py \
    --config "$CONFIG" \
    --upper-hours ${UPPER_HOURS[@]} \
    --start-hour $START_HOUR \
    --metrics $METRICS \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS \
    --split test \
    --mode test-only

echo ""
echo "=================================================="
echo "Experiment 1 Complete!"
echo "=================================================="
echo ""
echo "Results location: outputs/interval_sweep_analysis/<timestamp>/"
echo ""
echo "Output files:"
echo "  - interval_sweep_auc.png (single panel - test-only mode)"
echo "  - interval_sweep_accuracy.png"
echo "  - interval_sweep_f1.png"
echo "  - interval_sweep_data.json"
echo ""
echo "Checkpoints saved in:"
echo "  checkpoints/test-only_interval_*/"
echo "=================================================="
