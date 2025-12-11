#!/bin/bash

##############################################################################
# Interval Sweep Training Analysis - Two-Mode Comparison
#
# BOTH MODES TRAIN FRESH MODELS! The mode names indicate which splits use 
# the restricted interval [1, x]:
#
# Mode 1 ("test-only"): 
#   - TRAIN on infected [1, FULL] + all uninfected
#   - TEST on infected [1, x] + all uninfected
#   → Only TEST split uses restricted interval
#
# Mode 2 ("train-test"):
#   - TRAIN on infected [1, x] + all uninfected
#   - TEST on infected [1, x] + all uninfected  
#   → Both TRAIN and TEST use restricted interval
#
# Question: Does training on more data help, even when testing on restricted windows?
##############################################################################

echo "=================================================="
echo "Interval Sweep Training Analysis - Two Modes"
echo "=================================================="

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
UPPER_HOURS=(8 10 12 14 16 18 20 22 24 26 28 30)  # Test intervals [1, x]
START_HOUR=1
K_FOLDS=5
EPOCHS=10
METRICS="auc accuracy f1"

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Upper hours to test: ${UPPER_HOURS[@]}"
echo "  Start hour: $START_HOUR"
echo "  K-folds: $K_FOLDS"
echo "  Epochs per interval: $EPOCHS"
echo "  Metrics: $METRICS"
echo "=================================================="
echo ""

# Run the analysis
python analyze_interval_sweep_train.py \
    --config "$CONFIG" \
    --upper-hours ${UPPER_HOURS[@]} \
    --start-hour $START_HOUR \
    --metrics $METRICS \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS \
    --split test

echo ""
echo "=================================================="
echo "Analysis complete!"
echo "Check outputs/interval_sweep_analysis/<timestamp>/"
echo ""
echo "Output files:"
echo "  - interval_sweep_combined.png   (all metrics, two panels)"
echo "  - interval_sweep_auc.png        (AUC comparison)"
echo "  - interval_sweep_accuracy.png   (Accuracy comparison)"
echo "  - interval_sweep_f1.png         (F1 comparison)"
echo "  - interval_sweep_data.json      (raw data)"
echo "  - checkpoints/                  (trained models)"
echo "=================================================="
echo ""
echo "Interpretation:"
echo "  BOTH MODES TRAIN MODELS (not evaluation-only!)"
echo ""
echo "  Left panel (test-only):"
echo "    Train: infected [1, FULL] + all uninfected"
echo "    Test:  infected [1, x] + all uninfected"
echo ""
echo "  Right panel (train-test):"
echo "    Train: infected [1, x] + all uninfected"
echo "    Test:  infected [1, x] + all uninfected"
echo ""
echo "If left panel > right panel:"
echo "  → Training on MORE data (all frames) helps performance"
echo "If left panel ≈ right panel:"
echo "  → Can restrict training data to match deployment scenario"
echo "=================================================="
