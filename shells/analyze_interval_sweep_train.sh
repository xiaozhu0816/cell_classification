#!/bin/bash
# Interval Sweep Training Analysis Script
# Trains models with different infected interval ranges
# Shows how performance changes with varying time ranges

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
UPPER_HOURS="8 10 12 14 16 18 20"  # Upper bounds for [start, X] intervals
START_HOUR=1  # Lower bound for infected interval
SPLIT="test"  # Evaluation split (val or test)
METRICS="auc accuracy f1"  # Multiple metrics for analysis
K_FOLDS=5  # Number of cross-validation folds
EPOCHS=10  # Training epochs per interval

# Run interval sweep training analysis
echo "=================================================="
echo "Interval Sweep Training Analysis"
echo "=================================================="
echo "Intervals: [$START_HOUR, X] where X in {$UPPER_HOURS}"
echo "K-folds: $K_FOLDS"
echo "Epochs per interval: $EPOCHS"
echo "Metrics: $METRICS"
echo "=================================================="
echo ""

python analyze_interval_sweep_train.py \
    --config "$CONFIG" \
    --upper-hours $UPPER_HOURS \
    --start-hour $START_HOUR \
    --split "$SPLIT" \
    --metrics $METRICS \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS

echo ""
echo "=================================================="
echo "Analysis complete!"
echo "Check outputs/interval_sweep_analysis/<timestamp>/"
echo "  - Combined plot: interval_sweep_combined.png"
echo "  - Individual plots: interval_sweep_<metric>.png"
echo "  - Data: interval_sweep_data.json"
echo "  - Log: interval_sweep_train.log"
echo "=================================================="
echo ""
echo "Note: This script trains models in TWO modes:"
echo "  1. train-test: Both use [start, x]"
echo "  2. test-only: Train uses full range, test uses [start, x]"
echo "  Total runs = ${#UPPER_HOURS[@]} intervals × 2 modes × $K_FOLDS folds"
