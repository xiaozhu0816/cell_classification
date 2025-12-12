#!/bin/bash
# Sliding Window Training Analysis Script
# Trains separate models for each time window [x, x+k]
# Shows which time periods are most informative for classification

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
WINDOW_SIZE=6  # Window size in hours (k)
START_HOUR=1  # Starting hour for first window
END_HOUR=46  # Maximum ending hour
STRIDE=3  # Step size between windows
          # stride < window_size creates overlap
          # stride = window_size creates adjacent windows (no gap, no overlap)
          # stride > window_size creates gaps between windows
SPLIT="test"  # Evaluation split (val or test)
METRICS="auc accuracy f1"  # Multiple metrics for analysis
K_FOLDS=5  # Number of cross-validation folds
EPOCHS=10  # Training epochs per window
MATCH_UNINFECTED=true  # Set to true to apply same window to uninfected samples

# Run sliding window training analysis
echo "=================================================="
echo "Sliding Window Training Analysis"
echo "=================================================="
echo "Window size: $WINDOW_SIZE hours"
echo "Stride: $STRIDE hours"
echo "Range: [$START_HOUR, $END_HOUR] hours"
echo "K-folds: $K_FOLDS"
echo "Epochs per window: $EPOCHS"
echo "Metrics: $METRICS"
echo "Match uninfected window: $MATCH_UNINFECTED"
echo "=================================================="
echo ""

# Build command with optional flag
CMD="python analyze_sliding_window_train.py \
    --config $CONFIG \
    --window-size $WINDOW_SIZE \
    --start-hour $START_HOUR \
    --end-hour $END_HOUR \
    --stride $STRIDE \
    --split $SPLIT \
    --metrics $METRICS \
    --k-folds $K_FOLDS \
    --epochs $EPOCHS"

# Add flag if enabled
if [ "$MATCH_UNINFECTED" = true ]; then
    CMD="$CMD --match-uninfected-window"
fi

# Execute
eval $CMD

echo ""
echo "=================================================="
echo "Analysis complete!"
echo "Check outputs/sliding_window_analysis/<timestamp>/"
echo "  - Combined plot: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_combined.png"
echo "  - Individual plots: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_<metric>.png"
echo "  - Data: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_data.json"
echo "  - Log: sliding_window_train.log"
echo "=================================================="
echo ""
echo "Alternative usage examples:"
echo "  # Narrow 3-hour windows with overlap:"
echo "  bash shells/analyze_sliding_window_train.sh  # Edit WINDOW_SIZE=3, STRIDE=1"
echo ""
echo "  # Manual window positions:"
echo "  python analyze_sliding_window_train.py --window-size 5 --window-starts 0 5 10 15 20 25 --metrics auc f1 --k-folds 5 --epochs 10"
