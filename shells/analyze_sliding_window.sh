#!/bin/bash
# Sliding Window Analysis Script
# Analyzes model performance using sliding time windows [x, x+k]
# where both training and testing data come from the same interval

# Example usage:
# bash shells/analyze_sliding_window.sh

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
RUN_DIR="checkpoints/resnet50_baseline/20251208-162511"  # Update this to your actual run directory
WINDOW_SIZE=5  # Window size in hours (k)
WINDOW_STARTS="0 2 4 6 8 10 12 14 16 18 20 22 24"  # Starting hours (x values)
SPLIT="test"
METRIC="auc"
MAX_HOUR=30  # Optional: maximum hour to consider

# Run sliding window analysis
python analyze_sliding_window.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --window-size $WINDOW_SIZE \
    --window-starts $WINDOW_STARTS \
    --split "$SPLIT" \
    --metric "$METRIC" \
    --max-hour $MAX_HOUR

echo ""
echo "Analysis complete! Check the following locations:"
echo "  - Plot: $RUN_DIR/analysis/sliding_window_k${WINDOW_SIZE}_${METRIC}.png"
echo "  - Data: $RUN_DIR/analysis/sliding_window_k${WINDOW_SIZE}_${METRIC}.json"
