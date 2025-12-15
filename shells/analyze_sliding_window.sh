#!/bin/bash
# Sliding Window Analysis Script
# Analyzes model performance using sliding time windows [x, x+k]
# where both training and testing data come from the same interval

# Example usage:
# bash shells/analyze_sliding_window.sh

# Configuration
CONFIG="configs/resnet50_baseline.yaml"
RUN_DIR="checkpoints/resnet50_baseline/20251211-163914"  # Update this to your actual run directory
WINDOW_SIZE=6  # Window size in hours (k)
START_HOUR=1  # Starting hour for first window
END_HOUR=46  # Maximum ending hour
STRIDE=3  # Step size between windows (default: window size for no overlap)
          # stride < window_size creates overlap
          # stride = window_size creates adjacent windows (no gap, no overlap)
          # stride > window_size creates gaps between windows
SPLIT="test"
METRICS="auc accuracy f1"  # Multiple metrics for combined plot + individual plots

# Run sliding window analysis with overlap
echo "Running sliding window analysis with window size = $WINDOW_SIZE hours, stride = $STRIDE hours..."
python scripts/legacy/analyze_sliding_window.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR" \
    --window-size $WINDOW_SIZE \
    --start-hour $START_HOUR \
    --end-hour $END_HOUR \
    --stride $STRIDE \
    --split "$SPLIT" \
    --metrics $METRICS

echo ""
echo "Analysis complete! Check the following locations:"
echo "  - Combined plot: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_combined.png"
echo "  - Individual plots: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_<metric>.png"
echo "  - Data: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_data.json"

echo ""
echo "Alternative usage examples:"
echo "  # No overlap (adjacent windows):"
echo "  python analyze_sliding_window.py --config \$CONFIG --run-dir \$RUN_DIR --window-size 10 --stride 10 --start-hour 0 --end-hour 30 --metrics auc accuracy"
echo ""
echo "  # Manual window positions:"
echo "  python analyze_sliding_window.py --config \$CONFIG --run-dir \$RUN_DIR --window-size 5 --window-starts 0 5 10 15 20 25 --metrics auc f1"
