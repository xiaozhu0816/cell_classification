# Sliding Window Analysis Script (PowerShell)
# Analyzes model performance using sliding time windows [x, x+k]
# where both training and testing data come from the same interval

# Example usage:
# .\shells\analyze_sliding_window.ps1

# Configuration
$CONFIG = "configs/resnet50_baseline.yaml"
$RUN_DIR = "checkpoints/resnet50_baseline/20251208-162511"  # Update this to your actual run directory
$WINDOW_SIZE = 5  # Window size in hours (k)
$START_HOUR = 1  # Starting hour for first window
$END_HOUR = 30  # Maximum ending hour
$STRIDE = 2  # Step size between windows (default: window size for no overlap)
              # stride < window_size creates overlap
              # stride = window_size creates adjacent windows (no gap, no overlap)
              # stride > window_size creates gaps between windows
$SPLIT = "test"
$METRICS = "auc accuracy f1"  # Multiple metrics for combined plot + individual plots
                               # Or use single metric with $METRIC

# Example 1: Auto-generate windows with overlap (stride < window_size)
Write-Host "Running sliding window analysis with window size = $WINDOW_SIZE hours, stride = $STRIDE hours..." -ForegroundColor Cyan
python analyze_sliding_window.py `
    --config $CONFIG `
    --run-dir $RUN_DIR `
    --window-size $WINDOW_SIZE `
    --start-hour $START_HOUR `
    --end-hour $END_HOUR `
    --stride $STRIDE `
    --split $SPLIT `
    --metrics $METRICS

Write-Host ""
Write-Host "Analysis complete! Check the following locations:" -ForegroundColor Green
Write-Host "  - Combined plot: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_combined.png"
Write-Host "  - Individual plots: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_<metric>.png"
Write-Host "  - Data: $RUN_DIR/analysis/sliding_window_w${WINDOW_SIZE}_s${STRIDE}_data.json"

Write-Host ""
Write-Host "Alternative usage examples:" -ForegroundColor Yellow
Write-Host "  # No overlap (adjacent windows):"
Write-Host "  python analyze_sliding_window.py --config $CONFIG --run-dir $RUN_DIR --window-size 10 --stride 10 --start-hour 0 --end-hour 30 --metrics auc accuracy"
Write-Host ""
Write-Host "  # Manual window positions:"
Write-Host "  python analyze_sliding_window.py --config $CONFIG --run-dir $RUN_DIR --window-size 5 --window-starts 0 5 10 15 20 25 --metrics auc f1"
