# Sliding Window Analysis Script (PowerShell)
# Analyzes model performance using sliding time windows [x, x+k]
# where both training and testing data come from the same interval

# Example usage:
# .\shells\analyze_sliding_window.ps1

# Configuration
$CONFIG = "configs/resnet50_baseline.yaml"
$RUN_DIR = "checkpoints/resnet50_baseline/20251208-162511"  # Update this to your actual run directory
$WINDOW_SIZE = 5  # Window size in hours (k)
$WINDOW_STARTS = "0 2 4 6 8 10 12 14 16 18 20 22 24"  # Starting hours (x values)
$SPLIT = "test"
$METRIC = "auc"
$MAX_HOUR = 30  # Optional: maximum hour to consider

# Run sliding window analysis with k=5
Write-Host "Running sliding window analysis with window size = $WINDOW_SIZE hours..." -ForegroundColor Cyan
python analyze_sliding_window.py `
    --config $CONFIG `
    --run-dir $RUN_DIR `
    --window-size $WINDOW_SIZE `
    --window-starts $WINDOW_STARTS `
    --split $SPLIT `
    --metric $METRIC `
    --max-hour $MAX_HOUR

Write-Host ""
Write-Host "Analysis complete! Check the following locations:" -ForegroundColor Green
Write-Host "  - Plot: $RUN_DIR/analysis/sliding_window_k${WINDOW_SIZE}_${METRIC}.png"
Write-Host "  - Data: $RUN_DIR/analysis/sliding_window_k${WINDOW_SIZE}_${METRIC}.json"

Write-Host ""
Write-Host "To try different window sizes, run:" -ForegroundColor Yellow
Write-Host "  python analyze_sliding_window.py --config $CONFIG --run-dir $RUN_DIR --window-size 10 --window-starts 0 5 10 15 20 --split $SPLIT --metric $METRIC"
