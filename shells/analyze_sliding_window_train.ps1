# Sliding Window Training Analysis Script (PowerShell)
# Trains separate models for each time window [x, x+k]
# Shows which time periods are most informative for classification

# Configuration
$CONFIG = "configs/resnet50_baseline.yaml"
$WINDOW_SIZE = 5  # Window size in hours (k)
$START_HOUR = 1  # Starting hour for first window
$END_HOUR = 30  # Maximum ending hour
$STRIDE = 5  # Step size between windows
               # stride < window_size creates overlap
               # stride = window_size creates adjacent windows (no gap, no overlap)
               # stride > window_size creates gaps between windows
$SPLIT = "test"  # Evaluation split (val or test)
$METRICS = "auc accuracy f1"  # Multiple metrics for analysis
$K_FOLDS = 5  # Number of cross-validation folds
$EPOCHS = 10  # Training epochs per window

# Run sliding window training analysis
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Sliding Window Training Analysis" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Window size: $WINDOW_SIZE hours"
Write-Host "Stride: $STRIDE hours"
Write-Host "Range: [$START_HOUR, $END_HOUR] hours"
Write-Host "K-folds: $K_FOLDS"
Write-Host "Epochs per window: $EPOCHS"
Write-Host "Metrics: $METRICS"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

python analyze_sliding_window_train.py `
    --config $CONFIG `
    --window-size $WINDOW_SIZE `
    --start-hour $START_HOUR `
    --end-hour $END_HOUR `
    --stride $STRIDE `
    --split $SPLIT `
    --metrics $METRICS `
    --k-folds $K_FOLDS `
    --epochs $EPOCHS

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host "Check outputs/sliding_window_analysis/<timestamp>/" -ForegroundColor Green
Write-Host "  - Combined plot: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_combined.png"
Write-Host "  - Individual plots: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_<metric>.png"
Write-Host "  - Data: sliding_window_w${WINDOW_SIZE}_s${STRIDE}_data.json"
Write-Host "  - Log: sliding_window_train.log"
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Alternative usage examples:" -ForegroundColor Yellow
Write-Host "  # Narrow 3-hour windows with overlap:"
Write-Host "  Edit this script: Set WINDOW_SIZE=3, STRIDE=1"
Write-Host ""
Write-Host "  # Manual window positions:"
Write-Host "  python analyze_sliding_window_train.py --window-size 5 --window-starts 0 5 10 15 20 25 --metrics auc f1 --k-folds 5 --epochs 10"
