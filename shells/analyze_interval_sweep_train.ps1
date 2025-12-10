# Interval Sweep Training Analysis Script (PowerShell)
# Trains models with different infected interval ranges
# Shows how performance changes with varying time ranges

# Configuration
$CONFIG = "configs/resnet50_baseline.yaml"
$UPPER_HOURS = "8 10 12 14 16 18 20"  # Upper bounds for [start, X] intervals
$START_HOUR = 1  # Lower bound for infected interval
$SPLIT = "test"  # Evaluation split (val or test)
$METRICS = "auc accuracy f1"  # Multiple metrics for analysis
$K_FOLDS = 5  # Number of cross-validation folds
$EPOCHS = 10  # Training epochs per interval

# Run interval sweep training analysis
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Interval Sweep Training Analysis" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Intervals: [$START_HOUR, X] where X in {$UPPER_HOURS}"
Write-Host "K-folds: $K_FOLDS"
Write-Host "Epochs per interval: $EPOCHS"
Write-Host "Metrics: $METRICS"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

python analyze_interval_sweep_train.py `
    --config $CONFIG `
    --upper-hours $UPPER_HOURS `
    --start-hour $START_HOUR `
    --split $SPLIT `
    --metrics $METRICS `
    --k-folds $K_FOLDS `
    --epochs $EPOCHS

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Analysis complete!" -ForegroundColor Green
Write-Host "Check outputs/interval_sweep_analysis/<timestamp>/" -ForegroundColor Green
Write-Host "  - Combined plot: interval_sweep_combined.png"
Write-Host "  - Individual plots: interval_sweep_<metric>.png"
Write-Host "  - Data: interval_sweep_data.json"
Write-Host "  - Log: interval_sweep_train.log"
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Note: This script trains models in TWO modes:" -ForegroundColor Yellow
Write-Host "  1. train-test: Both use [start, x]"
Write-Host "  2. test-only: Train uses full range, test uses [start, x]"
$hourCount = ($UPPER_HOURS -split ' ').Count
Write-Host "  Total runs = $hourCount intervals × 2 modes × $K_FOLDS folds"
