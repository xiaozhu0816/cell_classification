# Analysis Scripts Update Summary

## Overview

This document summarizes the new features added to `analyze_sliding_window.py` and `analyze_interval_sweep.py` to support multi-metric visualization and configurable window overlap.

## Changes Made

### 1. Multi-Metric Support

Both analysis scripts now support evaluating and plotting multiple metrics simultaneously.

#### Features:
- **Combined plots**: When multiple metrics are specified, a combined plot is generated showing all metrics on the same chart for easy comparison
- **Individual plots**: Each metric also gets its own dedicated plot
- **Backward compatibility**: Single metric mode still works using `--metric` flag

#### Usage:

```powershell
# Multiple metrics
python analyze_sliding_window.py --metrics auc accuracy f1 precision recall ...
python analyze_interval_sweep.py --metrics auc accuracy f1 precision recall ...

# Single metric (backward compatible)
python analyze_sliding_window.py --metric auc ...
python analyze_interval_sweep.py --metric auc ...
```

#### Outputs:
- `*_combined.png`: Combined plot with all metrics (only when multiple metrics specified)
- `*_<metric>.png`: Individual plot for each metric
- `*_data.json`: JSON file containing results for all metrics

### 2. Sliding Window Overlap Control (analyze_sliding_window.py)

The sliding window script now supports configurable overlap between consecutive windows through a `--stride` parameter.

#### Features:
- **Overlapping windows**: `stride < window-size` creates windows that overlap
- **Adjacent windows**: `stride = window-size` creates touching windows with no gap or overlap
- **Gapped windows**: `stride > window-size` creates windows with gaps between them
- **Auto-generation**: Windows can be automatically generated based on start/end hours and stride
- **Manual specification**: Windows can still be manually specified via `--window-starts`

#### Parameters:

```powershell
--window-size    # Size of each window (k hours)
--stride         # Step size between windows (default: window-size)
--start-hour     # First window starts here (used with auto-generation)
--end-hour       # Last window ends before this (used with auto-generation)
--window-starts  # Manual list of window start positions (alternative)
```

#### Examples:

```powershell
# Overlapping 5-hour windows with 2-hour stride
python analyze_sliding_window.py \
    --window-size 5 \
    --stride 2 \
    --start-hour 0 \
    --end-hour 30 \
    --metrics auc accuracy

# Adjacent 10-hour windows (no overlap)
python analyze_sliding_window.py \
    --window-size 10 \
    --stride 10 \
    --start-hour 0 \
    --end-hour 30 \
    --metrics auc

# Manual window positions
python analyze_sliding_window.py \
    --window-size 5 \
    --window-starts 0 5 10 15 20 25 \
    --metrics auc f1
```

## File Modifications

### Modified Files:

1. **`analyze_sliding_window.py`**
   - Added `--stride` parameter for overlap control
   - Added `--start-hour` and `--end-hour` for auto-generation
   - Made `--window-starts` optional (auto-generated if not provided)
   - Added `--metrics` parameter for multiple metrics
   - Updated `evaluate_window()` to return results for all metrics
   - Created `plot_single_metric()` for individual metric plots
   - Created `plot_multi_metric()` for combined metric plots
   - Enhanced output naming with stride suffix

2. **`analyze_interval_sweep.py`**
   - Added `--metrics` parameter for multiple metrics
   - Updated `evaluate_interval()` to return results for all metrics
   - Created `plot_single_metric_sweep()` for individual metric plots
   - Created `plot_multi_metric_sweep()` for combined metric plots
   - Enhanced result storage structure to support multiple metrics

3. **`shells/analyze_sliding_window.sh`**
   - Updated to demonstrate stride usage
   - Added examples for overlapping windows
   - Added examples for multiple metrics

4. **`shells/analyze_sliding_window.ps1`**
   - Updated to demonstrate stride usage
   - Added examples for overlapping windows
   - Added examples for multiple metrics

5. **`README.md`**
   - Updated "Interval sweep error bars" section with multi-metric examples
   - Updated "Sliding window analysis" section with stride and multi-metric examples
   - Added detailed parameter descriptions
   - Added use case examples

## Usage Examples

### Example 1: Overlapping Windows with Multiple Metrics

```powershell
python analyze_sliding_window.py \
    --config configs/resnet50_baseline.yaml \
    --run-dir checkpoints/resnet50_baseline/20251208-162511 \
    --window-size 5 \
    --stride 2 \
    --start-hour 0 \
    --end-hour 30 \
    --metrics auc accuracy f1 precision recall \
    --split test
```

**Result**: Windows [0,5], [2,7], [4,9], [6,11], ... [25,30] evaluated with all 5 metrics, producing:
- `sliding_window_w5_s2_combined.png` (all metrics together)
- `sliding_window_w5_s2_auc.png`
- `sliding_window_w5_s2_accuracy.png`
- `sliding_window_w5_s2_f1.png`
- `sliding_window_w5_s2_precision.png`
- `sliding_window_w5_s2_recall.png`
- `sliding_window_w5_s2_data.json` (all metrics data)

### Example 2: Interval Sweep with Multiple Metrics

```powershell
python analyze_interval_sweep.py \
    --config configs/resnet50_baseline.yaml \
    --run-dir checkpoints/resnet50_baseline/20251208-162511 \
    --upper-hours 6 8 10 12 14 16 18 20 \
    --start-hour 1 \
    --metrics auc accuracy f1 \
    --split test
```

**Result**: Intervals [1,6], [1,8], [1,10], ... [1,20] evaluated with 3 metrics, producing:
- `interval_sweep_combined.png` (all metrics, two panels)
- `interval_sweep_auc.png` (two panels)
- `interval_sweep_accuracy.png` (two panels)
- `interval_sweep_f1.png` (two panels)
- `interval_sweep_data.json` (all metrics data)

## Benefits

### 1. Comprehensive Metric Comparison
- View multiple metrics side-by-side to understand trade-offs
- Identify which metrics are most sensitive to time windows
- Compare early-detection performance across different evaluation criteria

### 2. Flexible Window Overlap
- Find optimal overlap for continuous monitoring systems
- Balance computational cost vs. temporal resolution
- Identify redundancy in adjacent time windows

### 3. Efficient Experimentation
- Single script run generates all metric plots
- Reduced computation time (evaluate once, plot multiple metrics)
- Easier to share comprehensive results with collaborators

## Backward Compatibility

All changes maintain backward compatibility:
- Single metric mode works with existing `--metric` flag
- Default stride equals window size (no overlap, original behavior)
- Manual `--window-starts` still supported
- Existing shell scripts will continue to work (though updated versions are provided)

## Next Steps

Users can now:
1. Experiment with different overlap strategies to optimize early detection
2. Compare multiple performance metrics across time windows simultaneously
3. Generate comprehensive reports with single commands
4. Identify which time windows and metrics are most reliable for infection classification
