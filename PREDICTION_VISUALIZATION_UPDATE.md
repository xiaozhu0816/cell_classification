# Prediction Visualization Update

## Summary
Added scatter plot visualization to show predicted vs. true time values for the multitask model's regression task. This helps assess the quality of time predictions across infected and uninfected cells.

## Changes Made

### 1. Modified `train_multitask.py`
- **Changed `evaluate()` function** to return predictions along with metrics
  - Returns tuple: `(metrics, predictions_dict)`
  - `predictions_dict` contains: `time_preds`, `time_targets`, `cls_preds`, `cls_targets`
  
- **Updated validation calls** to unpack the new return format:
  ```python
  val_metrics, _ = evaluate(...)  # Discard predictions during training
  ```

- **Save test predictions** after final evaluation:
  ```python
  test_metrics, test_predictions = evaluate(...)
  np.savez("test_predictions.npz",
           time_preds=..., time_targets=..., 
           cls_preds=..., cls_targets=...)
  ```

### 2. Enhanced `analyze_multitask_results.py`
- **Added `plot_prediction_scatter()` function** that creates a 3-subplot figure:
  1. **All Samples**: Shows both infected (red) and uninfected (blue) cells
     - Reference line for perfect prediction (diagonal)
     - Vertical line marking infection onset time
  
  2. **Infected Cells Only**: Focused view of time-since-infection predictions
     - Displays R² score and MAE
     - Shows prediction quality for infection progression
  
  3. **Uninfected Cells Only**: Focused view of experiment time predictions
     - Displays R² score and MAE
     - Shows prediction quality for temporal tracking

- **Automatic scatter plot generation** in main():
  ```python
  if predictions_file.exists():
      plot_prediction_scatter(predictions_file, output_path, infection_onset_hour)
  ```

## Output Files

### New Files Generated After Training
- `test_predictions.npz`: NumPy archive containing prediction arrays
  - `time_preds`: Predicted time values (hours)
  - `time_targets`: True time values (hours)
  - `cls_preds`: Classification probabilities
  - `cls_targets`: True class labels

- `prediction_scatter.png`: Scatter plot visualization with 3 subplots

### Existing Files (Still Generated)
- `training_curves.png`: Loss curves over epochs
- `validation_metrics.png`: Validation metrics over epochs
- `training_summary.txt`: Text summary report
- `results.json`: Final test metrics

## Usage

### For New Training Runs
Just train as usual - prediction plots are automatically generated:
```bash
python train_multitask.py --config configs/multitask_example.yaml
```

Output directory will now contain `prediction_scatter.png` automatically!

### For Existing Results (Without Predictions)
If you have old results without `test_predictions.npz`, you need to:

1. **Re-run test evaluation** with the updated `train_multitask.py`
2. Or **retrain** to get the new prediction files

### Manual Analysis (If Predictions Exist)
```bash
python analyze_multitask_results.py --result-dir outputs/multitask_resnet50/20251215-164539
```

## Interpretation of Scatter Plots

### Perfect Prediction
- Points lie on the diagonal reference line (y = x)
- R² = 1.0, MAE = 0

### Good Prediction
- Points cluster tightly around diagonal
- R² > 0.9, MAE < 2 hours
- Consistent across all time ranges

### Poor Prediction
- Points scattered far from diagonal
- R² < 0.7, MAE > 5 hours
- Systematic bias (points consistently above/below diagonal)

### What to Look For
1. **Infected vs Uninfected**: Are predictions equally good for both classes?
2. **Time Range Coverage**: Does accuracy degrade at early/late timepoints?
3. **Outliers**: Which samples have large prediction errors?
4. **Bias**: Are predictions systematically over/under-estimating?

## Example Output

The scatter plot shows:
- **Left subplot**: Combined view with infection onset marker
- **Middle subplot**: Infected cells with R² and MAE metrics
- **Right subplot**: Uninfected cells with R² and MAE metrics

Each subplot includes:
- Reference line (perfect prediction)
- Data points with transparency to show density
- Axis labels indicating what time values mean for each class
- Equal aspect ratio for accurate visual comparison

## Benefits

1. **Visual Quality Assessment**: Quickly see if time predictions are accurate
2. **Class-Specific Analysis**: Separate evaluation for infected vs uninfected
3. **Error Distribution**: See where predictions fail (early vs late times)
4. **Model Debugging**: Identify systematic biases or failure modes
5. **Temporal Pattern Discovery**: Understand if model captures time dynamics

## Technical Details

### Dependencies
- NumPy (already required)
- Matplotlib (already required)
- scikit-learn (for R² calculation)

### Prediction File Format
```python
# Load predictions
data = np.load("test_predictions.npz")
time_preds = data["time_preds"]      # Shape: (N,) - predicted hours
time_targets = data["time_targets"]  # Shape: (N,) - true hours
cls_preds = data["cls_preds"]        # Shape: (N,) - probabilities [0-1]
cls_targets = data["cls_targets"]    # Shape: (N,) - binary labels {0, 1}
```

### Metric Calculations
- **R² Score**: `sklearn.metrics.r2_score(y_true, y_pred)`
- **MAE**: `np.mean(np.abs(y_true - y_pred))`
- **RMSE**: `np.sqrt(np.mean((y_true - y_pred)**2))`

## Next Steps

For future training runs:
1. ✅ Predictions automatically saved
2. ✅ Scatter plots automatically generated
3. ✅ Summary statistics printed to console

You can now visually assess time prediction quality for every training run!
