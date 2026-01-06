# Multi-Task Model Data Usage Analysis

## Summary: YOUR MULTITASK MODEL USES **ALL** TIME RANGES ✅

Based on the training log and configuration in `outputs/multitask_resnet50/20251215-164539/`:

## Configuration Used

```yaml
data:
  frames:
    infected_window_hours: [0, 48]      # ✅ ALL infected timepoints
    uninfected_window_hours: [0, 48]    # ✅ ALL uninfected timepoints
```

## Dataset Split

- **Train samples**: 4,650
- **Val samples**: 558  
- **Test samples**: 1,488
- **Total**: 6,696 samples

## Comparison with Other Experiments

### 1. Baseline Single-Task Model (resnet50_baseline.yaml)
```yaml
infected_window_hours: [16, 30]   # ❌ FILTERED - only middle timepoints
uninfected_use_all: true          # ✅ All uninfected
```
**Problem**: Test set only has infected samples from [16-30h]
- This is why your sliding window analysis showed gaps!

### 2. Interval Sweep Experiments ✅
```yaml
# Uses apply_interval_override() to set different ranges
infected_window_hours: [1, 46]    # ✅ Varies per experiment
```
**Good**: Explicitly sets the range for each interval
- train-test_interval_1-46 has full range [1, 46h]

### 3. Multi-Task Model ✅
```yaml
infected_window_hours: [0, 48]    # ✅ FULL RANGE
uninfected_window_hours: [0, 48]  # ✅ FULL RANGE
```
**Excellent**: Uses ALL available data!

## Impact on Each Model

### Baseline Model (LIMITED)
- **Train data**: Infected [16-30h], Uninfected [0-48h]
- **Test data**: Infected [16-30h], Uninfected [0-48h]
- **Issue**: Cannot evaluate on early/late infection timepoints
- **Use case**: Only valid for mid-stage infection detection

### Interval Sweep (CONTROLLED)
- **Train data**: Varies by experiment (e.g., [1-46h])
- **Test data**: Controlled by experiment design
- **Good for**: Studying data efficiency and temporal patterns
- **Checkpoint train-test_interval_1-46**: Full range [1-46h] ✅

### Multi-Task Model (COMPREHENSIVE) ✅
- **Train data**: Infected [0-48h], Uninfected [0-48h]
- **Test data**: Infected [0-48h], Uninfected [0-48h]
- **Excellent for**: 
  - Full temporal coverage
  - Early infection detection
  - Time prediction across all stages
  - Comprehensive evaluation

## Why This Matters

### For Sliding Window Analysis
When you ran `analyze_final_model_sliding_window_fast.py`:

**Using baseline checkpoint** ❌
```
Window [1-7h]:   0 infected, 104 uninfected  (AUC: N/A)
Window [16-22h]: 104 infected, 104 uninfected (AUC: 1.0)  ← Only this range works!
Window [31-37h]: 0 infected, 104 uninfected  (AUC: N/A)
```

**Using interval sweep checkpoint (train-test_interval_1-46)** ✅
```
Window [1-7h]:   X infected, Y uninfected  (AUC: computable)
Window [16-22h]: X infected, Y uninfected  (AUC: computable)
Window [31-37h]: X infected, Y uninfected  (AUC: computable)
```

**Using multitask checkpoint** ✅
```
Full temporal coverage [0-48h] with ALL timepoints available!
```

## Recommendations

### 1. For Sliding Window Analysis
Use either:
- **Multitask checkpoint** (best - trained on [0-48h])
- **Interval sweep checkpoint** (`train-test_interval_1-46`)

**Don't use**: Baseline checkpoint (limited to [16-30h])

### 2. For Future Training
Always check `infected_window_hours` in your config:
```yaml
# Good - for comprehensive coverage
infected_window_hours: [0, 48]

# Bad - limits your model
infected_window_hours: [16, 30]
```

### 3. For Production Use
**Multi-task model** is your best option because:
- ✅ Trained on ALL timepoints [0-48h]
- ✅ Learns both classification AND time prediction
- ✅ Can detect infection at any stage
- ✅ Provides temporal context
- ✅ Test set has complete coverage

## Verification Commands

Check what data ANY model was trained on:

```bash
# Check config
python -c "
import json
with open('outputs/MODEL_DIR/results.json') as f:
    # Read until last complete field
    content = f.read().rsplit(',', 1)[0] + '}'
    data = json.loads(content)
    print('Infected:', data['config']['data']['frames']['infected_window_hours'])
    print('Uninfected:', data['config']['data']['frames'].get('uninfected_window_hours', 'all'))
"

# Or check training log
grep "infected_window_hours" outputs/MODEL_DIR/train.log/*.log
```

## Conclusion

✅ **Your multitask model uses ALL time ranges [0-48h]** - No filtering applied!

This makes it superior to the baseline model for:
- Full temporal analysis
- Early infection detection  
- Comprehensive evaluation
- Sliding window generalization testing

The baseline model's [16-30h] filter was appropriate for initial experiments but limits temporal generalization analysis.
