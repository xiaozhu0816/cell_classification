# Draft – Multitask infection state + progression time from microscopy

## Candidate title

Joint prediction of infection state and progression time from microscopy using a multi-task ResNet with temporal reliability profiling

## Abstract (bullet scaffold)

- Background: automated infection phenotyping from microscopy; need both state and stage.

- Methods: multi-task ResNet50 predicting (i) infection status and (ii) progression time; 5-fold CV; temporal sliding-window analysis.

- Leakage mitigation / final setting: because GMU noted ~5% field-of-view overlap between adjacent acquisition positions, we use a simple, robust preprocessing step that removes the outer 5% border of each frame (center-crop), then resizes to the network input size. This reduces the chance that near-duplicate border content appears across train/test folds.

- Results: include overall metrics + temporal early-stage difficulty; error profiling and reliability discussion.

- Conclusion: joint modeling provides accurate state classification and hour-level staging with stage-dependent reliability.

## Results folder

- CV results: `/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/multitask_resnet50_crop5pct/20260114-170730_5fold`

- Key figures already generated: prediction_scatter, classification_by_time_window, cv_temporal_generalization, error_distribution_by_time_range, regression_residual_over_time

## Key numbers (final, crop-5% setting)

Cross-validation summary (5-fold, mean ± std):

- Classification accuracy: 0.9925 ± 0.0010
- Classification F1: 0.9924 ± 0.0010
- Classification AUC: 0.9998 ± 0.0001
- Regression MAE: 1.176 ± 0.097 h
- Regression RMSE: 1.491 ± 0.102 h

Temporal reliability (6 h windows, 3 h stride; window centers in hours):

- Early-stage performance is lower and then rapidly improves:
	- 3 h window: accuracy 0.9443 ± 0.0043; AUC 0.9908 ± 0.0026; recall 0.8825 ± 0.0170
	- 6 h window: accuracy 0.9646 ± 0.0051; AUC 0.9964 ± 0.0014; recall 0.9333 ± 0.0182
- From 9 h onward, temporal-window classification is effectively saturated (AUC≈1.0; accuracy≈1.0 in all windows).



## Figure list (proposed)

1. Overview of task + model (diagram to be drawn).

2. Classification performance overall + temporal windows (`classification_by_time_window.png`).

3. Regression accuracy (scatter) + residual over time (`prediction_scatter.png`, `regression_residual_over_time.png`).

4. Error distribution by time range and valley period (`error_distribution_by_time_range.png`, `valley_period_analysis.png`).

5. Temporal generalization (`cv_temporal_generalization.png`).

## Tables (final, crop-5% setting)

### Table 1. Overall 5-fold CV metrics (mean ± std; min/max)

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| cls_accuracy | 0.99253 | 0.00096 | 0.99117 | 0.99389 |
| cls_precision | 0.99945 | 0.00111 | 0.99723 | 1.00000 |
| cls_recall | 0.98544 | 0.00283 | 0.98214 | 0.99038 |
| cls_f1 | 0.99239 | 0.00099 | 0.99099 | 0.99380 |
| cls_auc | 0.99983 | 0.00005 | 0.99975 | 0.99992 |
| reg_mae (h) | 1.17622 | 0.09701 | 1.04282 | 1.29194 |
| reg_rmse (h) | 1.49116 | 0.10206 | 1.35687 | 1.62636 |

### Table 2. Temporal window classification metrics (6 h window, 3 h stride; mean ± std)

| Window center (h) | Accuracy | F1 | AUC | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.9443 ± 0.0043 | 0.9350 ± 0.0057 | 0.9908 ± 0.0026 | 0.9947 ± 0.0107 | 0.8825 ± 0.0170 |
| 6 | 0.9646 ± 0.0051 | 0.9634 ± 0.0057 | 0.9964 ± 0.0014 | 0.9958 ± 0.0084 | 0.9333 ± 0.0182 |
| 9 | 0.9938 ± 0.0039 | 0.9937 ± 0.0039 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9875 ± 0.0078 |
| 12 | 0.9990 ± 0.0021 | 0.9990 ± 0.0021 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9979 ± 0.0042 |
| 15 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 18 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 21 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 24 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 27 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 30 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 33 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 36 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 39 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 42 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| 45 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## Methods (what to describe)

- Dataset: acquisition, labeling, time definition, splitting strategy (avoid leakage).

- Anti-overlap preprocessing (final experiment): to mitigate possible leakage caused by approximate ~5% overlap between adjacent acquisition positions reported by GMU, we remove the outer 5% border from each frame using a center crop (i.e., keep the central 90%×90% region), then perform the standard resize to the model input resolution. This preprocessing is applied consistently to train/validation/test.

- Model: ResNet50 backbone, two heads, losses and weighting, training details.

- Evaluation: 5-fold CV, metrics, temporal windows (6h/3h), statistical reporting.

## Notes / TODO

- Verify worst_predictions_report classification section matches CV summary (should now be fixed).

- Add dataset description + imaging interval + biological interpretation of early-stage/valley period.
