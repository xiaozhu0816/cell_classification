python compare_all_models_improved.py \
  --single-classification "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/interval_sweep_analysis/20251212-145928/train-test_interval_1-46_sliding_window_fast_20251231-161811/final_model_sliding_w6_s3_data.json" \
  --multitask "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/multitask_resnet50/20260109-164300_5fold" \
  --multitask-conditioned "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/multitask_cls_conditioned/20260109-164308_5fold" \
  --regression-infected "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/regression_infected/20260109-164401_5fold" \
  --regression-uninfected "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/outputs/regression_uninfected/20260109-164348_5fold" \
  --output-dir comparison_results_improved_1