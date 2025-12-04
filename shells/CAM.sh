cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/
python visualize_cam.py \
  --config configs/resnet50_baseline.yaml \
  --checkpoint ./checkpoints/resnet50_baseline/20251204-144709/fold_01of05/best.pt \
  --split val \
  --num-samples 8 \
  --num-folds 5 \
  --fold-index 0