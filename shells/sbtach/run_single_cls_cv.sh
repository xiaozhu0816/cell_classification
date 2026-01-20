#!/bin/bash
#SBATCH --job-name=run_single_cls
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=./slurm_LOG/out_%j.log
#SBATCH --error=./slurm_LOG/err_%j.log

# ---- Optional: Load conda or modules ----
# module load anaconda
# source activate my_env

cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/

# Single-task classification baseline (crop-5% variant)
python train.py --config configs/resnet50_baseline_crop5pct.yaml --k-folds 5
