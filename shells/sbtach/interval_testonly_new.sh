#!/bin/bash
#SBATCH --job-name=interval_sweep_both
#SBATCH --partition=ciaq             # 你自己在用 ciaq 分区 → 只能 1 GPU
#SBATCH --gres=gpu:1                 # ciaq 上最多 1 块 GPU
#SBATCH --cpus-per-task=16           # 数据加载够用
#SBATCH --mem=40G                    # 按教程 (你可以调大也可以调小)
#SBATCH --time=7-00:00:00            # 你的实验应该比较久
#SBATCH --output=./slurm_LOG/out_%j.log          # 输出日志
#SBATCH --error=./slurm_LOG/err_%j.log           # 错误日志

# ---- Optional: Load conda or modules ----
# module load anaconda                 # 你的集群普遍有这个
# source activate my_env               # 换成你自己的 conda 环境

# ---- 进入你项目所在路径（非常重要） ----
cd /isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_classification/

# ---- 最关键：运行你的 experiment 脚本 ----
python analyze_interval_sweep_train.py \
    --upper-hours 7 10 13 16 19 22 25 28 31 34 37 40 43 46 \
    --mode test-only \
    --eval-only \
    --checkpoint-dir "outputs/interval_sweep_analysis/20251212-145928" \
    --match-uninfected-window
