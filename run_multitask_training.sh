#!/bin/bash
#SBATCH --job-name=multitask_train
#SBATCH --output=slurm_LOG/multitask_%j.out
#SBATCH --error=slurm_LOG/multitask_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Multi-Task Training Script
# Updates:
# - infection_onset_hour: 1.0 (changed from 2.0)
# - epochs: 30 (increased from 20)
# - Auto-generates ALL visualizations after training:
#   * training_curves.png
#   * validation_metrics.png
#   * prediction_scatter_regression.png (with regression lines!)
#   * training_summary.txt

echo "========================================="
echo "Multi-Task Training with Auto-Visualization"
echo "========================================="
echo "Config: configs/multitask_example.yaml"
echo "Infection onset: 1.0 hour"
echo "Epochs: 30"
echo ""

# Activate environment if needed
# source activate your_env

# Run training
echo "Starting training..."
python train_multitask.py --config configs/multitask_example.yaml

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Check outputs/multitask_resnet50/<run_id>/ for:"
echo "  - results.json (metrics)"
echo "  - training_curves.png (loss curves)"
echo "  - validation_metrics.png (validation metrics)"
echo "  - prediction_scatter_regression.png (pred vs true with regression)"
echo "  - training_summary.txt (text report)"
echo "  - test_predictions.npz (raw predictions)"
echo "========================================="
