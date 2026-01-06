#!/bin/bash
#SBATCH --job-name=multitask_cell
#SBATCH --output=slurm_LOG/multitask_%j.out
#SBATCH --error=slurm_LOG/multitask_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Multi-Task Cell Classification Training
# Trains a model to jointly:
#   1. Classify cells as infected/uninfected
#   2. Predict time (different references for infected vs uninfected)

echo "========================================="
echo "Multi-Task Cell Classification Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate conda environment if needed
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cell_classification

# Configuration file
CONFIG="configs/multitask_example.yaml"

echo "Configuration: $CONFIG"
echo ""

# Run training
python train_multitask.py --config "$CONFIG"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
