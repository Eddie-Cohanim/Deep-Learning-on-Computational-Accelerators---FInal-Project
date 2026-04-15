#!/bin/bash
#SBATCH --job-name=wine-cnn
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

cd "/home/eddiecohanim/FinalProject/Final Project"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw
python -u main.py

# Move slurm logs into the versioned results folder that main.py just created
LATEST_VERSION=$(ls -td results/v*/ 2>/dev/null | head -1)
if [ -n "$LATEST_VERSION" ]; then
    mv "results/slurm_${SLURM_JOB_ID}.out" "$LATEST_VERSION" 2>/dev/null
    mv "results/slurm_${SLURM_JOB_ID}.err" "$LATEST_VERSION" 2>/dev/null
    python -u plot_results.py "$LATEST_VERSION"
    (cd "$LATEST_VERSION" && python -u "../../confusion matrix generator.py" results.json)
fi
