#!/bin/bash
#SBATCH --job-name=confusion-matrices
#SBATCH --output=results/slurm_confusion_%j.out
#SBATCH --error=results/slurm_confusion_%j.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00

cd "/home/eddiecohanim/FinalProject/Final Project"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs236781-hw
export LD_PRELOAD=/home/eddiecohanim/miniconda3/envs/cs236781-hw/lib/libstdc++.so.6

for dir in results/v*/; do
    if [ -f "${dir}results.json" ] && [ ! -f "${dir}plot_1_metrics_bar_chart.png" ]; then
        echo "Generating plots for ${dir}..."
        (cd "$dir" && python -u "../../utilities/confusion matrix generator.py" results.json)
        echo "Done: ${dir}"
    fi
done

echo "All done."
