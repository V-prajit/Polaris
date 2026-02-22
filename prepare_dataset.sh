#!/bin/bash
#SBATCH --job-name=parkseg-prep
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err

module load anaconda3

cd ~/scratch/Hacklytics
mkdir -p logs

# Optional: set your HF token for higher rate limits
# export HF_TOKEN="hf_..."

python3 yolo/prepare_dataset.py
