#!/bin/bash
#SBATCH --job-name=parkseg-download
#SBATCH --account=coc
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

module load anaconda3
cd ~/scratch/Hacklytics
mkdir -p logs

python scripts/download_parkseg.py --output_dir data/parkseg12k
