#!/bin/bash
#SBATCH --job-name=segformer-train
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=500G
#SBATCH --time=03:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load anaconda3
cd ~/scratch/Hacklytics
mkdir -p logs

export PYTORCH_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/train_segformer.py \
  --data_dir    ~/scratch/Hacklytics/data/parkseg12k \
  --output_dir  ~/scratch/Hacklytics/checkpoints \
  --epochs 50 \
  --batch_size 16 \
  --lr 6e-5 \
  --weight_decay 0.01 \
  --img_size 512 \
  --num_workers 8
