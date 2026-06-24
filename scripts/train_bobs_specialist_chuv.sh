#!/bin/bash
# S4 — bobs_specialist
#SBATCH --job-name=S4_bobs_specialist
#SBATCH --chdir=/cluster/home/da3999/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crro.daniela@gmail.com
#SBATCH --account=rad
#SBATCH --partition=rad2
#SBATCH --qos=16cpu
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/S4_bobs_specialist_%j.out
#SBATCH --error=slurm_logs/S4_bobs_specialist_%j.err

date
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/bach/da3999/miniconda3/etc/profile.d/conda.sh
conda activate fetalsyngen

which python

python /cluster/home/da3999/projects/segsynthtemplate/segsynthtemplate/train.py \
    experiment=mthesis/bobs_specialist_real \
    data.datasets.0.bids_path=/data/bach/Vlad/data/neonatal_bobsV1.0/resampled75/ \
    data.datasets.0.split_file=/data/bach/Vlad/data/neonatal_bobsV1.0/resampled75/splits_40_10_15_35.csv \
    data.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    model.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    data.generator.device=cuda:0 \
    logger.wandb.project=mthesis \
    logger.wandb.name=bobs_specialist \
    +resume_from=/cluster/home/da3999/None/logs/bobs_specialist/runs/2026-06-22_13-39-47/checkpoints/epoch_874_val_dice_0.4831_step_13125.ckpt \
    +logger.wandb.id=xkl1akq4 \
    +logger.wandb.resume=must
