#!/bin/bash
# S2 — dhcp_specialist_conemos (fetal dHCP only)
#SBATCH --job-name=S2_dhcp_specialist
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
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/S2_dhcp_specialist_%j.out
#SBATCH --error=slurm_logs/S2_dhcp_specialist_%j.err

date
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/bach/da3999/miniconda3/etc/profile.d/conda.sh
conda activate fetalsyngen

which python

python /cluster/home/da3999/projects/segsynthtemplate/segsynthtemplate/train.py \
    experiment=mthesis/dhcp_specialist_conemos \
    data.datasets.0.bids_path=/data/bach/Vlad/data/fetalDHCP/resampled75/ \
    data.datasets.0.split_file=/data/bach/Vlad/data/fetalDHCP/resampled75/splits_40_10_15_35.csv \
    data.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    model.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    data.generator.device=cuda:0 \
    logger.wandb.project=mthesis \
    logger.wandb.name=dhcp_specialist_conemos \
    +resume_from=/cluster/home/da3999/None/logs/dhcp_specialist_conemos/runs/2026-06-21_11-42-43/checkpoints/last.ckpt \
    +logger.wandb.id=y5ejjwpl \
    +logger.wandb.resume=must
    
    
