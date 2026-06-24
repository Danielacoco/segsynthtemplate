#!/bin/bash
# S3 — neonatal_dhcp_specialist
#SBATCH --job-name=S3_neonatal_dhcp_specialist
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
#SBATCH --output=slurm_logs/S3_neonatal_dhcp_specialist_%j.out
#SBATCH --error=slurm_logs/S3_neonatal_dhcp_specialist_%j.err

date
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/bach/da3999/miniconda3/etc/profile.d/conda.sh
conda activate fetalsyngen

which python

python /cluster/home/da3999/projects/segsynthtemplate/segsynthtemplate/train.py \
    experiment=mthesis/neonatal_dhcp_specialist_real \
    data.datasets.0.bids_path=/data/bach/da3999/neonataldHCP/resampled75/ \
    data.datasets.0.split_file=/data/bach/da3999/neonataldHCP/resampled75/splits_40_10_15_35.csv \
    data.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    model.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    data.generator.device=cuda:0 \
    logger.wandb.project=mthesis \
    logger.wandb.name=neonatal_dhcp_specialist \
    +resume_from=/cluster/home/da3999/None/logs/neonatal_dhcp_specialist/runs/2026-06-23_13-14-27/checkpoints/last.ckpt \
    +logger.wandb.id=000i86bo \
    +logger.wandb.resume=must
