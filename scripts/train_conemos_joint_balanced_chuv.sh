#!/bin/bash
# J2 — conemos_joint_balanced
#SBATCH --job-name=J2_conemos_joint_balanced
#SBATCH --chdir=/cluster/home/da3999/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crro.daniela@gmail.com
#SBATCH --account=rad
#SBATCH --partition=rad3
#SBATCH --qos=32cpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/J2_conemos_joint_balanced_%j.out
#SBATCH --error=slurm_logs/J2_conemos_joint_balanced_%j.err

date
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/bach/da3999/miniconda3/etc/profile.d/conda.sh
conda activate fetalsyngen

which python

python /cluster/home/da3999/projects/segsynthtemplate/segsynthtemplate/train.py \
    experiment=mthesis/train_conemos_balanced \
    data.datasets.0.bids_path=/data/bach/da3999/FeTa/resampled75/ \
    data.datasets.0.split_file=/data/bach/da3999/FeTa/resampled75/splits_40_10_15_35.csv \
    data.datasets.1.bids_path=/data/bach/Vlad/data/CHUV_fetal/resampled75/ \
    data.datasets.1.split_file=/data/bach/Vlad/data/CHUV_fetal/resampled75/splits_40_10_15_35.csv \
    data.datasets.2.bids_path=/data/bach/Vlad/data/fetalDHCP/resampled75/ \
    data.datasets.2.split_file=/data/bach/Vlad/data/fetalDHCP/resampled75/splits_40_10_15_35.csv \
    data.datasets.3.bids_path=/data/bach/da3999/neonataldHCP/resampled75/ \
    data.datasets.3.split_file=/data/bach/da3999/neonataldHCP/resampled75/splits_40_10_15_35.csv \
    data.datasets.4.bids_path=/data/bach/Vlad/data/neonatal_bobsV1.0/resampled75/ \
    data.datasets.4.split_file=/data/bach/Vlad/data/neonatal_bobsV1.0/resampled75/splits_40_10_15_35.csv \
    data.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    model.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    data.generator.device=cuda:0 \
    logger.wandb.project=mthesis \
    logger.wandb.name=conemos_joint_balanced
