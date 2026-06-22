#!/bin/bash
#SBATCH --job-name=feta_specialist
#SBATCH --chdir=/cluster/home/da3999/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=crro.daniela@gmail.com
#SBATCH --account=rad
#SBATCH --partition=rad1
#SBATCH --qos=8cpu
#SBATCH --gres=gpu:v100:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-23:59:00
#SBATCH --output=slurm_logs/feta_specialist_%j.out
#SBATCH --error=slurm_logs/feta_specialist_%j.err

date
nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /data/bach/da3999/miniconda3/etc/profile.d/conda.sh
conda activate fetalsyngen

which python

python /cluster/home/da3999/projects/segsynthtemplate/segsynthtemplate/train.py \
    experiment=mthesis/feta_specialist_conemos \
    data.datasets.0.bids_path=/data/bach/Vlad/data/FETA/merged_feta_spinabifida/derivatives/resampled05/ \
    data.datasets.0.split_file=/data/bach/da3999/feta_splits_40_10_15_35.csv \
    data.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    model.label_map_csv=/cluster/home/da3999/projects/segsynthtemplate/label_map_multiprotocol.csv \
    "model.protocol_names=[feta]"
