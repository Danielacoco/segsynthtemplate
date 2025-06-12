#!/bin/bash
#SBATCH --job-name=synthart
#SBATCH --chdir=/cluster/home/vl3393/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vladyslav.zalevskyi@chuv.ch
#SBATCH --ntasks=1
#SBATCH --time=1-23:59:00
#SBATCH --output=slurm_logs/synthart.out
#SBATCH --error=slurm_logs/synthart.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=64gb
## Apparently these lines are needed for GPU execution
#SBATCH --account rad
#SBATCH --partition rad
#SBATCH --gres=gpu:rtx6000:1

date

nvidia-smi

source /cluster/home/vl3393/miniconda3/etc/profile.d/conda.sh

conda activate fetalsyngen

which python

python /cluster/home/vl3393/projects/segsynthtemplate/segsynthtemplate/train.py \
experiment=fdhcp_synth_train_yaartif

date