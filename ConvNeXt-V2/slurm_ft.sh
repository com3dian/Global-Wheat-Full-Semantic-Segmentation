#!/bin/bash

#SBATCH --job-name=ft_gwfss_exp7
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/ft_gwfss_exp7_%j.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/ft_gwfss_exp7_%j.out
#SBATCH --constraint='nvidia&A100'
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

#############################################################################################################
# slurm script for pretraining the MP-MAE 
#############################################################################################################

module load 2023 OpenBLAS/0.3.23-GCC-12.3.0
module load GPU cuDNN/8.7.0.84-CUDA-11.8.0

conda activate mmearth



python -u main_finetune.py \
    --finetune /lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/ConvNeXt-V2/pt_exp7/checkpoint-199.pth \
    --epochs 100 \
    --batch_size 32 \
    --update_freq 1 \
    --blr 1e-3 \
    --class_weights_beta 0.99 \
    --cutoff_epoch 0





### params :
#### epochs 10-100  
#### cutoff_epoch 0 or half of epochs
#### class_weights_beta 0-1 (in paper they used 0.99)
#### base lr (blr) : any range around 1e-2 (default in the paper)
#### update_freq : 1 (final epochs is epochs*update_freq) - we have 100 images only.