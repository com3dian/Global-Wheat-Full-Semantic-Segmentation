#!/bin/bash

#SBATCH --job-name=pt_exp7
#SBATCH --error=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/pt_exp7_%j.err
#SBATCH --output=/lustre/scratch/WUR/AIN/nedun001/slurm_logs/pt_exp7_%j.out
#SBATCH --constraint='nvidia&A100'
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:00:00

#############################################################################################################
# slurm script for pretraining the MP-MAE 
#############################################################################################################

module load 2023 OpenBLAS/0.3.23-GCC-12.3.0
module load GPU cuDNN/8.7.0.84-CUDA-11.8.0

conda activate mmearth

# python  -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
#     --model convnextv2_atto \
#     --batch_size 64 \
#     --update_freq 8 \
#     --blr 1.5e-4 \
#     --epochs 1600 \
#     --warmup_epochs 40 \
#     --data_path /lustre/scratch/WUR/AIN/nedun001/gwfss/data \
#     --output_dir ./gwfss_results/ \
#     --distributed True


python  -u main_pretrain.py \
    --model convnextv2_atto \
    --batch_size 128 \
    --update_freq 8 \
    --blr 1.5e-4 \
    --epochs 200 \
    --warmup_epochs 40 \
    --data_path /lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/data \
    --output_dir /lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/ConvNeXt-V2/pt_exp7 \
    --distributed False \
    --domain_task True \
    --edge_task False \
    --inverse_domain_task True



# python  -u main_pretrain.py \
#     --model convnextv2_atto \
#     --batch_size 128 \
#     --update_freq 8 \
#     --blr 1.5e-4 \
#     --epochs 200 \
#     --warmup_epochs 40 \
#     --data_path /lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/data \
#     --output_dir test \
#     --distributed False \
#     --domain_task False \
#     --edge_task False \
#     --inverse_domain_task False