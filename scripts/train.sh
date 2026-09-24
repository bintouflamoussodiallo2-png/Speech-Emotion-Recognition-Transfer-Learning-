#!/bin/bash 
#SBATCH --job-name=SER_train 
##SBATCH --account=def-smoolak
#SBATCH --gres=gpu:h100:1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=32G
#SBATCH --mail-user=ebd4258@umoncton.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --output=results/SER_%j.log 


source .venv/bin/activate

MODEL=$1
DATASET=$2
DATA_PATH=$3

echo "-------------------------"
echo " Job ID      : $SLURM_JOB_ID"
echo " Model       : $MODEL"
echo " Dataset     : $DATASET"
echo " Data path   : $DATA_PATH"
echo " GPU         : $CUDA_VISIBLE_DEVICES"
echo "-------------------------"

cd $SLURM_SUBMIT_DIR

python -m src.train \
    --model $MODEL \
    --dataset $DATASET \
    --data-path $DATA_PATH \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-4

