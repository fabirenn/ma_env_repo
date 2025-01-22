#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=32GB
#SBATCH --time=5-00:00:00
#SBATCH --job-name="unet-tune"
#SBATCH --partition=clara-long
#SBATCH --output=../log/%x.out-%j
#SBATCH --error=../log/%x.error-%j
#SBATCH --mail-type=END

source ../ma_env/bin/activate

module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

#python src/u_net/train_unet.py
python src/u_net/optuna_unet.py
