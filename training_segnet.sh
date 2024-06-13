#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=10:00:00
#SBATCH --mem=100G
#SBATCH --partition=clara
#SBATCH --output=../log/%x.out-%j
#SBATCH --error=../log/%x.error-%j
#SBATCH --mail-type=END

source ../ma_env/bin/activate

module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0


python src/segnet/train_segnet.py
