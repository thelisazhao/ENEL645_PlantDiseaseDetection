#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n3
#SBATCH --gres=gpu:1
#SBATCH --time=20:0:0
#SBATCH --mem=16G

export PATH=~/software/miniconda3/bin:$PATH
source activate tf2gpu

python /home/hunter.kimmett/ENEL645-Project/train_tomato_ssl.py
