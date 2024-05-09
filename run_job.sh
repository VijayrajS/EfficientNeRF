#!/bin/bash
#SBATCH -J Prune_NeRF_SingleNode              # Job name
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-cpu=16G
#SBATCH -t 480                                  # Duration of the job (Max = 480 min = 8h)
#SBATCH -oReport-%j.out                         # Combined output and error messages file

# module load cuda
source $CONDA_SRC
cd $NERF_FOLDER

# Activate your Conda environment
conda activate nerf
python prune.py --config configs/prune_fern.txt
# python run_nerf.py --config configs/fern.txt
wait
