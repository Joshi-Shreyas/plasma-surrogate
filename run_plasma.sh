#!/bin/bash
#SBATCH --job-name=plasma_surrogate
#SBATCH --output=plasma_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=short

source /shared/EL9/explorer/miniconda3/24.11.1/miniconda3/etc/profile.d/conda.sh && conda activate plasma_env

cd ~/plasma_upgraded
python generate_etch_data.py
python train_lam_model.py
python predict_etch.py
python sensitivity_analysis.py
python plot_wafer_map.py
