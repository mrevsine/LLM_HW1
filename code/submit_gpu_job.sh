#!/usr/bin/env bash
  
#SBATCH --time 12:00:00
#SBATCH --job-name=bitfit_roberta
#SBATCH --partition=mig_class
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --output=log_bitfit.txt
#SBATCH --export=ALL

ml anaconda
conda activate hw1
$@
