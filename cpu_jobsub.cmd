#!/bin/bash

#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH -t 04:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 16				# number of cores 


module load pyger/0.9
conda activate sem

lr="${1}"
n_epochs="${2}"
log_alpha="${3}"
log_lambda="${4}"

srun python PY-run_simulations.py "SEM" "${lr}" "${n_epochs}" "${log_alpha}" "${log_lambda}"
srun python PY-run_simulations.py "LSTM" "${lr}" "${n_epochs}" "${log_alpha}" "${log_lambda}"

sacct --format="CPUTime,MaxRSS"