#!/usr/bin/env bash

#SBATCH -t 6:00:00   # runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1         # node count 
#SBATCH -c 4         # number of cores 
#SBATCH -o ./slurms/output.%j.%a.out


# module load pyger/0.9
conda init bash
conda activate sem

lr="${1}"
epoch="${2}"
alpha="${3}"
lambda="${4}"

srun python PY-run_simulations.py "SEM" "${lr}" "${epoch}" "${alpha}" "${lambda}"
srun python PY-run_simulations.py "LSTM" "${lr}" "${epoch}" "${alpha}" "${lambda}"

sacct --format="CPUTime,MaxRSS"
