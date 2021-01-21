#!/usr/bin/env bash

#SBATCH -t 48:00:00   # runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1         # node count 
#SBATCH -c 4         # number of cores 
#SBATCH -o ./slurms/output.%j.%a.out


# module load pyger/0.9
conda init bash
conda activate sem

model="${1}"
lr="${2}"
epoch="${3}"
alpha="${4}"
lambda="${5}"


srun python PY-run_simulations.py "${model}" "${lr}" "${epoch}" "${alpha}" "${lambda}"

sacct --format="CPUTime,MaxRSS"
