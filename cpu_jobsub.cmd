#!/usr/bin/env bash

#SBATCH -t 48:00:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 1         # number of cores 
#SBATCH --mem 3000
#SBATCH -o ./slurms/output.%j.%a.out
#SBATCH -e ./slurms/err.%j.%a


# module load pyger/0.9
conda init bash
conda activate sem

seed="${1}"
lr="${2}"
epoch="${3}"

srun python PY-run_simulations.py "${seed}" "${lr}" "${epoch}" 

sacct --format="CPUTime,MaxRSS"

