#!/usr/bin/env bash

#SBATCH -t 24:00:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 1         # number of cores 
#SBATCH --mem 3000
#SBATCH -o ./slurms/output.%j.%a.out
#SBATCH -e ./slurms/err.%j.%a


# module load pyger/0.9
conda init bash
conda activate sem

model=${1}
cond=${2}
lr=${3}
epoch=${4}
alfa=${5}
lmbda=${6}
seed=${7}

srun python PY-batch_exp.py "${model}" "${cond}" "${lr}" "${epoch}" "${alfa}" "${lmbda}" "${seed}"

sacct --format="CPUTime,MaxRSS"

