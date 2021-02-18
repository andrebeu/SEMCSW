#!/usr/bin/env bash

#SBATCH -t 5:59:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 1         # number of cores 
#SBATCH --mem 4000
#SBATCH -o ./slurms/output.%j.%a.out


# module load pyger/0.9
conda init bash
conda activate sem

nosplit=${1}
condition=${2}
learn_rate=${3}

alfa=1
lmda=1

seed=${SLURM_ARRAY_TASK_ID}

# submit job
srun python PY-sem_seed_exp.py "${nosplit}" "${condition}" "${learn_rate}" "${alfa}" "${lmda}" "${seed}"

sacct --format="CPUTime,MaxRSS"

