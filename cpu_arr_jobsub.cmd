#!/usr/bin/env bash

#SBATCH -t 24:00:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 1         # number of cores 
#SBATCH --mem 4000
#SBATCH -o ./slurms/output.%j.%a.out
#SBATCH -e ./slurms/err.%j.%a


# module load pyger/0.9
conda init bash
conda activate sem

seed=${1}

# get param str
slurm_arr_idx=${SLURM_ARRAY_TASK_ID}
param_str=`python get_param_jobsub.py ${slurm_arr_idx}`

# submit job
srun python PY-batch_exp.py "${seed}" "${param_str}" 

sacct --format="CPUTime,MaxRSS"

