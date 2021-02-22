#!/usr/bin/env bash

#SBATCH -t 2:59:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 1         # number of cores 
#SBATCH --mem 4000
#SBATCH -o ./slurms/output.%j.%a.out


# module load pyger/0.9
conda init bash
conda activate sem

# nosplit=${1}
# condition=${2}
# learn_rate=${3}

# alfa=${4}
# lmda=${5}

seed=${SLURM_ARRAY_TASK_ID}

param_str=`python get_param_jobsub.py ${slurm_arr_idx}`
echo ${param_str}

# submit job
srun python PY-sem_batch_exp.py "${param_str}"

sacct --format="CPUTime,MaxRSS"

