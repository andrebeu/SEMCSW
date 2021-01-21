#!/bin/bash

#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH -t 04:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 16				# number of cores 


module load pyger/0.9
conda activate sem

corpus_fpath="${1}"


srun python ${wd_dir}/w2v_train.py "${corpus_fpath}"


sacct --format="CPUTime,MaxRSS"