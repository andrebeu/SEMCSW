#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 3000 # memory pool for all cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
source activate schema
cd ~/SchemaPrediction
python -u job_v061020.py no_split=False n_hidden=10 LSTM=False epsilon=1e-07 lr=0.006 n_epochs=64 log_alpha=-1.0 log_lambda=50 batch_n=2  &> ./logs/_no_split=False_n_hidden=10_LSTM=False_epsilon=1e-07_lr=0.006_n_epochs=64_log_alpha=-1.0_log_lambda=50_batch_n=2_c.log
sleep 10
