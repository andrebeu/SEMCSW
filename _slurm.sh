#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 3000 # memory pool for all cores
#SBATCH -t 1-12:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
conda create -n schema
source activate schema
cd ~/SchemaPrediction
pip install -r requirements.txt
python -u job_v060220.py no_split=True LSTM=False n_hidden=10 lr=0.0005 n_epochs=16 log_alpha=-8 log_lambda=0 batch_n=0  &> _no_split=True_LSTM=False_n_hidden=10_lr=0.0005_n_epochs=16_log_alpha=-8_log_lambda=0_batch_n=0_c.log
sleep 10
