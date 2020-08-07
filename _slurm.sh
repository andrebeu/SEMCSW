#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 3000 # memory pool for all cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
conda create --name rbaooituveqguuz -y
source activate rbaooituveqguuz
cd ~/SchemaPrediction
pip install -r requirements.txt &> ./logs/sem_install.log
python -u job_v071420_MLP.py no_split=False LSTM=False epsilon=1e-05 lr=0.001 n_epochs=16 log_alpha=8.0 log_lambda=13.0 batch_n=0 batch_update=False actor_weight=0.0 mixed=False  &> ./logs/_no_split=False_LSTM=False_epsilon=1e-05_lr=0.001_n_epochs=16_log_alpha=8.0_log_lambda=13.0_batch_n=0_batch_update=False_actor_weight=0.0_mixed=False_c.log
sleep 10
conda remove --name rbaooituveqguuz --all -y
sleep 10
