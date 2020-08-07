#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 1000 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
conda create -n schema
source activate schema
cd ~/SchemaPrediction
rm ~/SchemaPrediction/logs/*
rm ~/SchemaPrediction/slurm_output/*
rm ~/SchemaPrediction/json_files_v071420/*
rm ~/SchemaPrediction/json_files_v071420_MLP/*
# pip install -r requirements.txt &> ./logs/sem_install.log

# sleep 5h

# python -u schema_prediction_batch_runner_07-14-20_Andre.py &> ./logs/batch_submitter_2.log
# python -u schema_prediction_batch_runner_07-22-20_customGRU.py &> ./logs/batch_submitter_1.log
python -u schema_prediction_batch_runner_08-03-20.py &> ./logs/batch_submitter_0.log

sleep 10
