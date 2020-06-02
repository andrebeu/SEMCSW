#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -W # wait for the subprocesses to end? 
#SBATCH --mem 100 # memory pool for all cores
#SBATCH -t 1-0:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
conda create -n schema
source activate schema
cd ~/SchemaPrediction
pip install -r requirements.txt
python -u schema_prediction_batch_runner_05-12-20.py 1 &> batch_runner_1.log
sleep 10
