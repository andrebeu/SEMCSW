#!/bin/bash
#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 1000 # memory pool for all cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT
#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR

module load Anaconda3/2019.10
conda create -n schema
source activate schema
cd ~/SchemaPrediction
pip install -r requirements.txt
python -u schema_prediction_batch_runner_06-12-20.py &> ./logs/batch_submitter_5.log
sleep 10
