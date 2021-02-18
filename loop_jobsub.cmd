#!/usr/bin/env bash

# parameter search over learn_rate, n_epochs, alpha, lambda
# declare -a learn_rate_arr=(0.001 0.005 0.01 0.05)
# declare -a epoch_arr=(1 2 4 8 16 32 64)

declare -a learn_rate_arr=(0.005 0.01 0.05 0.1)
declare -a condition_arr=("blocked" "interleaved")
# declare -a alfa_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)
# declare -a lmbda_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)

## slurm array idx passed to py script: 
# builds product of params
# takes idx as input
# returns str param set


# sbatch --array=0-100 cpu_arr_jobsub.cmd arg arg

for learn_rate in "${learn_rate_arr[@]}"; do
  for condition in "${condition_arr[@]}"; do
    # LSTM
    sbatch --array=0-100 cpu_arr_jobsub.cmd "1" "${condition}" "${learn_rate}"
  done
done
