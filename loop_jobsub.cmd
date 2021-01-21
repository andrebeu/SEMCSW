#!/usr/bin/env bash

# parameter search over lr, n_epochs, alpha, lambda
# declare -a lr_arr=(0.0001 0.0005 0.001 0.005 0.01 0.05 0.1)
declare -a lr_arr=(0.01 0.05 0.1)
# declare -a epoch_arr=(1 2 4 8 16 32 64)
declare -a epoch_arr=(1 4 16 64)
# declare -a alpha_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)
declare -a alpha_arr=(-32 -8 0 16 32)
# declare -a lambda_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)
declare -a lambda_arr=(-32 -8 0 16 32)

for lr in "${lr_arr[@]}"; do 
  for epoch in "${epoch_arr[@]}"; do 
    for alpha in "${alpha_arr[@]}"; do 
      for lambda in "${lambda_arr[@]}"; do 
        sbatch cpu_jobsub.cmd "${lr}" "${epoch}" "${alpha}" "${lambda}"
      done
    done
  done
done
