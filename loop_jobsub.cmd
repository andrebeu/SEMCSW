#!/usr/bin/env bash

# parameter search over lr, n_epochs, alpha, lambda
declare -a lr_arr=(0.0005 0.001 0.005 0.01 0.05 0.1)
declare -a epoch_arr=(1 2 4 8 16 32 64)

for lr in "${lr_arr[@]}"; do 
  for epoch in "${epoch_arr[@]}"; do 
    for seed in {1..20}; do 
      sbatch cpu_jobsub.cmd "${seed}" "${lr}" "${epoch}"
    done
  done
done
