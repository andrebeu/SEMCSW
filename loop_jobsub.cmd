#!/usr/bin/env bash

# parameter search over lr, n_epochs, alpha, lambda
declare -a lr_arr=(0.001 0.005 0.01 0.05)
declare -a epoch_arr=(1 2 4 8 16 32 64)
declare -a cond_arr=("blocked" "interleaved")

declare -a alfa_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)
declare -a lmbda_arr=(-32 -16 -8 -4 -2 0 2 4 8 16 32)

for seed in {0..19}; do 
  for lr in "${lr_arr[@]}"; do
    for alfa in "${alfa_arr[@]}"; do
      for lmbda in "${lmbda_arr[@]}"; do 
        for epoch in "${epoch_arr[@]}"; do 
          for cond in "${cond_arr[@]}"
            sbatch cpu_jobsub.cmd "SEM" "${cond}" "${lr}" "${epoch}" "${alfa}" "${lmbda}" "${seed}"
            sbatch cpu_jobsub.cmd "LSTM" "${cond}" "${lr}" "${epoch}" "${alfa}" "${lmbda}" "${seed}"
          done
        done
      done
    done
  done
done
