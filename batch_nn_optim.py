#!/bin/bash

for epsilon in '1e-5':
do
    


dropouts = [0.0]
# lrs = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
lrs = [0.001, 0.0015, 0.002, 0.004, 0.008, 0.01]
# n_epochs_list = [5, 6, 7, 8, 9, 10]
n_epochs_list = [8, 9, 10, 11, 12, 13, 14, 15]
epsilons = [1e-5]

# dropouts = [0.0]
# lrs = [0.008]
# n_epochs_list = [12]
# epsilons = [1e-5]

for epsilon in epsilons:
    for lr in lrs:
        for n_epochs in n_epochs_list:
            for dropout in dropouts:
                command = "args"