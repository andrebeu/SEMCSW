import numpy as np
import time, os
import random, string
import pandas as pd
from tqdm import tqdm

job_file = 'job_v090820.py'

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def make_kw_string(kwargs):
    kw_string = ''
    for k, v in kwargs.items():
        kw_string += str(k) + '=' + str(v) + ' '
    return kw_string

def make_kw_tag(kwargs):
    kw_string = ''
    for k, v in kwargs.items():
        if k is not 'output_file_path':
            kw_string += str(k) + '=' + str(v) + '_'    
    return kw_string[:-1] # remove final underscore


def make_slurm_shell(kwargs, filename="_slurm.sh"):

    kw_string = make_kw_string(kwargs)
    tag = make_kw_tag(kwargs)


    # generate a random name of a conda enviornment
    conda_name = 'temp_' + get_random_string(4)  

    # these are the file lines we will have
    lines = [
        "#!/bin/bash",
        "#SBATCH -p shared # partition (queue)",
        "#SBATCH -N 1 # number of nodes",
        "#SBATCH -n 1 # number of cores",
        "#SBATCH --mem 3000 # memory pool for all cores",
        "#SBATCH -t 0-16:00 # time (D-HH:MM)",
        "#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT",
        "#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR",
        "",
        "module load Anaconda3/2019.10",
        "sleep 30",
        "source activate schema",
        "cd ~/SchemaPrediction",
        "python -u {file} {kw_string} &> ./logs/{tag}c.log".format(
            file=job_file, kw_string=kw_string, tag=tag),
        "sleep 10",
        "conda deactivate",
        "sleep 10",
    ]

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    return 


if __name__ == "__main__":

    output_file_path = './json_files_v090820/'
    
    #online version or batch update?
    batch_update = False

    # run both the mixed or just blocked/interleaved?
    mixed = False

    # run the instructed case?
    instructed = True
    
    # run the LSTM version or the MLP?
    LSTM = True

    # Save the Prediction error and boundary info? (storage intensive)
    condensed_output = True

    # dont' change these.
    #   Extensive testing says these values are fine and relatively unimportant!
    n_hidden = None
    epsilons = [1e-5]  

    # parameter search over lr, n_epochs, alpha, lambda
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    n_epochs_ = [1, 2, 4, 8, 16, 32, 64]
    log_alphas = [-32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32]
    log_lambdas = [-32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32]
    
    # How many batches per simulation? Should be kept low for parameter searches
    n_batches = 50

    list_kwargs = []

    for epsilon in epsilons:
        for lr in lrs:
            for n_epochs in n_epochs_:
                for log_alpha in log_alphas:
                    for log_lambda in log_lambdas:
                        kwargs = dict(
                            no_split=False,
                            LSTM=LSTM,
                            epsilon=epsilon,
                            lr=lr,
                            n_epochs=n_epochs,
                            n_hidden=n_hidden,
                            log_alpha=log_alpha,
                            log_lambda=log_lambda,
                            n_batches=n_batches,
                            batch_update=batch_update,
                            actor_weight=0.0,
                            mixed=mixed,
                            output_file_path=output_file_path,
                            condensed_output=condensed_output,
                            instructed=instructed,
                        )
                        list_kwargs.append(kwargs)

                # append the LSTM Simulations
                kwargs = dict(
                    no_split=True,
                    LSTM=True,
                    epsilon=epsilon,
                    lr=lr,
                    n_epochs=n_epochs,
                    n_hidden=n_hidden,
                    log_alpha=-208,
                    log_lambda=208,
                    n_batches=n_batches,
                    batch_update=batch_update,
                    actor_weight=0.0,
                    mixed=mixed,
                    output_file_path=output_file_path,
                    condensed_output=condensed_output,
                    instructed=instructed,
                )
                list_kwargs.append(kwargs)

                # append the MLP Simulations
                kwargs = dict(
                    no_split=True,
                    LSTM=False,
                    epsilon=epsilon,
                    lr=lr,
                    n_epochs=n_epochs,
                    n_hidden=n_hidden,
                    log_alpha=-208,
                    log_lambda=208,
                    n_batches=n_batches,
                    batch_update=batch_update,
                    actor_weight=0.0,
                    mixed=mixed,
                    output_file_path=output_file_path,
                    condensed_output=condensed_output,
                    instructed=instructed,
                )
                list_kwargs.append(kwargs)

    # randomize the simulation order for effective sampling speed 
    # (i.e. intermediate progress is more meaningful)
    list_kwargs = np.random.permutation(list_kwargs)
    n = len(list_kwargs)
    print(n)
    print(n_epochs_)
    print(lrs)
    print(log_alphas)
    print(log_lambdas)

    
    # create the slurm submissions 
    for ii, kwargs in enumerate(list_kwargs):
        print('Submitting job {} of {}'.format(ii + 1, n))
        make_slurm_shell(kwargs, filename="_slurm.sh")

        os.system('sbatch _slurm.sh')
        time.sleep(0.25)
        os.remove('_slurm.sh')

