import numpy as np
from schema_prediction_task_7_14_20 import generate_exp, batch_exp
import time, os

output_file_path = './json_files_v071420_MLP/'

import random, string
def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def make_kw_string(kwargs):
    kw_string = ''
    for k, v in kwargs.items():
        kw_string += str(k) + '=' + str(v) + ' '
    return kw_string

def make_slurm_shell(kwargs, filename="_slurm.sh"):

    kw_string = make_kw_string(kwargs)
    tag = ''
    for s in kw_string.split(' '):
        tag += '_' + s

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
        "conda create --name {} -y".format(conda_name),
        "source activate {}".format(conda_name),
        "cd ~/SchemaPrediction",
        "pip install -r requirements.txt &> ./logs/sem_install_{}.log".format(conda_name),
        "python -u job_v071420_MLP.py {kw_string} &> ./logs/{tag}c.log".format(kw_string=kw_string, tag=tag),
        "sleep 10",
        "conda deactivate",
        "conda remove --name {} --all -y".format(conda_name),
        "rm -rf ~/.conda/envs/{}".format(conda_name),
        "sleep 10",
    ]

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    return 


if __name__ == "__main__":

    #online version or batch update?
    batch_update = False

    # dont' change these.
    #   Extensive testing says these values are fine and relatively unimportant!
    n_hidden_ = [None]
    epsilons = [1e-5]  
    # lrs = [0.0009]  # there are other lr that work as well, 
    # but this is in the set that includes max clustering performance (cf. SchemaPrediction v071420; pre-run)


    # extensive testing shows that a good learning rate is an order of magnitiude arround 1e-3 
    # lrs = [np.round(ii*10**-4,4) for ii in range(5, 10, 2)] + \
    #     [np.round(ii*10**-3,3) for ii in range(1, 7, 2)]
    # lrs = [np.round(ii*10**-4,4) for ii in range(5, 10, 2)] + [0.001]
    lrs = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001,
        0.002, 0.003, 0.004, 0.005]
    n_epochs_ = [4, 6, 8, 12, 16, 20, 24, 32]


    # n_epochs_ = [ii for ii in range(4, 128, 4)]
    # log_lambdas = [int(ii) for ii in np.logspace(2, 5, base=2, num=10)]



    # n_epochs_ = [int(ii) for ii in np.logspace(1.2, 1.5, base=10, num=10)]
    # n_epochs_ = [8, 11, 16, 23, 32, 45, 64, 91]
    log_alphas = [-9, -3, 0, 3, 9]
    log_lambdas = [0, 4, 8.0, 12., 16., 24., 32.]
    # log_lambdas = [-2.5, 0, 2.5, 5.0, 7.]

    n_batches = 25

    list_kwargs = []

    for no_split in [False]:
        for LSTM in [False]:
            for mixed in [True, False]:
                for epsilon in epsilons:
                    for lr in lrs:
                        for n_epochs in n_epochs_:
                            for log_alpha in log_alphas:
                                for log_lambda in log_lambdas:
                                    kwargs = dict(
                                        no_split=no_split,
                                        LSTM=LSTM,
                                        epsilon=epsilon,
                                        lr=lr,
                                        n_epochs=n_epochs,
                                        log_alpha=log_alpha,
                                        log_lambda=log_lambda,
                                    n_batches=n_batches,
                                        batch_update=batch_update,
                                        actor_weight=0.0,
                                        mixed=mixed
                                    )
                                    list_kwargs.append(kwargs)

    log_alphas = [-208]
    log_lambdas = [208]

    n_batches = 25
    
    for no_split in [True]:
        for LSTM in [False]:
            for mixed in [True, False]:
                for epsilon in epsilons:
                    for lr in lrs:
                        for n_epochs in n_epochs_:
                            for log_alpha in log_alphas:
                                for log_lambda in log_lambdas:
                                    kwargs = dict(
                                        no_split=no_split,
                                        LSTM=LSTM,
                                        epsilon=epsilon,
                                        lr=lr,
                                        n_epochs=n_epochs,
                                        log_alpha=log_alpha,
                                        log_lambda=log_lambda,
                                        n_batches=n_batches,
                                        batch_update=batch_update,
                                        actor_weight=0.0,
                                        mixed=mixed
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
    # for kwarg in list_kwargs:
    #     print(make_kw_string(kwarg))
        
    # # create the slurm submissions 
    # for ii, kwargs in enumerate(list_kwargs):
    #     print('Submitting job {} of {}'.format(ii + 1, n))
    #     make_slurm_shell(kwargs, filename="_slurm.sh")

    #     os.system('sbatch _slurm.sh')
    #     time.sleep(0.1)
    #     os.remove('_slurm.sh')

