import numpy as np
from schema_prediction_task_6_10_20 import generate_exp, batch_exp
import time, tqdm, os

## this version uses the 6_10_20 task, but a modified event model

output_file_path = './json_files_v07020/'


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

    # these are the file lines we will have
    lines = [
        "#!/bin/bash",
        "#SBATCH -p shared # partition (queue)",
        "#SBATCH -N 1 # number of nodes",
        "#SBATCH -n 1 # number of cores",
        "#SBATCH --mem 3000 # memory pool for all cores",
        "#SBATCH -t 0-02:00 # time (D-HH:MM)",
        "#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT",
        "#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR",
        "",
        "module load Anaconda3/2019.10",
        "source activate schema",
        "cd ~/SchemaPrediction",
        "python -u job_v070820.py {kw_string} &> ./logs/{tag}c.log".format(kw_string=kw_string, tag=tag),
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
    
    # extensive testing shows that a good learning rate is an order of magnitiude arround 1e-3 
    lrs = [np.round(ii*10**-4,4) for ii in range(1, 10, 2)] + \
        [np.round(ii*10**-3,3) for ii in range(1, 11, 2)]

    # n_epochs_ = [ii for ii in range(4, 128, 4)]
    log_alphas = [-round(2**ii,1) for ii in np.arange(6, -.1, -1.0)] + \
        [0] + \
        [round(2**ii,1) for ii in np.arange(0, 5.1, 1.0)]
    log_lambdas = [int(ii) for ii in np.logspace(3.5, 7, base=2, num=8)]



    n_epochs_ = [int(round(2**ii)) for ii in np.arange(3, 7.1, 0.5)]
    n_batch = 1

    list_kwargs = []

    for no_split in [False]:
        for LSTM in [False]:
            for mixed in [False]:
                for epsilon in epsilons:
                    for lr in lrs:
                        for n_epochs in n_epochs_:
                            for log_alpha in log_alphas:
                                for log_lambda in log_lambdas:
                                    for b in range(n_batch):
                                        kwargs = dict(
                                            no_split=no_split,
                                            LSTM=LSTM,
                                            epsilon=epsilon,
                                            lr=lr,
                                            n_epochs=n_epochs,
                                            log_alpha=log_alpha,
                                            log_lambda=log_lambda,
                                            batch_n=b,
                                            batch_update=batch_update,
                                            actor_weight=0.0,
                                            mixed=mixed
                                        )
                                        list_kwargs.append(kwargs)

    log_alphas = [-208]
    log_lambdas = [208]
    
    for no_split in [True]:
        for LSTM in [False]:
            for mixed in [True]:
                for epsilon in epsilons:
                    for lr in lrs:
                        for n_epochs in n_epochs_:
                            for log_alpha in log_alphas:
                                for log_lambda in log_lambdas:
                                    for b in range(n_batch):
                                        kwargs = dict(
                                            no_split=no_split,
                                            LSTM=LSTM,
                                            epsilon=epsilon,
                                            lr=lr,
                                            n_epochs=n_epochs,
                                            log_alpha=log_alpha,
                                            log_lambda=log_lambda,
                                            batch_n=b,
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


    # # create the slurm submissions 
    for ii, kwargs in enumerate(list_kwargs):
        print('Submitting job {} of {}'.format(ii + 1, n))
        make_slurm_shell(kwargs, filename="_slurm.sh")

        os.system('sbatch _slurm.sh')
        time.sleep(0.25)
        os.remove('_slurm.sh')

