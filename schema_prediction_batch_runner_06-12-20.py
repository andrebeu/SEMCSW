import numpy as np
from schema_prediction_task_6_12_20 import generate_exp, batch_exp
import time, tqdm, os

output_file_path = './json_files_v061220/'


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
        "python -u job_v061220.py {kw_string} &> ./logs/{tag}c.log".format(kw_string=kw_string, tag=tag),
        "sleep 10",
    ]

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()

    return 


if __name__ == "__main__":

    ## these are the keywords for the jobs script
    # n_epochs=28,batch_size=25,lr=0.001,log_alpha=0.0,
    # log_lambda=0.0, n_hidden=10, epsilon=1e-5, no_split=False,
    #  batch_n=0, LSTM=False, batch_update=True

    # # power_scale = \
    # #     [-round(2**ii,1) for ii in np.arange(7, -.1, -1.0)] + \
    # #     [0] + \
    # #     [round(2**ii,1) for ii in np.arange(0, 7.1, 1.0)]
    # n_hidden_ = [10, 40]
    # epsilons = [10**-ii for ii in range(10,4,-1)]
    #     # lrs = [3e-3, 4e-5, 5e-3, 6e-5, 7.5e-3]
    # # lrs = [2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2]
    # # n_epochs_ = [4, 8, 16, 32, 64, 128]
    # lrs = [np.round(ii*10**-4,4) for ii in range(1, 10)] + \
    #     [np.round(ii*10**-3,3) for ii in range(1, 10)] + \
    #     [np.round(ii*10**-2,2) for ii in range(1, 11)]
    # n_epochs_ = [ii for ii in range(4, 128, 4)]
    # # log_alphas = [-round(2**ii,1) for ii in np.arange(5, -.1, -1.0)] + \
    # #     [0] + \
    # #     [round(2**ii,1) for ii in np.arange(0, 3.1, 1.0)]
    # # log_lambdas = [ii for ii in np.arange(12, 74, 2)]
    # log_alphas = [-208]
    # log_lambdas = [208]


    # # n_hidden_ = [10,]
    # epsilons = [1e-9, 1e-7, 1e-5, 1e-3]
    # lrs = [1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2]
    # n_epochs_ = [round(2**ii,1) for ii in np.arange(1, 7.1, 1.0)]
    # log_alphas = [-round(2**ii,1) for ii in np.arange(5, -.1, -1.0)] + \
    #     [0] + \
    #     [round(2**ii,1) for ii in np.arange(0, 3.1, 1.0)]
    # log_lambdas = [0] + \
    #     [round(2**ii,1) for ii in np.arange(0, 7.1, 1.0)]

    # n_batch = 1
    # batch_update = True

    list_kwargs = []

    # for no_split in [False]:
    #     for LSTM in [False]:
    #         for epsilon in epsilons:
    #             for lr in lrs:
    #                 for n_epochs in n_epochs_:
    #                     for log_alpha in log_alphas:
    #                         for log_lambda in log_lambdas:
    #                             for b in range(n_batch):
    #                                 kwargs = dict(
    #                                     no_split=no_split,
    #                                     LSTM=LSTM,
    #                                     epsilon=epsilon,
    #                                     lr=lr,
    #                                     n_epochs=n_epochs,
    #                                     log_alpha=log_alpha,
    #                                     log_lambda=log_lambda,
    #                                     batch_n=b,
    #                                     batch_update=batch_update,
    #                                 )
    #                                 list_kwargs.append(kwargs)


    epsilons = [1e-7]
    lrs = [8e-2, 9e-2, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]
    n_epochs_ = [ii for ii in range(5, 100, 5)]
    log_alphas = [-208]
    log_lambdas = [208]

    n_batch = 16
    batch_update = True

    for no_split in [True]:
        for LSTM in [False]:
            # for n_hidden in n_hidden_:
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
                                    )
                                    list_kwargs.append(kwargs)


    epsilons = [1e-7]
    lrs = [8e-2, 9e-2, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]
    n_epochs_ = [ii for ii in range(5, 100, 5)]
    log_alphas = [-208]
    log_lambdas = [208]

    n_batch = 16
    batch_update = False

    for no_split in [True]:
        for LSTM in [False]:
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
                                    )
                                    list_kwargs.append(kwargs)

    # randomize the simulation order for effective sampling speed 
    # (i.e. intermediate progress is more meaningful)
    list_kwargs = np.random.permutation(list_kwargs)
    n = len(list_kwargs)
    # print(list_kwargs)
    for kwarg in list_kwargs:
        print(make_kw_string(kwarg) + '\n' )
    print(n)

    # # # create the slurm submissions 
    # for ii, kwargs in enumerate(list_kwargs):
    #     print('Submitting job {} of {}'.format(ii + 1, n))
    #     make_slurm_shell(kwargs, filename="_slurm.sh")

    #     os.system('sbatch _slurm.sh')
    #     time.sleep(0.25)
    #     os.remove('_slurm.sh')


