import numpy as np
from schema_prediction_task_6_02_20 import generate_exp, batch_exp
from sem.event_models import GRUEvent, LSTMEvent
import time, tqdm, os

output_file_path = './json_files_v060220/'


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
        "#SBATCH -t 0-01:00 # time (D-HH:MM)",
        "#SBATCH -o slurm_output/slurm.%N.%j.out # STDOUT",
        "#SBATCH -e slurm_output/slurm.%N.%j.err # STDERR",
        "",
        "module load Anaconda3/2019.10",
        "conda create -n schema",
        "source activate schema",
        "cd ~/SchemaPrediction",
        "pip install -r requirements.txt",
        "python -u job_v060220.py {kw_string} &> ./logs/{tag}c.log".format(kw_string=kw_string, tag=tag),
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

    half_powers_of_two = \
        [-round(2**ii,1) for ii in np.arange(7, -.1, -0.5)] + \
        [0] + \
        [round(2**ii,1) for ii in np.arange(0, 7.1, 0.5)]

    lrs = [0.005]
    n_epochs_ = [32]
    log_alphas = half_powers_of_two
    log_lambdas = half_powers_of_two

    n_batch = 8

    list_kwargs = []

    for no_split in [False]:
        for LSTM in [False]:
            for n_hidden in [20]:
                for lr in lrs:
                    for n_epochs in n_epochs_:
                        for log_alpha in log_alphas:
                            for log_lambda in log_lambdas:
                                for b in range(n_batch):
                                    kwargs = dict(
                                        no_split=no_split,
                                        n_hidden=n_hidden,
                                        LSTM=LSTM,
                                        lr=lr,
                                        n_epochs=n_epochs,
                                        log_alpha=log_alpha,
                                        log_lambda=log_lambda,
                                        batch_n=b,
                                    )
                                    list_kwargs.append(kwargs)
            #                     break
            #                 break
            #             break
            #         break
            #     break
            # break
        # break


    # randomize the simulation order for effective sampling speed 
    # (i.e. intermediate progress is more meaningful)
    list_kwargs = np.random.permutation(list_kwargs)
    n = len(list_kwargs)
    # print(list_kwargs)
    print(n)

    # create the slurm submissions 
    for ii, kwargs in enumerate(list_kwargs):
        print('Submitting job {} of {}'.format(ii + 1, n))
        make_slurm_shell(kwargs, filename="_slurm.sh")

        os.system('sbatch _slurm.sh')
        time.sleep(0.25)
        # os.remove('_slurm.sh')


