import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn import metrics

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
from scipy.stats import norm
from glob import glob


"""## Define experiment"""
from schema_prediction_task import generate_exp, batch_exp

"""## Story parameters"""

seed = None
err = 0.01; # error probability for plate's formula

# story_generator = generate_exp_balanced
story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)


x, y, e, embedding_library = generate_exp('blocked', **story_kwargs)
d = np.shape(x)[-1]
print("Dimensions: {}".format(d))
print("Median feature variance: {}".format(np.median(np.concatenate(x).var(axis=0))))

"""# Other SEM Parameters

"""

# Testing suggests the NN parameters dominate the behavior, so we only worry
# about those

dropout           = 0.0
l2_regularization = 0.0
n_epochs          = 13
batch_size        = 25
lr                = 0.01
epsilon           = 1e-5
log_alpha         = 0.0
log_lambda        = 0.0

optimizer_kwargs = dict(
    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=False
)

f_opts=dict(
    dropout=dropout,
    batch_size=batch_size,
    l2_regularization=l2_regularization,
    n_epochs=n_epochs,batch_update=True,
    optimizer_kwargs=optimizer_kwargs)



def make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, mixed=False):
    # go through these lists in a random order for better sampling
    parameter_tuples = []
    for lr in lrs:
        for n_epochs in n_epochs_:
            for log_alpha in log_alphas:
                for log_lambda in log_lambdas:
                    parameter_tuples.append((lr, n_epochs, log_alpha, log_lambda))

    n = len(parameter_tuples)  # total number of simulations to be run

    # loop through the parameters in a random order, look for corresponding files,
    # and if none are found, add to the cue.
    parameters_queue = []
    for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameter_tuples):
        
        # # list of files that specify the simulation experiment exactly
        # args = [epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, int(batch_n)]
        # files = glob('./json_files/*{}*{}*{}*{}*{}*{}*{}*'.format(*args))

        args = [epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, int(batch_n)]
        file_string = './json_files/results*{}*{}*{}*{}*{}*{}*{}*'.format(*args)
        
        if mixed:
            file_string += 'mixed*'

        files = glob(file_string)

        if not files:
            parameters_queue.append((lr, int(n_epochs), log_alpha, log_lambda))

    t = len(parameters_queue)
    print('Found {} of {} simulations previous completed'.format(n-t, n))
    return parameters_queue


def run_single_batch(batch_n, lr, n_epochs, log_alpha, log_lambda, mixed=False, tag='', no_split=False):
    # sem prior params (dosen't rely on mutability)
    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts)

    f_opts['n_epochs'] = int(n_epochs)
    optimizer_kwargs['lr'] = lr # this relies on mutability to change the f_opts dictionary

    json_tag = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n, tag)
    json_tag_mixed = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}_mixed'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n, tag)

    if not mixed:
        print(json_tag)
        _, _, _ = batch_exp(
            sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
            progress_bar=False, block_only=False,
            run_mixed=False,
            save_to_json=True,
            json_tag=json_tag,
            json_file_path='./json_files/', 
            no_split=no_split)
    else:
        print(json_tag_mixed)
        _, _, _ = batch_exp(
            sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
            progress_bar=False, block_only=False,
            run_mixed=True,
            save_to_json=True,
            json_tag=json_tag_mixed,
            json_file_path='./json_files/', 
            no_split=no_split)

# lrs = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.008]
# n_epochs_ = [12, 16, 20, 24, 28, 32, 36, 40, 44]
# log_alphas = [0.0, -5.0, -2.5,  2.5, 5.0, 7.5, -7.5, -10.0, -15.0]
# log_lambdas = [0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0]


if __name__ == "__main__":
    import sys
    try:
        batch_n = int(sys.argv[1])
        "Running Batch {}:".format(batch_n)
    except:
        print('Please specify batch number!')
        raise(Exception)

    mixed=False

    def make_batch_runner_queue(parameters_queue, mixed=False, tag='', no_split=False):
        
        list_functions = []
        
        for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameters_queue):
            
            def internal_func():

                args = [batch_n, lr, int(n_epochs), log_alpha, log_lambda]
                kw_args = dict(mixed=mixed, tag=tag, no_split=no_split)
                
                return run_single_batch(*args, **kw_args)

            list_functions.append(internal_func)
        
        return list_functions

    parameters_queue = []

    lrs = [0.002]
    n_epochs_ = [36, 44, 60]
    log_alphas = [0.0, -15.0, -25.0, -35.0]
    log_lambdas = [0.0, 20.0, 40.0, 60.0]
    parameters_queue_mixed = make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, mixed=True)
    batch_runner_queue = make_batch_runner_queue(parameters_queue, mixed=True)

    # # let's look also at a completely different and unoverlapping portion of the parameter space
    lrs = [0.002]
    n_epochs_ = [36, 44, 60]
    log_alphas = [0.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0]
    log_lambdas = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0]
    parameters_queue += make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas)
    batch_runner_queue = make_batch_runner_queue(parameters_queue)

    # let's look also at a completely different and unoverlapping portion of the parameter space
    lrs = [0.00005, 0.0001, 0.0005, 0.002, 0.004, 0.008]
    n_epochs_ = [4, 8, 12, 16, 20, 28, 36, 44, 60]
    log_alphas = [-101]
    log_lambdas = [101]
    parameters_queue += make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas)
    batch_runner_queue = make_batch_runner_queue(parameters_queue, tag='_no_split', no_split=True)
    
    t = 0
    n = len(batch_runner_queue)
    for batch_runner in np.random.permutation(batch_runner_queue):
        print("Running simulation {} of {}".format(t, n))
        batch_runner()
        t += 1


    # t = 0
    # n = len(parameters_queue)
    # for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameters_queue):
    #     print("Running simulation {} of {}".format(t, n))
    #     run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed)
    #     t += 1
