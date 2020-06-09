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
from schema_prediction_task_5_12_20 import generate_exp, batch_exp

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
log_alpha         = 0.0
log_lambda        = 0.0

beta_1            = 0.9
beta_2            = 0.999
lr                = 0.001
epsilon           = 1e-8

optimizer_kwargs = dict(lr=lr, beta_1=beta_1, beta_2=beta_2,
    epsilon=epsilon, amsgrad=False)

f_opts=dict(
    dropout=dropout,
    batch_size=batch_size,
    l2_regularization=l2_regularization,
    n_epochs=n_epochs,batch_update=True,
    optimizer_kwargs=optimizer_kwargs)


def make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, tag='', mixed=False, epsilon=1e-5):
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

        json_tag = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}'.format(
            epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, batch_n, tag)
        json_tag_mixed = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}_mixed'.format(
            epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, batch_n, tag)

        file_string = './json_files/results{}.json'.format(json_tag)
        
        if mixed:
            file_string = './json_files/results{}.json'.format(json_tag_mixed)

        files = glob(file_string)

        if not files:
            parameters_queue.append((lr, int(n_epochs), log_alpha, log_lambda))

    t = len(parameters_queue)
    print('Found {} of {} simulations previous completed'.format(n-t, n))
    return parameters_queue


def run_single_batch(batch_n, lr, n_epochs, log_alpha, log_lambda, mixed=False, tag='', 
                    no_split=False, batch_update=True):

    # sem prior params (dosen't rely on mutability)
    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts)

    f_opts['n_epochs'] = int(n_epochs)
    optimizer_kwargs['lr'] = lr # this relies on mutability to change the f_opts dictionary

    # run batch update or online training?
    f_opts['batch_update'] = batch_update

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
    no_split=True

    lrs = [1e-3]
    n_epochs_ = [16, 32, 64, 128]
    # log_alphas = [-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
    # log_lambdas = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    log_alphas = [0]
    log_lambdas = [0]

    

    # lrs = [1e-5]
    # n_epochs_ = [1]
    # log_alphas = [0.0]
    # log_lambdas = [0.0]

    # Online training only
    tag='_online_nosplit'
    no_split=True
    batch_update = False
    parameters_queue = make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, tag=tag, mixed=mixed, epsilon=epsilon)
    t = 0
    n = len(parameters_queue)
    for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameters_queue):
        print("Running simulation {} of {}".format(t, n))
        run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed,
             no_split=no_split, tag=tag, batch_update=batch_update)
        t += 1


    # # here, we'll just use a single parameter queue and assume run everything together
    
    # tag='_online'
    # parameters_queue = make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, tag=tag, mixed=mixed)
    # t = 0
    # n = len(parameters_queue)
    # for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameters_queue):
        
    #     print("Running simulations {} of {}".format(t, n))

    #     #  batch training
    #     no_split=False
    #     tag=''
    #     n = len(parameters_queue)
    #     run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed,
    #         no_split=no_split, tag=tag)


    #     # no splitting, batch training
    #     no_split=True
    #     tag='_nosplit'
    #     n = len(parameters_queue)
    #     run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed,
    #         no_split=no_split, tag=tag)

    #     # with splitting, online training   
    #     no_split=False
    #     tag='_online'
    #     run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed,
    #         no_split=no_split, tag=tag, batch_update=False)



    #     # with splitting, online training   
    #     no_split=True
    #     tag='_online_nosplit'
    #     run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=mixed,
    #         no_split=no_split, tag=tag, batch_update=False)
  
    #     t += 1