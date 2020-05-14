import numpy as np
import tensorflow as tf
import json
from sklearn import metrics

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from glob import glob

from AndreTask import generate_exp, batch_exp

import sys, os
# add the current path to python to allow loading SEM
current_path = os.path.abspath('.')
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from SEM2.core.sem import SEM
from SEM2.core.hrr import plate_formula, encode, decode, embed_gaussian
from SEM2.core.sem import SEM
from SEM2.core.event_models import GRUEvent, map_variance, NonLinearEvent_normed
from SEM2.core.utils import fast_mvnorm_diagonal_logprob, get_prior_scale


def main():
    # ## Story parameters

    n_batch = 1

    np.random.seed(0)

    seed = None
    err = 0.2; # error probability for plate's formula

    # story_generator = generate_exp_balanced
    story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)

    x, y, e, embedding_library = generate_exp('blocked', **story_kwargs)
    d = np.shape(x)[-1]
    print("Dimensions: {}".format(d))
    print("Median feature variance: {}".format(np.median(np.concatenate(x).var(axis=0))))


    # Testing suggests the NN parameters dominate the behavior, so we only worry about those

    # slow down learning in the network as much as possible
    # dropout           = 0.0
    l2_regularization = 0.0
    n_epochs          = 8
    batch_size        = 25

    # prior_log_prob = -2 ** 10 # a low enough likelihood will prevent any segementation
    # a high enough lamdba and low enough alpha will prevent any segementation
    alfa = 1e-308  
    lmda = 1e308  # this was as much as I could push lmda with out getting infinity

    f_class = GRUEvent

    optimizer_kwargs = dict(
        beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=False
    )


    f_opts=dict(
        batch_size=batch_size,
        l2_regularization=l2_regularization,
        # prior_log_prob=prior_log_prob,
        n_epochs=n_epochs,batch_update=True, 
        optimizer_kwargs=optimizer_kwargs)

    sem_kwargs = dict(alfa=alfa, lmda=lmda, f_class=f_class, f_opts=f_opts)


    dropouts = [0.0, 0.5]
    # lrs = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    lrs = [0.001, 0.0015, 0.002, 0.004, 0.008, 0.01,]
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

                    # set the parameters
                    optimizer_kwargs['lr'] = lr
                    optimizer_kwargs['epsilon'] = epsilon
                    f_opts['n_epochs'] = n_epochs
                    f_opts['dropout'] = dropout

                    for batch in range(n_batch):
                        
                        file_tag = '_e{}_lr{}_n{}_d{}_batch_{}'.format(
                                epsilon, lr, n_epochs, dropout, batch)

                        x, _, _, embedding_library = generate_exp('blocked', **story_kwargs)

                        from 

                        # for this test, we only look at the first block (40 stories)
                        x = x[:80]

                        # do a file check before running.
                        files = glob('./json_files/*{}*'.format(file_tag))
    
                        if not files:
                            print("running: {}".format(file_tag))
                            sem_model = SEM(**sem_kwargs)
                            sem_model.run_w_boundaries(x, save_x_hat=True, progress_bar=False)

                            # cache the original embedded vectors for the memory model
                            sem_model.results.x_orig = np.concatenate(x)
                            x = None

                            pe = np.linalg.norm(sem_model.results.x_hat - sem_model.results.x_orig,
                                            axis=1)

                            kk = 0  # counter for all of the scenes
                            json_pe = []
                            json_lp = []
                            for story0 in range(80):
                                for scene0 in range(6):
                                    json_pe.append({
                                        'lr': lr,
                                        'n_epochs': n_epochs,
                                        'epsilon': epsilon,
                                        'pe': pe[kk],
                                        'story': story0,
                                        'scene': scene0,
                                        'log_like_0': sem_model.results.scene_log_like[kk, 0],
                                        'log_like_1': sem_model.results.scene_log_like[kk, 1],
                                        'dropout': dropout,
                                        'batch': batch,
                                    })

                                    kk += 1

                                json_lp.append({
                                    'lr': lr,
                                    'n_epochs': n_epochs,
                                    'epsilon': epsilon,
                                    'log_post_0': sem_model.results.log_like[story0, 0] + 
                                        sem_model.results.log_prior[story0, 0],
                                    'log_post_1': sem_model.results.log_like[story0, 0] + 
                                        sem_model.results.log_prior[story0, 0],
                                    'story': story0,
                                    'dropout': dropout,
                                    'batch': batch,
                                })

                            # save results
                            with open('./json_files/nn_optimization_data_pe{}.json'.format(file_tag), 'w') as fp:
                                json.dump(json_pe, fp)

                            with open('./json_files/nn_optimization_data_lp{}.json'.format(file_tag), 'w') as fp:
                                json.dump(json_lp, fp)

                            
                            sem_model.clear()
                            sem_model = None

                            json_lp = None
                            json_pe = None

                        else: 
                            print("Skipping: {}".format(file_tag))


if __name__ == '__main__':
    main()

