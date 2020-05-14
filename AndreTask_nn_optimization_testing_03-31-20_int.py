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

from AndreTask import generate_exp, classify_verbs

import sys, os
# add the current path to python to allow loading SEM
current_path = os.path.abspath('.')
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from SEM2.core.sem import sem_run_with_boundaries


def main():
    # ## Story parameters

    n_batch = 1
    
    seed = None
    err = 0.01; # error probability for plate's formula

    # story_generator = generate_exp_balanced
    story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)

    x, y, e, embedding_library = generate_exp('interleaved', **story_kwargs)
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


    optimizer_kwargs = dict(
        beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=False
    )


    f_opts=dict(
        batch_size=batch_size,
        l2_regularization=l2_regularization,
        # prior_log_prob=prior_log_prob,
        n_epochs=n_epochs,batch_update=True, 
        optimizer_kwargs=optimizer_kwargs)

    sem_kwargs = dict(alfa=alfa, lmda=lmda,  f_opts=f_opts)


    dropouts = [0.0, 0.5]
    lrs = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02]
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
                        
                        file_tag = '_e{}_lr{}_n{}_d{}_batch_{}_low_err_int'.format(
                                epsilon, lr, n_epochs, dropout, batch)

                        x, y, _, embedding_library = generate_exp('interleaved', **story_kwargs)

                        # for this test, we only look at the first block (40 stories)
                        x = x[:80]

                        # do a file check before running.
                        files = glob('./json_files/*{}*'.format(file_tag))
    
                        if not files:
                            print("running: {}".format(file_tag))

                            run_kwargs = dict(save_x_hat=True, progress_bar=True, 
                                minimize_memory=True, aggregator=np.sum)

                            results = sem_run_with_boundaries(x, sem_kwargs, run_kwargs)
                            results.x_orig = np.concatenate(x)

                            # create a decoder based on both the training stimuli from both experiments
                            decoding_acc, decoding_prob_corr, prob_corr_2afc = classify_verbs(results, y[:480])

                            pe = np.linalg.norm(results.x_hat - results.x_orig, axis=1)

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
                                        'log_like_0': results.scene_log_like[kk, 0],
                                        'log_like_1': results.scene_log_like[kk, 1],
                                        'dropout': dropout,
                                        'batch': batch,
                                        'verb Accuracy': float(decoding_acc[kk]),
                                        'verb Accuracy Prob': float(decoding_prob_corr[kk]),
                                        'verb 2 AFC Prob': float(prob_corr_2afc[kk]),
                                    })

                                    kk += 1

                                json_lp.append({
                                    'lr': lr,
                                    'n_epochs': n_epochs,
                                    'epsilon': epsilon,
                                    'log_post_0': results.log_like[story0, 0] + 
                                        results.log_prior[story0, 0],
                                    'log_post_1': results.log_like[story0, 0] + 
                                        results.log_prior[story0, 0],
                                    'story': story0,
                                    'dropout': dropout,
                                    'batch': batch,
                                })

                            # save results
                            with open('./json_files/nn_optimization_data_pe{}.json'.format(file_tag), 'w') as fp:
                                json.dump(json_pe, fp)

                            with open('./json_files/nn_optimization_data_lp{}.json'.format(file_tag), 'w') as fp:
                                json.dump(json_lp, fp)

                            # save stimuli for debuggin
                            json_stims = {
                                'x': [x0.tolist() for x0 in x],
                                'embeddings': {k: v.tolist() for k, v in embedding_library.items()},
                                'y': y.tolist(),
                                'e': e
                            }
                            with open('./json_files/nn_optimization_stims{}.json'.format(file_tag), 'w') as fp:
                                json.dump(json_stims, fp)

                        
                            json_lp = None
                            json_pe = None

                        else: 
                            print("Skipping: {}".format(file_tag))


if __name__ == '__main__':
    main()

