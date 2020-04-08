import numpy as np
import tensorflow as tf
import json
from sklearn import metrics

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
from scipy.stats import norm

from glob import glob

from schema_prediction_task import generate_exp, classify_verbs

from sem import sem_run_with_boundaries


def main():
    # ## Story parameters

    n_batch = 1
    
    seed = None
    err = 0.01; # error probability for plate's formula

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
    # alfa = 1e-308  
    # lmda = 1e308  # this was as much as I could push lmda with out getting infinity
    log_alpha = 0
    log_lambda = 0

    optimizer_kwargs = dict(
        beta_1=0.9, beta_2=0.999, epsilon=1e-05, amsgrad=False
    )


    f_opts=dict(
        batch_size=batch_size,
        l2_regularization=l2_regularization,
        # prior_log_prob=prior_log_prob,
        n_epochs=n_epochs,batch_update=True, 
        optimizer_kwargs=optimizer_kwargs)

    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts)


    def run_single_batch(epsilon, lr, n_epochs, dropout, batch=0):

        # set the parameters
        optimizer_kwargs['lr'] = lr
        optimizer_kwargs['epsilon'] = epsilon
        f_opts['n_epochs'] = int(n_epochs)
        f_opts['dropout'] = dropout

            
        file_tag = '_e{}_lr{}_n{}_d{}_logalfa{}_loglmda{}_batch_{}'.format(
                epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch)

        x, y, e, embedding_library = generate_exp('blocked', **story_kwargs)

        # for this test, we only look at the first block (40 stories)
        x = x[:80]

        # do a file check before running.
        files = glob('./json_files/*{}*'.format(file_tag))

        if not files:
            print("running: {}".format(file_tag))

            run_kwargs = dict(save_x_hat=True, progress_bar=True, minimize_memory=True)

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
                        'verb 2 AFC Probb': float(prob_corr_2afc[kk]),
                    })

                    kk += 1

                json_lp.append({
                    'lr': lr,
                    'n_epochs': n_epochs,
                    'e_hat': int(results.e_hat[story0]),
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

    dropouts = [0.0,]
    lrs = [0.002, 0.004, 0.008, 0.01, 0.02]
    n_epochs_list = [12, 14, 16, 18, 20, 22, 24, 26, 26, 28, 30, 34, 38, 42]
    epsilons = [1e-6, 5e-6, 1e-5]

    n_batch = 3
    t = 0

    for batch_n in range(n_batch):

        # go through these lists in a random order for better sampling
        parameter_tuples = []
        for epsilon in epsilons:
            for lr in lrs:
                for n_epochs in n_epochs_list:
                    for dropout in dropouts:
                        parameter_tuples.append((epsilon, lr, n_epochs, dropout))

        for epsilon, lr, n_epochs, dropout in np.random.permutation(parameter_tuples):
            run_single_batch(epsilon, lr, n_epochs, dropout, batch=batch_n)

    


if __name__ == '__main__':
    main()

