import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import json

from tqdm import tqdm
from sklearn import metrics

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm


# from memory_profiler import profile

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
from SEM2.core.utils import fast_mvnorm_diagonal_logprob, get_prior_scale, processify


# we will pre-define all of the functions so they take no arguments and make 
# them have a single profiler line
def gen_exp():

    np.random.seed(0)

    seed = None
    err = 0.2; # error probability for plate's formula

    # story_generator = generate_exp_balanced
    story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)

    return generate_exp('blocked', **story_kwargs)

@processify
def init_run(x, sem_kwargs):
    sem_model = SEM(**sem_kwargs)
    sem_model.run_w_boundaries(x, save_x_hat=True, progress_bar=False)
    return sem_model

@processify
def analyses(sem_model):
    
    pe = np.linalg.norm(sem_model.results.x_hat - sem_model.results.x_orig,
                    axis=1)

    kk = 0  # counter for all of the scenes
    for story0 in range(80):
        for scene0 in range(5):
            df_pe = {
                'lr': 0.008,
                'n_epochs': 12,
                'epsilon': 1e-5,
                'pe': pe[kk],
                'story': story0,
                'scene': scene0,
                'log_like_0': sem_model.results.scene_log_like[kk, 0],
                'log_like_1': sem_model.results.scene_log_like[kk, 1],
                'dropout': 0.0,
            }

            kk += 1

        df_lp = {
            'lr': 0.008,
            'n_epochs': 12,
            'epsilon': 1e-5,
            'log_post_0': sem_model.results.log_like[story0, 0] + 
                sem_model.results.log_prior[story0, 0],
            'log_post_1': sem_model.results.log_like[story0, 0] + 
                sem_model.results.log_prior[story0, 0],
            'story': story0,
            'dropout': 0.0,
        }
    return df_pe, df_lp

def main():

    n_batch = 5

    # ## Story parameters

    # x, _, _, _ = gen_exp()
    # d = np.shape(x)[-1]
    # print("Dimensions: {}".format(d))
    # print("Median feature variance: {}".format(np.median(np.concatenate(x).var(axis=0))))

    # set all of the parameters for SEM
    l2_regularization = 0.0
    batch_size        = 25
    n_epochs = 12
    dropout =0.0

    # prior_log_prob = -2 ** 10 # a low enough likelihood will prevent any segementation
    # a high enough lamdba and low enough alpha will prevent any segementation
    alfa = 1e-308  
    lmda = 1e308  # this was as much as I could push lmda with out getting infinity

    f_class = GRUEvent
    lr = 0.008
    epsilon = 1e-5
    optimizer_kwargs = dict(
        beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=False, lr=lr,
    )

    f_opts=dict(
        batch_size=batch_size,
        l2_regularization=l2_regularization,
        n_epochs=n_epochs, batch_update=True, 
        dropout=dropout,
        optimizer_kwargs=optimizer_kwargs)

    sem_kwargs = dict(alfa=alfa, lmda=lmda, f_class=f_class, f_opts=f_opts)

    df_pe = []
    df_lp = []
    
    # batch_label = 'Batch (epsilon={}, lr={}, n_epochs={}, dropout={})'.format(
    #     epsilon, lr, n_epochs, dropout)

    # set the parameters

    for batch in range(n_batch):
        
        x, _, _, _ = gen_exp()

        # for this test, we only look at the first block (40 stories)
        x = x[:80]


        sem_model = init_run(x, sem_kwargs)

        # cache the original embedded vectors for the memory model
        sem_model.results.x_orig = np.concatenate(x)

        _lp, _pe = analyses(sem_model)
        df_lp.append(_lp)
        df_pe.append(_pe)
        sem_model.clear()
        sem_model = None

        # save the intermediate results
        with open('mem_test_pe.json', 'w') as fp:
            json.dump(df_pe, fp)
        with open('mem_test_lp.json', 'w') as fp:
            json.dump(df_lp, fp)


if __name__ == '__main__':
    main()
        


