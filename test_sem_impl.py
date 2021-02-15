#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
import pandas as pd

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
from scipy.stats import norm
from glob import glob

from cswTASK import generate_exp, seed_exp
from cswRNN import CSWEvent


model_type        = str('SEM')
lr                = float(0.05)
n_epochs          = int(1)   
seed = int(99)
condition = 'single'
n_train = 8
n_test = 2

optimizer_kwargs = dict(
    lr=lr, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-5, 
    amsgrad=False
) 

f_opts=dict(
  batch_size=1, 
  batch_update=False, 
  dropout=0.0,
  l2_regularization=0.0, 
  n_epochs=n_epochs,
  optimizer_kwargs=optimizer_kwargs
)

# alpha ~ pr new cluster
# lmbda ~ pr reuse
def get_params(case):
    if case==0:
        log_alpha         = int(24)  # sCRP alpha is set in log scale
        log_lambda        = int(15)  # sCRP lambda is set in log scale
    if case==1:
        log_alpha         = int(10)  # sCRP alpha is set in log scale
        log_lambda        = int(5)  # sCRP lambda is set in log scale
    if case==2:
        log_alpha         = int(17)  # sCRP alpha is set in log scale
        log_lambda        = int(-10)  # sCRP lambda is set in log scale
    if case==3:
        log_alpha         = int(16)  # sCRP alpha is set in log scale
        log_lambda        = int(-8)  # sCRP lambda is set in log scale
    return log_alpha,log_lambda


for case in range(1):
    print('\n\n**CASE%i\n'%case)
    log_alpha,log_lambda = get_params(case)

    sem_kwargs = dict(
      lmda=np.exp(log_lambda), 
      alfa=np.exp(log_alpha), 
      f_opts=f_opts, 
      f_class=CSWEvent
    )


    for model_type in ['SEM']:
      print('---',model_type)
      results, trialXtrial, _ = seed_exp( 
                    sem_kwargs, model_type=model_type, 
                    n_train=n_train, n_test=n_test,
                    condition=condition,seed=seed,
      )
