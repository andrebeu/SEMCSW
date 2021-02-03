#!/usr/bin/env python
# coding: utf-8

# This script runs one param_set. Outputs results{}.csv and trialxtrial{}.csv

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

from schema_prediction_task_9_8_20 import generate_exp, batch_exp
from vanilla_lstm import VanillaLSTM
from sem.event_models import NonLinearEvent

### simulation information

save_dir = 'data/gridsearch_full1/'


# parameter search over lr, n_epochs, alpha, lambda
model_type        = str(sys.argv[1])
lr                = float(sys.argv[2])
n_epochs          = int(sys.argv[3])    # what does this control?
log_alpha         = float(sys.argv[4])  # sCRP alpha is set in log scale
log_lambda        = float(sys.argv[5])  # sCRP lambda is set in log scale


model_tag = "%s-lr-%.3f-nepchs-%i-alpha-%.3f-lambda-%.3f"%(
  model_type,lr,n_epochs,log_alpha,log_lambda
)
print(model_tag)

# In[4]:

conditions = ['interleaved','blocked','early','middle','late']
n_batch = 50


# ### SEM configuration


# number of trials
n_train = 160
n_test = 40

story_kwargs = dict(seed=None, err=0.2, actor_weight=1.0, instructions_weight=0.0)

optimizer_kwargs = dict(
    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-5, amsgrad=False
) 

f_opts=dict(
  batch_size=25, 
  batch_update=False, 
  dropout=0.0,
  l2_regularization=0.0, 
  n_epochs=n_epochs,
    optimizer_kwargs=optimizer_kwargs
)

f_class = VanillaLSTM # event model class

# final param dict
sem_kwargs = dict(
  lmda=np.exp(log_lambda), 
  alfa=np.exp(log_alpha), 
  f_opts=f_opts, 
  f_class=f_class
)


# toggle between SEM (False) and LSTM (True)
if model_type == 'SEM':
  no_split=False
elif model_type == 'LSTM':
  no_split=True


""" 
batch_exp main fun call
"""

results, trialXtrial, _ = batch_exp(
              sem_kwargs, story_kwargs, no_split=no_split, 
              sem_progress_bar=False, progress_bar=False,
              n_train=n_train, n_test=n_test,
              n_batch=n_batch, conditions=conditions
)

# convert from JSON file format (dict) to pandas df
results = pd.DataFrame(results)
trialXtrial = pd.DataFrame(trialXtrial)


## save
results_fpath = save_dir + "results_" + model_tag + '.csv'
trial_fpath = save_dir + "trial_X_trial_" + model_tag + '.csv'


results.to_csv(results_fpath)
trialXtrial.to_csv(trial_fpath)

