#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from CSWSEM import generate_exp, seed_exp
from vanilla_lstm import VanillaLSTM
from sem.event_models import NonLinearEvent


# ### gridsearch setup - params and model tag

# In[2]:

# gs params
# model_type        = str(sys.argv[1])
# lr                = float(sys.argv[3])
# n_epochs          = int(sys.argv[4])    
# log_alpha         = float(sys.argv[5])  
# log_lambda        = float(sys.argv[6])  

model_type  = str(sys.argv[1])
seed        = int(sys.argv[2])
param_str   = str(sys.argv[3])
## parse input param string
n_epochs, lr, log_alpha, log_lambda = param_str.split()
n_epochs    = int(n_epochs)
lr          = float(lr)
log_alpha   = float(log_alpha)
log_lambda  = float(log_lambda)
seed        = int(seed)


## other vars
n_train = 160
n_test = 40

gs_name = 'sem1'
save_dir = 'gsdata/%s/'%gs_name


## param dict setup

optimizer_kwargs = dict(
    lr=lr, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-5, 
    amsgrad=False
) 

f_opts=dict(
  batch_size=25, 
  batch_update=False, 
  dropout=0.0,
  l2_regularization=0.0, 
  n_epochs=n_epochs,
  optimizer_kwargs=optimizer_kwargs
)

sem_kwargs = dict(
  lmda=np.exp(log_lambda), 
  alfa=np.exp(log_alpha), 
  f_opts=f_opts, 
  f_class=VanillaLSTM
)


## condition loop

for condition in ["blocked","interleaved"]:
  ## condition name
  model_tag = "%s_cond_%s_lr_%.3f_nepchs_%i_alpha_%.3f-lambda_%.3f_seed_%i"%(
    model_type,condition,lr,n_epochs,log_alpha,log_lambda,seed
  )
  print('\n\n -- BEGIN',model_tag)

  # RUN MODEL
  results, trialXtrial, _ = seed_exp( 
                sem_kwargs, model_type=model_type, 
                n_train=n_train, n_test=n_test,
                condition=condition,seed=seed,
  )

  ## save
  results_fpath = save_dir + "results_" + model_tag + '.csv'
  trial_fpath = save_dir + "trial_X_trial_" + model_tag + '.csv'

  results = pd.DataFrame(results)
  trialXtrial = pd.DataFrame(trialXtrial)

  results.to_csv(results_fpath)
  trialXtrial.to_csv(trial_fpath)


  print('\n\n -- FINISH',model_tag)

