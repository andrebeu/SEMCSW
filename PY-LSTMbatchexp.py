#!/usr/bin/env python
# coding: utf-8

# This script runs one param_set. Outputs results{}.csv and trialxtrial{}.csv

# In[1]:


import sys
# print("Python version")
# print (sys.version)

# %matplotlib inline
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


# ### gridsearch params 


# param search over
seed              = int(sys.argv[1])
lr                = float(sys.argv[2])
n_epochs          = int(sys.argv[3])   

# model params 
model_type        = str('LSTM')
log_alpha         = float(0.0)  # sCRP alpha is set in log scale
log_lambda        = float(0.0)  # sCRP lambda is set in log scale

# taks params
condition = 'single'
n_train = 200
n_test = 10



save_dir = 'gsdata/lstm1/'
model_tag = "%s_cond_%s_lr_%.3f_nepchs_%i_alpha_%.3f-lambda_%.3f_seed_%i"%(
  model_type,condition,lr,n_epochs,log_alpha,log_lambda,seed
)
print(model_tag)


# ### SEM configuration

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


# # Run model
# 
# main fun call

# In[6]:


""" 
main fun call
"""

results, trialXtrial, _ = seed_exp( 
              sem_kwargs, model_type=model_type, 
              n_train=n_train, n_test=n_test,
              condition=condition,seed=seed,
)


# In[7]:


# convert from JSON file format (dict) to pandas df
results = pd.DataFrame(results)
trialXtrial = pd.DataFrame(trialXtrial)


# # save
# 

# In[10]:


results_fpath = save_dir + "results_" + model_tag + '.csv'
trial_fpath = save_dir + "trial_X_trial_" + model_tag + '.csv'


# In[11]:


results.to_csv(results_fpath)
trialXtrial.to_csv(trial_fpath)

