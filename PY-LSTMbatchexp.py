#!/usr/bin/env python
# coding: utf-8

# This script runs one param_set. Outputs results{}.csv and trialxtrial{}.csv

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


# In[2]:


from CSWSEM import generate_exp, seed_exp
from vanilla_lstm import VanillaLSTM
from sem.event_models import NonLinearEvent


# ### gridsearch params 

# In[3]:


# parameter search over lr, n_epochs, alpha, lambda
model_type        = str('LSTM')
seed              = int(sys.argv[1])
lr                = float(sys.argv[2])
n_epochs          = int(sys.argv[3])    


# In[4]:


log_alpha         = float(0.0)  # sCRP alpha is set in log scale
log_lambda        = float(0.0)  # sCRP lambda is set in log scale


# In[5]:


condition = 'blocked'
n_train = 160
n_test = 40


# In[6]:


save_dir = 'gsdata/lstm2_aw0/'
model_tag = "%s_cond_%s_lr_%.3f_nepchs_%i_alpha_%.3f-lambda_%.3f_seed_%i"%(
  model_type,condition,lr,n_epochs,log_alpha,log_lambda,seed
)
print(model_tag)


# ### SEM configuration

# In[7]:


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

# final param dict
sem_kwargs = dict(
  lmda=np.exp(log_lambda), 
  alfa=np.exp(log_alpha), 
  f_opts=f_opts, 
  f_class=VanillaLSTM
)


# # Run model
# 
# main fun call

# In[ ]:


""" 
main fun call
"""
stories_kwargs = {'actor_weight':0.0}
results, trialXtrial, _ = seed_exp( 
              sem_kwargs, stories_kwargs=stories_kwargs,
              model_type=model_type, 
              n_train=n_train, n_test=n_test,
              condition=condition,seed=seed,
)


# In[ ]:


# convert from JSON file format (dict) to pandas df
results = pd.DataFrame(results)
trialXtrial = pd.DataFrame(trialXtrial)


# # save
# 

# In[ ]:


results_fpath = save_dir + "results_" + model_tag + '.csv'
trial_fpath = save_dir + "trial_X_trial_" + model_tag + '.csv'


# In[ ]:


results.to_csv(results_fpath)
trialXtrial.to_csv(trial_fpath)

