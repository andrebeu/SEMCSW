#!/usr/bin/env python
# coding: utf-8

import sys
import itertools
import os
import numpy as np
import torch as tr
import pandas as pd

from CSWSEM import *


## input params
gs_name = 'gs-acc'

## input params
param_str = str(sys.argv[1])
learn_rate,alfa,lmda = param_str.split()

learn_rate = float(learn_rate)
alfa = float(alfa)
lmda = float(lmda)

stsize = 25



num_seeds = 50
condL = ['blocked','interleaved']

exp_kwargs={
    'n_train':160,
    'n_test':40
}

# NB only SEM - no LSTM
sem_kwargs={
    'nosplit':False,
    'stsize':stsize,
    'lmda':lmda,
    'alfa':alfa,
    'learn_rate':learn_rate,
}


dataL = []
exphat = -np.ones([num_seeds,len(condL),200,5,10]) # (trials,tsteps,obsdim)

## loop over seeds and conditions (Blocked/interleaved)

for seed in np.arange(num_seeds):
  for cidx,cond in enumerate(condL):
    print(seed,cond)
    np.random.seed(seed)
    tr.manual_seed(seed)
    # setup
    sem_kwargs['seed'] = seed
    exp_kwargs['condition'] = cond
    task = CSWTask(seed)
    sem = SEM(**sem_kwargs)
    # run
    exp,curr = task.generate_experiment(**exp_kwargs)
    sem_data = sem.forward_exp(exp,curr)
    ### eval
    exphat[seed,cidx] = np.array([tdata['event_hat'] for tdata in sem_data])
    # record data
    sem.data.record_exp('condition',exp_kwargs['condition'])
    dataL.append(pd.DataFrame(sem_data))


## save

save_fpath = 'gsdata/%s/'%(gs_name)
batch_tag = 'learnrate_%.3f__alfa_%f__lmbda_%f'%(learn_rate,alfa,lmda)
np.save(save_fpath+'exphat__'+batch_tag,exphat)
pd.concat(dataL).to_csv(save_fpath+'semdata__'+batch_tag+'.csv')

