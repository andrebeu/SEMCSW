import time
import os
import numpy as np
import torch as tr
from scipy.stats import norm
from CSWSEM import *

sem_kwargs={
  'nosplit':1,
  'alfa':np.exp(1),
  'lmda':np.exp(3),
  'seed':5,
  'rnn_kwargs': {
    'stsize':5,
    'pdim':2,
    'learn_rate':0.01
  }
}


exp_kwargs = {
    'condition':'single',
    'n_train':10,'n_test':2
    }
sem = SEM(**sem_kwargs)
task = CSWTask()
exp,curr = task.generate_experiment(**exp_kwargs)
sem_data = sem.forward_exp(exp,curr)


print('** DONE')