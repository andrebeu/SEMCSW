import time
import os
import numpy as np
import torch as tr
from scipy.stats import norm
from CSWSEM import *

sem_kwargs={
  'nosplit':0,
  'alfa':10000000,
  'lmda':0.1,
  'seed':np.random.randint(99),
  'stsize':15,
  'learn_rate':0.05
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