import sys
import os
import numpy as np
import torch as tr

from CSWSEM import *


# sweep params
nosplit = int(sys.argv[1])
condition = str(sys.argv[2])
learn_rate = float(sys.argv[3])
alfa = float(sys.argv[4])
lmda = float(sys.argv[5])
seed = int(sys.argv[6])
stsize = 25


## gridsearch tag
gs_name = 'gs2'
## model tag
model_tag = 'nosplit_%i__cond_%s__learnrate_%.3f__alfa_%f__lmbda_%f__seed_%i'%(
  int(nosplit),condition,learn_rate,alfa,lmda,seed)
print(model_tag)


# params
exp_kwargs={
    'condition':condition,
    'n_train':160,
    'n_test':40
}


sem_kwargs={
    'nosplit':nosplit,
    'stsize':stsize,
    'lmda':lmda,
    'alfa':alfa,
    'learn_rate':learn_rate,
    'seed':seed,
}

# setup
task = CSWTask()
sem = SEM(**sem_kwargs)

# run
exp,curr = task.generate_experiment(**exp_kwargs)
sem_data = sem.forward_exp(exp)

# record curriculum (not idea, recording with every obs)
sem.data.record_exp('curriculum',curr)

# save
import pandas as pd
save_fpath = 'gsdata/%s/%s'%(gs_name,model_tag)
pd.DataFrame(sem_data).to_csv(save_fpath)

