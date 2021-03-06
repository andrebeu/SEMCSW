# import sys
# import os
# import pandas as pd
# import numpy as np
# import torch as tr
# from CSWSEM import *

# # sweep params
# learn_rate = float(sys.argv[])
# alfa = float(sys.argv[])
# lmda = float(sys.argv[])

# stsize = 25

# ## gridsearch tag
# gs_name = 'gs2'

# ## looping 
# seedL = np.arange(51)
# condL = ['blocked','interleaved']

# dfL = [] # populate from each condition
# for nosplit,condition,seed in itertools.product([0,1],condL,seedL):
#     np.random.seed(seed)
#     tr.manual_seed(seed)
#     ## model tag
#     model_tag = 'nosplit_%i__cond_%s__learnrate_%.3f__alfa_%f__lmbda_%f__seed_%i'%(
#       int(nosplit),condition,learn_rate,alfa,lmda,seed)
#     print('\n\n**',model_tag,'**\n\n')

#     # params
#     exp_kwargs={
#         'condition':condition,
#         'n_train':160,
#         'n_test':40
#     }

#     sem_kwargs={
#         'nosplit':nosplit,
#         'stsize':stsize,
#         'lmda':lmda,
#         'alfa':alfa,
#         'learn_rate':learn_rate,
#         'seed':seed,
#     }

#     # setup
#     task = CSWTask()
#     sem = SEM(**sem_kwargs)

#     # run
#     exp,curr = task.generate_experiment(**exp_kwargs)
#     sem_data = sem.forward_exp(exp)

#     # record curriculum (not ideal, recording with every obs)
#     sem.data.record_exp('curriculum',curr)

#     # save
#     save_fpath = 'gsdata/%s/%s'%(gs_name,model_tag)
#     sem_data_df = pd.DataFrame(sem_data)
#     sem_data_df.to_csv(save_fpath) # just in case
#     dfL.append(sem_data_df)

# ## save again: prefer this
# full_tag = 'learnrate_%.3f__alfa_%f__lmbda_%f'%(learn_rate,alfa,lmda)
# save_fpath = 'gsdata/%s/%s'%(gs_name,full_tag)
# pd.concat(dfL).to_csv(save_fpath)
