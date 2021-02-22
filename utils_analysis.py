import os
import numpy as np
import torch as tr

from glob import glob as glob
import pandas as pd

from matplotlib import pyplot as plt

from CSWSEM import *

def make_gsdf(gsname,save=False):
  gs_dir = "gsdata/%s/"%gsname
  fpathL = glob(gs_dir+'*')
  seed_df_L = []
  for fpath in fpathL:
    condition = fpath.split('/')[-1].split('__')[1].split('_')[1]
    seed_df = pd.read_csv(fpath)
    seed_df.loc[:,'model'] = ['LSTM','SEM'][sum(seed_df.loc[:,'nosplit']==1)>0]
    seed_df.loc[:,'condition'] = condition
    seed_df_L.append(seed_df)
  gsdf = pd.concat(seed_df_L)
  gsdf.index = np.arange(len(gsdf))
  gsdf.drop(columns=['Unnamed: 0','like','prior'])
  if save:
      gsdf.to_csv('gsdata/%s.csv'%gsname)
      print('saved %s.csv'%gsname)
  return gsdf


