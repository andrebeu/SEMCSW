import sys
import itertools

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 

param_set_idx = int(sys.argv[1])

lr = [0.01, 0.05]
alfa =[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
lmbda =[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

itrprod = itertools.product(lr,alfa,lmbda)

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


