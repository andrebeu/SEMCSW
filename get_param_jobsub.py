import sys
import itertools

"""
given an index, return parameter string
""" 

param_set_idx = int(sys.argv[1])

lr = [0.001, 0.005, 0.01]
ep =[2, 4, 8, 16, 32]
alfa =[-32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32]
lmbda =[-32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32]

for idx,(i,j,k,l) in enumerate(itertools.product(lr,ep,alfa,lmbda)):
  if idx == param_set_idx:
    print(i,j,j,l)
    break


