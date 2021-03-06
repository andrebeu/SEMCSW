# liketest (03/26/21)
- how to change interleaved without affecting blocked?
  - look into likelihood computation
    - setup 40 trials of interleaved vs single (blocked)
- note RNN size: realized that even when RNN size was set to 1, performance on blocked would not exhibit CI, suggesting learning happening in projection layers. to fix this, inputs are projected down to size 3. 
- refactor argument passing to RNN 
- made batch_sim and BIexp wrapper funs 
- gridsearch over pdim and stsize:
  - small B>I results when pdim=4,stsize=5
- todo:
  - refactor likelihood 
  - check effect of:
    - remove input layer
    - rnn vs gru vs lstm
    - update prior count by 1 vs 5
    - 
- result: good BIseparation on early trials when embed_size == stsize = 10 or 15 (lr=0.05)



# gs2 notes
- change to gridsearch procedure 
  - single job contains multiple seeds
  - parameter passing controlled `get_param_jobsub.py`
  - caution: this gridsearch procedure repeats LSTM sims for all SEM sims


# procedural notes
- NB-run_sem.pynb used for debugging and testing before launching gridsearch
  - code in here should be kept in a state that is amenable to gridsearching




# Model of Curriculum Effect on Schema Learning

## Results

- blocked > interleaved
- early > middle/late

## Modeling procedure

- Fit SEM on blocked/interleaved
- Evaluate on early/middle/late


** MOVING TO DIFFERENT FOLDER: when switching between AB and NF repos, github sometimes bugs out with finder: new files wont load. 