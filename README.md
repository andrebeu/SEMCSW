# discrete 04/03/21: discrete networks
- assumption: learning on one state does not affect others
- architecture: embed -> recurrent -> softmax
 


# handSEM 03/30/21: PE based schema selection 
- goal: verify intuition that B>I can be explained by poor schema selection
- control sim: random schema selection: B&I both around chance
- control sim: schema selection == curriculum: B&I both high acc
  - but note small diff early trials where B>I b/c only training 1 sch
- splitting rule based on PE magnitude:
  - i.e. if PE > PEthreshold, select new schema 
  - small range of PEthreshold where tiny effect test B>I
  - tried 2 "new schema selection" rules:
    - random and argmin
- splitting rule based on PE(t) - PE(t-1)
  - if PE(t)-PE(t-1)>thresh, select new schema
    - select new schema: rand vs PEargmin
  - first condition to show I acc remain low
  - however, B acc is also low


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
- result: when schema for trial t is selected at trial t-1, better deviation at block boundaries but B<I at test. 



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