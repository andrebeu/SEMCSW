# liketest (03/26/21)
- look into likelihood computation

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