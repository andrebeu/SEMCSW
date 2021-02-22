# gs-absem notes
- first gridsearch with AB implementation of SEM
- unscaled alfa/lmda (i.e. not in log scale)
- does not record trial type information (needed at test) to calculate adjrand score
- main result: learning rate needs to be at least 0.05 to fit first learning block

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