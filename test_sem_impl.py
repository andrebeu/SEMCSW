from CSWSEM import *

seed = 0
np.random.seed(10)

task = CSWTask()

exp_kwargs={
    'condition':'single',
    'n_train':4,
    'n_test':0
}
exp = task.generate_experiment(**exp_kwargs)



sch_kwargs={
    'stsize':10,
    'seed':seed
}
sch = CSWSchema(**sch_kwargs)
event = exp[0]
event_hat = sch.forward(event)



sem_kwargs={
    'lmda':1,
    'alfa':10,
    'nosplit':False
}
sem = SEM(**sem_kwargs)
# sem.forward_trial(event)
sem.forward_exp(exp)













