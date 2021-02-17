from CSWSEM import *

seed = 0
np.random.seed(10)

task = CSWTask()

exp_kwargs={
    'condition':'single',
    'n_train':100,
    'n_test':10
}
exp = task.generate_experiment(**exp_kwargs)


learn_rate = 0.01
sch_kwargs={
    'stsize':10,
    'seed':seed,
    "learn_rate":learn_rate
}
sch = CSWSchema(**sch_kwargs)
event = exp[0]
event_hat = sch.forward(event)



sem_kwargs={
    'lmda':1,
    'alfa':10,
    'nosplit':True
}
sem = SEM(**sem_kwargs)
# sem.forward_trial(event)
sem.forward_exp(exp)













