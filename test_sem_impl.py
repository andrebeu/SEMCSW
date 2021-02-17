from CSWSEM import *



task = CSWTask()

sem_kwargs={
    'lmda':1,
    'alfa':10,
    'nosplit':False
}
sem = SEM(**sem_kwargs)


exp_kwargs={
    'condition':'blocked',
    'n_train':4,
    'n_test':0
}

# n = 3
# n = np.arange(0, 9)[np.sum(np.cumsum(t[n, :]) < np.random.uniform(0, 1))]
# print(n)

task.generate_experiment(**exp_kwargs)




