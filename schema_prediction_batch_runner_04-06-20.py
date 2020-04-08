import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn import metrics

from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp
from scipy.stats import norm
from glob import glob


"""## Define experiment"""
from schema_prediction_task import generate_exp, batch_exp

"""## Story parameters"""

seed = None
err = 0.01; # error probability for plate's formula

# story_generator = generate_exp_balanced
story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)


x, y, e, embedding_library = generate_exp('blocked', **story_kwargs)
d = np.shape(x)[-1]
print("Dimensions: {}".format(d))
print("Median feature variance: {}".format(np.median(np.concatenate(x).var(axis=0))))

"""# Other SEM Parameters

"""

# Testing suggests the NN parameters dominate the behavior, so we only worry
# about those

dropout           = 0.0
l2_regularization = 0.0
n_epochs          = 13
batch_size        = 25
lr                = 0.01
epsilon           = 1e-5
log_alpha         = 0.0
log_lambda        = 0.0

optimizer_kwargs = dict(
    lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=False
)

f_opts=dict(
    dropout=dropout,
    batch_size=batch_size,
    l2_regularization=l2_regularization,
    n_epochs=n_epochs,batch_update=True,
    optimizer_kwargs=optimizer_kwargs)


# def run_single_batch(batch_number, lr, n_epochs, log_alpha, log_lambda, mixed=False):

def run_single_batch(batch_n, lr, n_epochs, log_alpha, log_lambda, mixed=False):
    # sem prior params (dosen't rely on mutability)
    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts)

    f_opts['n_epochs'] = int(n_epochs)
    optimizer_kwargs['lr'] = lr # this relies on mutability to change the f_opts dictionary

    json_tag = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n)
    json_tag_mixed = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}_mixed'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n)
    print(json_tag)

    if not mixed:
        # Don't want to re-run things...
        # check for the file existance of this simulation
        files = glob('./json_files/*{}*'.format(json_tag))
        files += glob('./*{}*'.format(json_tag))
        if not files:
            _, _, _ = batch_exp(
                sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
                progress_bar=False, block_only=False,
                run_mixed=False,
                save_to_json=True,
                json_tag=json_tag,
                json_file_path='./json_files/')
    else:
        files = glob('./json_files/*{}*'.format(json_tag_mixed))
        files += glob('./*{}*'.format(json_tag_mixed))
        if not files:
            _, _, _ = batch_exp(
                sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
                progress_bar=False, block_only=False,
                run_mixed=True,
                save_to_json=True,
                json_tag=json_tag_mixed,
                json_file_path='./json_files/')

lrs = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.01]
n_epochs_ = [20, 24, 28, 32]
log_alphas = [0.0, -5.0, -2.5,  2.5, 5.0, 7.5, -7.5]
log_lambdas = [0.0, 2.5, 5.0, 7.5, 10.0,]

t = 0

if __name__ == "__main__":

    batch_n = 7

    # go through these lists in a random order for better sampling
    parameter_tuples = []
    for lr in lrs:
        for n_epochs in n_epochs_:
            for log_alpha in log_alphas:
                for log_lambda in log_lambdas:
                    parameter_tuples.append((lr, n_epochs, log_alpha, log_lambda))

    for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameter_tuples):
        run_single_batch(batch_n, lr, int(n_epochs), log_alpha, log_lambda, mixed=False)
