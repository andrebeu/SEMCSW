# -*- coding: utf-8 -*-
"""Schema Prediction, Silvy version, 12/3/19, updated for TF2 7/23/20

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14VdDVH6flmYZSqf1lwcTJXXXrJyPrrAD

This notebook includes all of the code for the experients and analyses.  It uses a python v3 and tensorflow v1.14 port of the SEM model, and should run as is in Colab.  

I've hidden most of the code except for the relevant parameters to switch between Blocked > Interleaved and Interleaved > Blocked.  Hopefully the plots are clear, but let me know if they are not.

## Parameters to Manipulate
"""

# these two parmaeters define the prior over the event noise. 
# The prior an inverse chi-squared distribution, which I have parameterized
# in a non-standard way, by the degrees of freedom and the mode 
#
# For Blocked greater than interleaved, set the mode to 2 ** -7.5 
# For interleaved greater than blocked, set the mode to 2 ** -5.5
df0 = 2. ** -10.5  # degrees of freedom
mode = 2 ** -6.25  # mode of inverse chi-squred


# Sample size of simulations. Note: blocked > interleaved has a smaller effect
# size than interleaved > blocked, and requires more samples to show cleanly 
n_batch = 250



"""# Code for experiment and analyeses

## Load Libraries
"""



# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
sns.set_context('talk')

# imports from SEM model
from sem.hrr import plate_formula, encode, decode, embed_gaussian
from sem import sem_run_with_boundaries, SEM
from sem.event_models import GRUEvent
from sem.utils import fast_mvnorm_diagonal_logprob
from sem.memory import gibbs_memory_sampler, reconstruction_accuracy, create_corrupted_trace

from sem.event_models import LinearEvent, map_variance, RecurrentLinearEvent 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, Lambda, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import l2_normalize
from sem.utils import fast_mvnorm_diagonal_logprob, unroll_data, get_prior_scale, delete_object_attributes
from scipy.stats import norm
import numpy as np

import json


class GRUEvent_normed(RecurrentLinearEvent):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=0.0, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None):

        RecurrentLinearEvent.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                                     l2_regularization=l2_regularization, batch_size=batch_size,
                                     kernel_initializer=kernel_initializer, init_model=False,
                                     prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                     batch_update=batch_update, optimizer_kwargs=optimizer_kwargs)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(GRU(self.n_hidden, input_shape=(self.t, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.add(Lambda(lambda x: l2_normalize(x, axis=-1)))  
        self.model.compile(**self.compile_opts)

# used in the experimental design
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import pandas as pd
from scipy.special import logsumexp

"""## Define Experiment"""

len([ii for ii in range(1, 13)] + [ii for ii in range(1, 13)] +[ii for jj in 
                          [ii  for ii in range(13, 18, 2) for _ in range(2)]
                          for ii in range(jj, jj+2)] + [19, 20])

def get_boundaries(e_hat):
    return np.array(
        np.concatenate([np.array([0]), e_hat[0:-1] != e_hat[1:]]), dtype=bool
    )

def logsumexp_mean(x):
    """ return the log of the mean, given a 1-d array of log values"""
    return logsumexp(x) - np.log(len(x))

def _encode_sequence(words, list_s, list_c, agent_mult=1.0, ctx_weight=1.0):
    
    def encode_scene(s, c):
        X = encode(words[s[0]], words['isVerb']) + \
            agent_mult * encode(words[s[1]], words['isAgent']) + \
            agent_mult * encode(words[s[2]], words['isAgent']) + \
            ctx_weight * encode(words[c], words['isContext'])
        # divide by sqrt(4), as specified in plate, 1995, to keep expected 
        # length at ~1.0
        return X / np.sqrt(1 + agent_mult * 2. + ctx_weight)

    stories = []
    for s, c in zip(list_s, list_c):
        story = []
        for s0 in s:
            story.append(
                encode_scene(s0, c)
            )
        stories.append(np.concatenate(story))

    return stories


def generate_exp_blocked(seed=None, n_stories=12, err=0.1, ctx_weight=1.0):
    """
    :param seed: (int), random seed for consistency
    :param n_stories: (int, default=12), number of stories per wedding type
    :param err: (float, default=0.1) error probability in Plate's forumla
    :param ctx_weight: (float, default=1.0) how much to weight the context 
                        vector in the embedding space

    Each scene in the experiment is the conjunction of five terms: (1) context, 
    (2-3) actors, (4) verb, and (5) features.

    The verbs (nodes in the task graph) are listed as integers with the 
    following correspondance:
        Start:      0    
        Campfire:   1
        Flower:     2
        Coin:       3
        Torch:      4
        Egg:        5
        Painting:   6
        Gifts:      7

    Note that each feature is unique to each scene and there is no shared 
    correlation structure for these features.


    """
    if seed is not None:
        assert type(seed) == int
        np.random.seed(seed)

    # transition functions are defined T(s, s') = p
    transition_probs_g = {
        (0, 1): 0.5, (0, 2): 0.5, 
        (1, 3): 1.0, (1, 4): 0.0, 
        (2, 3): 0.0, (2, 4): 1.0, 
        (3, 5): 1.0, (3, 6): 0.0,
        (4, 5): 0.0, (4, 6): 1.0,
        (5, 7): 1.0, (6, 7): 1.0,
    }

    # transition functions are defined T(s, s') = p
    transition_probs_y = {
        (0, 1): 0.5, (0, 2): 0.5, 
        (1, 3): 0.0, (1, 4): 1.0, 
        (2, 3): 1.0, (2, 4): 0.0, 
        (3, 5): 0.0, (3, 6): 1.0,
        (4, 5): 1.0, (4, 6): 0.0,
        (5, 7): 1.0, (6, 7): 1.0,
    }
    
    def make_t_matrix(transition_prob_dict):
        t = np.zeros((8, 8))
        for (x, y), p in transition_prob_dict.items():
            t[x, y] = p
        return t

    transitions_y = make_t_matrix(transition_probs_y)
    transitions_g = make_t_matrix(transition_probs_g)

    # the test stimuli is hand chosen
    test_sequence = [
        [0, 1, 3, 5, 7],  # g
        [0, 1, 3, 5, 7],  # g
        [0, 2, 3, 6, 7],
        [0, 1, 4, 5, 7],
        [0, 2, 4, 6, 7],  # g
        [0, 2, 4, 6, 7],  # g
        [0, 1, 4, 5, 7],
        [0, 2, 3, 6, 7],
        [0, 1, 3, 5, 7],  # g
        [0, 2, 4, 6, 7],  # g
        [0, 2, 3, 6, 7],
        [0, 1, 4, 5, 7],
        [0, 1, 3, 5, 7],  # g
        [0, 2, 4, 6, 7],  # g
    ]

    n_verbs = 8  # verbs
    n_actors_train = 2 * n_stories * 2  # number of actors in the training
    n_actors_test = len(test_sequence) * 2

    # n_actors = 2 * (n_actors_train + n_actors_test) * 2  # number of actors in training and test

    # draw a single set of stories
    s_train = []
    y_train = []
    for ii in range(n_actors_train // 2):
        a0 = 'Actor{}'.format(ii)
        a1 = 'Actor{}'.format(ii + n_actors_train // 2)

        # draw the nodes
        if ii < int(n_actors_train / 4):
            t = transitions_g
        else:
            t = transitions_y

        # encode the initial scene
        n = 0  # pick the initial scene
        story = [['Verb0', a0, a1]]
        y_train.append(n)
        while n < 7:  # draw all of the scenes until the last scene (scene 7) is reached
            # draw a random scene, conditioned on the current scene, using the transition function
            n = np.arange(0, 8)[np.sum(np.cumsum(t[n, :]) < np.random.uniform(0, 1))]

            # encode the scenes
            verb = 'Verb{}'.format(n)
            story.append([verb, a0, a1])
            y_train.append(n)
        s_train.append(story)

    # list contexts
    c_blocked = ['ContextG'] * n_stories + ['ContextY'] * n_stories
    c_test = [
        'ContextG', 'ContextG',
        'ContextY', 'ContextY',
        'ContextG', 'ContextG',
        'ContextY', 'ContextY',
        'ContextG', 'ContextG',
        'ContextY', 'ContextY',
        'ContextG', 'ContextG',
    ]

    s_test = []
    for ii, vs in enumerate(test_sequence, n_actors_train):
        story = []
        for v in vs:
            story.append(['Verb{}'.format(v), 'Actor{}'.format(ii), 'Actor{}'.format(ii + n_actors_test // 2)])
        s_test.append(story)

    # figure out how many dimensions we need using plates formula
    n = n_actors_train + n_verbs + 2  # this need to be fixed, by adding n_actors_test (but this will change previous parameterization)
    k = 6  # maximum number of terms to be combined

    d = plate_formula(n, k, err)
    
    verb_property = embed_gaussian(d)
    agent_property = embed_gaussian(d)

    # when combining terms, devide by sqrt(n_terms) to keep expected length ~1.0
    embedding_library = {'Verb{}'.format(ii): (embed_gaussian(d) + verb_property) / np.sqrt(2.0) for ii in range(n_verbs)}
    embedding_library.update({'Actor{}'.format(ii): (embed_gaussian(d) + agent_property)  / np.sqrt(2.0) for ii in range(n_actors_train + n_actors_test)})
    embedding_library.update({'isAgent': embed_gaussian(d), 'isVerb': embed_gaussian(d)})
    embedding_library.update({'ContextG': embed_gaussian(d), 'ContextY': embed_gaussian(d), 'isContext': embed_gaussian(d)})

    keys = list(embedding_library.keys())
    keys.sort()

    s_train = _encode_sequence(embedding_library, s_train, c_blocked, ctx_weight=ctx_weight)
    s_test = _encode_sequence(embedding_library, s_test, c_test, ctx_weight=ctx_weight)

    # create a vector or random items, each one representing a summation of all detials associated with each 
    # this is a hassle to pre-create in the library, so it's manually defined here
    t = 0
    for ii in range(len(s_train)):
        items = embed_gaussian(d, n=len(s_train[ii]))
        s_train[ii] = (s_train[ii] * np.sqrt(4) + items) / np.sqrt(5)

        # cache each scene-specific feature vector
        for jj in range(len(s_train[ii])):
            embedding_library['TrainItem{}'.format(t)] = items[jj, :].reshape(1, -1)
            t += 1

    # create unique items for each of the sentances.  Again, this is a hassle to pre-create in the library, so 
    # it's manually defined here
    n_items = len(s_test) * len(s_test[0])
    for ii in range(n_items):
        embedding_library['TestItem{}'.format(ii)] = embed_gaussian(d)

    t = 0
    for ii in range(len(s_test)):
        for jj in range(len(s_test[0])):
            s_test[ii][jj, :] = (s_test[ii][jj, :] * np.sqrt(4) + embedding_library['TestItem{}'.format(t)]) / np.sqrt(5)
            t += 1

    # return s_blocked, s_test, embedding_library
    y_train = np.array(y_train)
    y_test = np.concatenate(test_sequence)
    return s_train, s_test, y_train, y_test, embedding_library


def get_new_event_prob(e_hat):
    return np.array([True] + [e_hat[ii] not in set(e_hat[:ii]) for ii in range(1, len(e_hat))])

def get_total_n_events(e_hat):
    return np.array([np.max(e_hat[:ii]) for ii in range(1, len(e_hat) + 1)])

def count_repeats(e_hat):
    counts = {k:1 for k in set(e_hat)}
    repeat_list = []
    for ii in range(len(e_hat)):
        repeat_list.append(counts[e_hat[ii]])
        counts[e_hat[ii]] += 1
        
    return repeat_list


def classify_verbs(results, y):
    # create a decoder based on both the training stimuli from both experiments
    
    clf_class = LogisticRegression
    clf_kwargs = dict(C=10.0, multi_class='multinomial', solver='lbfgs')
    verb_decoding_clf = clf_class(**clf_kwargs)
    verb_decoding_clf.fit(results.x_orig, y)

    # use the decoder
    y_hat = verb_decoding_clf.predict(results.x_hat)
    decoding_acc = y == y_hat
    decoding_prob_corr = np.array(
            [verb_decoding_clf.predict_proba(results.x_hat)[ii, y0]
                for ii, y0 in enumerate(y)]
            )

        # vector of scene labels
    scene = np.zeros_like(y)
    scene[y >= 1] += 1
    scene[y >= 3] += 1
    scene[y >= 5] += 1 
    scene[y >= 7] += 1



    def decode_2afc(scene_selected):
        clf_class = LogisticRegression
        clf_kwargs = dict(C=10.0, multi_class='multinomial', solver='lbfgs')
        verb_decoding_clf = clf_class(**clf_kwargs)

        _sel = scene_selected == scene
        _y = y[_sel] - np.min(y[_sel])
        verb_decoding_clf.fit(results.x_orig[_sel, :], _y)
        probs = verb_decoding_clf.predict_proba(results.x_hat[_sel, :])
        prob_corr = np.array([probs[ii, y0] for ii, y0 in enumerate(_y)])
        return prob_corr
        
    prob_corr_2afc = np.ones_like(decoding_prob_corr)
    prob_corr_2afc[scene == 1] = decode_2afc(1)
    prob_corr_2afc[scene == 2] = decode_2afc(2)
    prob_corr_2afc[scene == 3] = decode_2afc(3)
    # prob_corr_2afc[scene == 4] = decode_2afc(4)


    return decoding_acc, decoding_prob_corr, prob_corr_2afc



def score_results(results, e, y, n_train=24, n_test=14):
    """ function that takes in the completed SEM results object, plus vector of 
    true nodes and create prelimiarly analyses
    """

    # hard code these for now
    n_trials = n_train + n_test

    # create a decoder based on both the training stimuli from both experiments
    decoding_acc, decoding_prob_corr, prob_corr_2afc = classify_verbs(results, y)
    
    ###### model scoring and analyses ##### 
    e_hat = results.e_hat
    schema_repeats = count_repeats(e)
    schema_reps_inferred = count_repeats(e_hat)

    # # calculate prediction error (max distance ~ 1.0)
    pes = np.linalg.norm(results.x_hat - results.x_orig, axis=1) / np.sqrt(2)

    # # these are the relevant prediction trials in the test phase
    t = np.array([t0 for _ in range(n_trials) for t0 in range(5)])
    pred_trials = ((t == 2) | (t == 3))
    is_test = np.array([t0 >= n_train for t0 in range(n_trials) for _ in range(5)])

    results = [{
        'Trials': 'All',
        'adjRand': float(adjusted_rand_score(e_hat, e)),
        'nClusters': len(set(e_hat)), 
        'pe': float(np.mean(pes)),
        'pe (probes)': float(np.mean(pes[pred_trials])),
        'verb decoder Accuracy': float(np.mean(decoding_acc[pred_trials])),
        'verb decoder Accuracy Prob': float(np.mean(decoding_prob_corr[pred_trials])),
        'verb 2 AFC decoder Prob': float(np.mean(prob_corr_2afc[pred_trials & pred_trials])),
    }]
    results.append({
        'Trials': 'Training',
        'adjRand': float(adjusted_rand_score(e_hat[:n_train], e[:n_train])),
        'nClusters': len(set(e_hat[:n_train])), 
        'pe': float(np.mean(pes[is_test == False])),
        'pe (probes)': float(np.mean(pes[pred_trials & (is_test == False)])),
        'verb decoder Accuracy': float(np.mean(decoding_acc[pred_trials & (is_test == False)])),
        'verb decoder Accuracy Prob': float(np.mean(decoding_prob_corr[pred_trials & (is_test == False)])),
        'verb 2 AFC decoder Prob': float(np.mean(prob_corr_2afc[pred_trials & (is_test == False)])),
    })
    results.append({
        'Trials': 'Test',
        'adjRand': float(adjusted_rand_score(e_hat[n_train:], e[n_train:])),
        'nClusters': len(set(e_hat[n_train:])), 
        'pe': float(np.mean(pes[is_test])),
        'pe (probes)': float(np.mean(pes[pred_trials & is_test])),
        'verb decoder Accuracy': float(np.mean(decoding_acc[pred_trials & is_test])),
        'verb decoder Accuracy Prob': float(np.mean(decoding_prob_corr[pred_trials & is_test])),
        'verb 2 AFC decoder Prob': float(np.mean(prob_corr_2afc[pred_trials & is_test])),
        'cluster re-use': float(np.mean([c in set(e_hat[:n_train]) for c in e_hat[n_train:]])),
    })

    # loop through these measure to create a memory-efficient list of dictionaries
    # (as opposed to a memory-inefficient list of pandas DataFrames)
    _bounds_vec = get_boundaries(e_hat)
    _new_event_prob = get_new_event_prob(e_hat)
    _total_n_events = get_total_n_events(e_hat)
    scence_counter = 0
    boundaries = []
    prediction_err = []
    for t in range(n_trials):
        boundaries.append(
            {
                'Trials': ['Training', 'Test'][t >= n_train],
                # 'Condition': condition, 
                # 'batch': batch,
                'Boundaries': int(_bounds_vec[t]),
                't': t,
                'Schema Repeats': schema_repeats[t],
                'New Event': int(_new_event_prob[t]),
                'Total N Events': int(_total_n_events[t]),
                'e_hat': int(e_hat[t]),
            }
        )

        for kk in range(5):
            prediction_err.append(
                {
                    'Story': t,
                    # 'Condition': condition,
                    # 'batch': batch,
                    't': kk,
                    'Trials': ['Training', 'Test'][t >= n_train],
                    'pe': float(pes[scence_counter]),
                    'Schema True': int(e[t]),
                    'Schema Repeats': schema_repeats[t],
                    'Schema Repeats (Inferred)': schema_reps_inferred[t],
                    'verb decoder Accuracy': float(decoding_acc[scence_counter]),
                    'verb decoder Accuracy Prob': float(decoding_prob_corr[scence_counter]),
                    'verb 2 AFC decoder Prob': float(prob_corr_2afc[scence_counter]),
                }
            )
            scence_counter += 1


    # Delete SEM, clear all local variables
    x, y, e = None, None, None, 
    e_hat, schema_repeats, schema_reps_inferred, pes = None, None, None, None
    t, pred_trials, decoding_acc = None, None, None
    decoding_prob_corr, decoding_acc, prob_corr_2afc = None, None, None

    # return results!
    return results, boundaries, prediction_err


def batch_exp(sem_kwargs, stories_kwargs, gibbs_kwargs, n_batch=12,
         n_stories=12, epsilon_e=0.25, reconstruct=True):

    results = []
    boundaries = []
    prediction_err = []

    # set fixed params
    n_train = n_stories * 2
    n_test  = 14

    # construct sequence of true events
    e_blk_train = [0] * 12 + [1] * 12
    e_int_train = [0, 1] * 12
    e_test = [0, 0, 1, 1] * 3 + [0, 0]

    # keep track of how many times a schema has been seen
    schema_repeats_blk = [ii for ii in range(1, 13)] + [ii for ii in range(1, 13)]
    schema_repeats_int = [ii  for ii in range(1, 13) for _ in range(2)]
    schema_repeats_tst = [ii for jj in 
                          [ii  for ii in range(13, 18, 2) for _ in range(2)]
                          for ii in range(jj, jj+2)] + [19, 20]
    schema_repeats_blk += schema_repeats_tst
    schema_repeats_int += schema_repeats_tst

    for kk in tqdm(range(n_batch), desc='Batches'):

        # generate the stories for the model
        blocked_stories, test_stories, y_blocked, y_test, embedding_library = \
          generate_exp_blocked(**stories_kwargs)

        # generate the interleaved stories by re-ordering the blocked
        interleaved_ordering = []
        for ii, jj in zip(range(n_stories), range(n_stories, n_stories * 2)):
            interleaved_ordering.append(ii)
            interleaved_ordering.append(jj)
        interleaved_stories = [blocked_stories[ii] for ii in interleaved_ordering]
        y_interleaved = np.reshape([y_blocked.reshape((-1, 5)).tolist()[ii] 
                                    for ii in interleaved_ordering], -1)

        # run the blocked condition
        run_kwargs = dict(save_x_hat=True, progress_bar=False)
        results_blk = sem_run_with_boundaries(blocked_stories + test_stories, sem_kwargs, run_kwargs)
        results_blk.x_orig = np.concatenate(blocked_stories + test_stories)
        
        # run the interleaved condition
        run_kwargs = dict(save_x_hat=True, progress_bar=False)
        results_int = sem_run_with_boundaries(interleaved_stories + test_stories, sem_kwargs, run_kwargs)
        results_int.x_orig = np.concatenate(interleaved_stories + test_stories)
    
        ###### model scoring and analyses #####

        # add batch number and condition to all of the results
        def add_batch_cond(json_data, condition):
            for ii in range(len(json_data)):
                json_data[ii]['batch'] = kk
                json_data[ii]['Condition'] = condition
            return json_data

        for sem_results, condition in zip([results_blk, results_int], ['Blocked', 'Interleaved']):

            if condition == 'Blocked':
                e = e_blk_train + e_test
                y = np.concatenate([y_blocked, y_test])
            else:
                e = e_int_train + e_test
                y = np.concatenate([y_interleaved, y_test])

            _res, _bound, _pred = score_results(sem_results, e, y)
            _res = add_batch_cond(_res, condition)
            _bound = add_batch_cond(_bound, condition)
            _pred = add_batch_cond(_pred, condition)

            results += _res
            boundaries += _bound
            prediction_err += _pred

        # save the output to json files (will overwrite?)

        with open('./SilvyTask_newAnal_results.json', 'w') as fp:
            json.dump(results, fp)
        with open('./SilvyTask_newAnal_boundaries.json', 'w') as fp:
            json.dump(boundaries, fp)
        with open('./SilvyTask_newAnal_prederr.json', 'w') as fp:
            json.dump(prediction_err, fp)

    
    return 

"""## Story parameters"""

# np.random.seed(0)

n_stories = 12 # 12 in silvy's experiment / group
seed = None
err = 0.1; # error probability for plate's formula

ctx_weight = 1.0

# story_generator = generate_exp_balanced
story_kwargs = dict(n_stories=n_stories, seed=seed, err=err, ctx_weight=ctx_weight)

blocked_stories, test_stories, y_blocked, y_test, embedding_library = generate_exp_blocked(**story_kwargs)
d = np.shape(blocked_stories)[-1]
print("Dimensions: {}".format(d))
print("Blocked median feature variance: {}".format(
    np.median(np.concatenate(blocked_stories).var(axis=0))))

"""# Other SEM Parameters

Here, we define all of the remaining parameters necessary to run the experiment.  These do not need to be changed between experimental variants.
"""

# Likelihood Parameters

# df0 = 2. ** -4.5   # best -4.5 
# mode = 2 ** -7.5 # best -7.25 ?
scale0 = (mode * (df0 + 2)) / df0
# print scale0


prior_log_prob = 1.07614614 * 199. * 1.0
# SEM parameters

# scaling factor for popularity-based clustering, higher values == higher popularity weight
omega = -5 # (expressed as a power of 2)  (best, -5)

lmda = 2.0 ** (4 - omega) # stickyness parameter
alfa = 2.0 ** (5 - omega) # concentration parameter


# slow down learning in the network as much as possible
dropout           = 0.5
l2_regularization = 0.0
n_epochs          = 5


f_class = GRUEvent_normed


optimizer_kwargs = dict(
    lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False
)

# optimizer_kwargs = None


f_opts=dict(var_scale0=scale0, var_df0=df0, dropout=dropout, batch_size=25, 
            l2_regularization=l2_regularization, n_epochs=n_epochs,
            prior_log_prob=prior_log_prob, batch_update=True,
            optimizer_kwargs=optimizer_kwargs)

sem_kwargs = dict(lmda=lmda, alfa=alfa, f_class=f_class, f_opts=f_opts)

# set the parameters for the Gibbs sampler
b = 2
gibbs_kwargs = dict(
    memory_alpha = alfa,
    memory_lambda = lmda,
    memory_epsilon = np.exp(-(2**2)),
    b = b,  # re-defined here for completeness
    tau = scale0 * (10**-0.75),  # ibid
    n_samples = 50,
    n_burnin = 0,
    progress_bar=False,
)

epsilon_e = 0.25

"""# Run model"""
batch_exp(
    sem_kwargs, story_kwargs, gibbs_kwargs, n_batch=n_batch, reconstruct=False)

# results, boundaries, new_event, prediction_err = batch_exp_balanced_mem(
#     sem_kwargs, story_kwargs, gibbs_kwargs, n_batch=n_batch, reconstruct=False)


# results.to_pickle('SilvyTask_results.pkl')
# boundaries.to_pickle('SilvyTask_boundaries.pkl')
# new_event.to_pickle('SilvyTask_new_event.pkl')
# prediction_err.to_pickle('SilvyTask_prediction_err.pkl')