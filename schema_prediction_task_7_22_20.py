"""Schema Prediction, Andre's version, 5/12/20


This notebook includes all of the code for the experients and analyses.  It uses a python v3 and 
tensorflow v2 port of the SEM model.  This version is mimics the tasks that Andre ran but 
is simper in its representation.

## Load Libraries
"""

import numpy as np
import json
from tqdm import tqdm

# used in the experimental design
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing # used sparingly
from scipy.special import logsumexp

# ########## code to load SEM ##########
# ##### imports from SEM model
# import sys, os
# # add the current path to python to allow loading SEM
# current_path = os.path.abspath('.')
# parent_path = os.path.dirname(current_path)
# sys.path.append(parent_path)

from sem import sem_run_with_boundaries, SEM
from sem.hrr import plate_formula, encode, decode, embed_gaussian
from sem.utils import fast_mvnorm_diagonal_logprob, get_prior_scale, processify
from no_split_sem import no_split_sem_run_with_boundaries, NoSplitSEM



"""## Define Experiment"""


## these are utility functions

def get_boundaries(e_hat):
    return np.array(
        np.concatenate([np.array([0]), e_hat[0:-1] != e_hat[1:]]), dtype=bool
    )

def logsumexp_mean(x):
    """ return the log of the mean, given a 1-d array of log values"""
    return logsumexp(x) - np.log(len(x))

def generate_exp(condition, seed=None, err=0.1, n_train=80, n_test=40, embedding_library=None, 
                    actor_weight=1.0, context_weight=1.0):
    """
    :param condition: (str), either 'blocked', 'interleaved', 'early', 'middle', or 'late'
    :param seed: (int), random seed for consistency
    :param err: (float, default=0.1) error probability in Plate's forumla
    :param n_train: (int, default=160), number of train events
    :param n_test: (int, default=40), number of test events
    :param embedding_library: (dictionary, default=None) contains the word embeddings,
        to allow a single library to be reused mutliple times
    :param actor_weight: (float, default=1) how much to weight the actor in the representation


    Each scene in the experiment is the conjunction of 2 terms: 
    (1) agent + (2) node

    Andre's task has the following two graph structures:

    "Brew house"
    -------------
                        ->  Node11 -> Node21 -> Node31
    Begin -> LocNodeB                                   -> End
                        ->  Node12 -> Node22 -> Node32


    "Deep Ocean Cafe"
    -------------
                        ->  Node11 -> Node22 -> Node31
    Begin -> LocNodeC                                   -> End
                        ->  Node12 -> Node21 -> Node32

    These graphs are diffrentiaed by "LocNodeB" and "LocNodeC" and the 
    transitions after Nodes11/12 and Nodes21/22.  The graphs are otherwise identicial,
    will all tranistions determinstic, except those following LocNodeB and LocNodeC.

    The nodes in the task graps are listed as integers with the 
    following correspondance:
        Begin:      0    
        LocNodeB:   1
        LocNodeC:   2
        Node11:     3
        Node12:     4
        Node21:     5
        Node22:     6
        Node31:     7
        Node32:     8
        End:        9 (NOTE: Not using the end node anymore)

    Note that each feature is unique to each scene and there is no shared 
    correlation structure for these features.


    """

    # check for valid condition type
    assert type(condition) == str        
    if condition[0].lower() == 'b':
        condition = 'blocked'
    elif (condition[0].lower() == 'i'):
        condition = 'interleaved'
    elif (condition[0].lower() == 'e'):
        condition = 'early'
    elif (condition[0].lower() == 'm'):
        condition = 'middle'
    elif (condition[0].lower() == 'l'):
        condition = 'late'
    else:
        message = 'Condition Specified: {}\nPlease specify either blocked or interleaved!'.format(condition)
        raise(Exception(message))


    if seed is not None:
        assert type(seed) == int
        np.random.seed(seed)

    # transition functions are defined T(s, s') = p
    # these are the "Brew house" stories
    transition_probs_b = {
        (0, 1): 1.0, 
        (1, 3): 0.5, (1, 4): 0.5, 
        (3, 5): 1.0,
        (4, 6): 1.0, 
        (5, 7): 1.0,
        (6, 8): 1.0,
    }

    # these are the "Deep Ocean Cafe" stories
    transition_probs_c = {
        (0, 2): 1.0, 
        (2, 3): 0.5, (2, 4): 0.5, 
        (3, 6): 1.0,
        (4, 5): 1.0, 
        (5, 8): 1.0,
        (6, 7): 1.0,
    }
    
    def make_t_matrix(transition_prob_dict):
        t = np.zeros((10, 10))
        for (x, y), p in transition_prob_dict.items():
            t[x, y] = p
        return t

    # python logic makes it easier to rename these types as b=0, c=1
    transitions = {
        0: make_t_matrix(transition_probs_b),
        1: make_t_matrix(transition_probs_c)
    }

    # generate the order of the events
    list_transitions = []   # this is a list of the 
                            # transitions to use in each trial, as a function of trial number
    if condition == 'blocked':
        list_transitions =  \
            [0] * (n_train // 4) + \
            [1] * (n_train // 4) + \
            [0] * (n_train // 4) + \
            [1] * (n_train // 4 )
    elif condition == 'early':
        list_transitions =  \
            [0] * (n_train // 4) + \
            [1] * (n_train // 4) + \
            [0, 1] * (n_train // 4)
    elif condition == 'middle':
        list_transitions =  \
            [0, 1] * (n_train // 8) + \
            [0] * (n_train // 4) + \
            [1] * (n_train // 4) + \
            [0, 1] * (n_train // 8)
    elif condition == 'late':
        list_transitions =  \
            [0, 1] * (n_train // 4) + \
            [0] * (n_train // 4) + \
            [1] * (n_train // 4)
    else:
        list_transitions = [0, 1] * (n_train // 2)

    # randomly flip which one starts off
    _X = int(np.random.rand() < 0.5)
    list_transitions = [abs(l-_X) for l in list_transitions]

    # add the test trials. These are always random
    list_transitions += [int(np.random.rand() < 0.5) for _ in range(n_test)]

    # draw the training/test stories
    stories = []
    y = []
    e = []
    for ii in range(n_train + n_test):
        a = 'Actor{}'.format(ii)
        # draw the nodes
        t = transitions[list_transitions[ii]]
        e.append(list_transitions[ii])

        # encode the initial scene
        n = 0  # pick the initial scene
        x0 = [['Verb0', a]]
        y.append(n)
        while n < 7:  # draw all of the scenes until the last scene (scene 7 or 8) is reached
            # draw a random scene, conditioned on the current scene, using the transition function
            n = np.arange(0, 9)[np.sum(np.cumsum(t[n, :]) < np.random.uniform(0, 1))]

            # encode the scenes
            v = 'Verb{}'.format(n)
            x0.append([v, a])
            y.append(n)
        stories.append(x0)


    # # figure out how many dimensions we need using plates formula
    n_verbs = 10  # verbs
    # n = n_train * 2 + n_test + n_verbs
    # k = 2  # maximum number of terms to be combined

    # d = plate_formula(n, k, err)

    # as a simplifying assumption, we will not use the full embedding or calculate it's
    # dimensions with Plate's formula.  Instead, we will set the number of dimensions to the 
    # the number of nodes in the graph
    d = n_verbs
    
    if embedding_library is None:
        verb_property = embed_gaussian(d)
        agent_property = embed_gaussian(d)

        # when combining terms, devide by sqrt(n_terms) to keep expected length ~1.0
        embedding_library = {
            'Verb{}'.format(ii): (embed_gaussian(d) + verb_property) / np.sqrt(2.0) 
            for ii in range(n_verbs)
        }
        embedding_library.update({
            'Actor{}'.format(ii): (embed_gaussian(d) + agent_property)  / np.sqrt(2.0) 
            for ii in range(n_train + n_test)
        })

    keys = list(embedding_library.keys())
    keys.sort()


    # create a context library
    context_lib = {ii: embed_gaussian(d) for ii in range(2)}

    # ~~~~~ 
    #### encode the stories  as vectors       
    def encode_scene(s, c):
        X = embedding_library[s[0]] + embedding_library[s[1]] * actor_weight \
            + context_lib[c]
        # divide by sqrt(2), as specified in plate, 1995, to keep expected 
        # length at ~1.0
        return X / np.sqrt(1 + actor_weight + context_weight)

    x = []  
    for s, c in zip(stories, e):
        x0 = [] # vector for a single story
        for s0 in s:
            x0.append(
                encode_scene(s0, c)
            )
        x.append(np.concatenate(x0))
    # ~~~~~ 

    return x, np.array(y), e, embedding_library


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
    prob_corr_2afc[scene == 4] = decode_2afc(4)


    return decoding_acc, decoding_prob_corr, prob_corr_2afc


def score_results(results, e, y, n_train=160, n_test=40):
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
    pred_trials = ((t == 3) | (t == 4))
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


def batch_exp(sem_kwargs, stories_kwargs, n_batch=8, n_train=160, n_test=40, progress_bar=True,
    sem_progress_bar=False, block_only=False, interleaved_only=False, aggregator=np.sum, run_mixed=False, 
    debug=False, save_to_json=False, json_tag='', json_file_path='./', no_split=False, normalize=False):
    """
    :param sem_kwargs: (dictionary)
    :param stories_kwargs: (dictionary) 
    :param n_batches: (int, default=8), a batch is one each of blocked and interleaved
    :param n_train: (int, default=160)
    :param n_test: (int, default=40)
    :param progress_bar: (bool, default=True)
    :param sem_progress_bar: (bool, default=False)
    :param block_only: (bool, default=False)
    :param interleaved_only: (bool, default=False)
    :param aggregator: (function, default=np.sum), defunct
    :param run_mixed: (bool, default=False) 
    :param debug: (bool, default=False), only run the 1st two blocks of trials
    :param save_to_json: (bool, default=False), save intermediate results to file
    """
    # if hasattr(stories_kwargs, 'n_train'):
    #     assert(stories_kwargs['n_train'] == n_train)
    # if hasattr(stories_kwargs, 'n_test'):
    #     assert(stories_kwargs['n_test'] == n_test)

    stories_kwargs['n_train'] = n_train
    stories_kwargs['n_test'] = n_test
    
    # create a function that takes in "blocked" or "interleaved" as an argument and runs a batch of trials
    def run_condition(condition, batch=1, no_split=False):
        """
        :param condition: (str), either 'blocked', 'interleaved', 'early', 'middle', or 'late'
        """
    
        # generate the stories for the model
        x, y, e, _ = generate_exp(condition, **stories_kwargs)

        if normalize:
            x = [preprocessing.normalize(x0) for x0 in x]

        n_trials = n_train + n_test
        if debug: 
            x = x[:n_train//2]
            y = y[:(n_train // 2) * 5]
            e = y[:(n_train // 2)]
            n_trials = n_train // 2


        # run the model
        run_kwargs = dict(save_x_hat=True, progress_bar=sem_progress_bar)

        if not no_split:
            results = sem_run_with_boundaries(x, sem_kwargs, run_kwargs)
            # sem_model = SEM(**sem_kwargs)
            # sem_model.run_w_boundaries(x, save_x_hat=True, progress_bar=sem_progress_bar)
            # results = sem_model
        else:
            results = no_split_sem_run_with_boundaries(x, sem_kwargs, run_kwargs)
        results.x_orig = np.concatenate(x)
        
        return score_results(results, e, y, n_train=n_train, n_test=n_test)

    results = []
    boundaries = []
    prediction_err = []

    # this code just controls the presence/absence of a progress bar -- it isn't important
    if progress_bar:
        def my_it(l):
            return tqdm(range(l), desc='Batches')
    else:
        def my_it(l):
            return range(l)

    for kk in my_it(n_batch):
        
        conditions = ['Blocked', "Interleaved"]
        if block_only:
            conditions.remove('Interleaved')
        if interleaved_only:
            conditions.remove('Blocked')

        if run_mixed:
            conditions = []
            conditions.append('Early')
            conditions.append('Middle')
            conditions.append('Late')

        if not conditions:
            raise(Exception("No conditions to run!"))           

        for condition in conditions:

            _res, _bound, _pred = run_condition(condition, kk, no_split)

            # add batch number and condition to all of the results
            def add_batch_cond(json_data):
                for ii in range(len(json_data)):
                    json_data[ii]['batch'] = kk
                    json_data[ii]['Condition'] = condition
                return json_data

            _res = add_batch_cond(_res)
            _bound = add_batch_cond(_bound)
            _pred = add_batch_cond(_pred)

            results += _res
            boundaries += _bound
            prediction_err += _pred

            if save_to_json:
                # save the intermediate results
                with open('{}results{}.json'.format(json_file_path, json_tag), 'w') as fp:
                    json.dump(results, fp)
                with open('{}boundaries{}.json'.format(json_file_path, json_tag), 'w') as fp:
                    json.dump(boundaries, fp)
                with open('{}prediction_err{}.json'.format(json_file_path, json_tag), 'w') as fp:
                    json.dump(prediction_err, fp)

    if not save_to_json:
        return results, boundaries, prediction_err
    return None, None, None

if __name__ == "__main__":
    pass