"""

"""

import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# used in the experimental design
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing # used sparingly
from scipy.special import logsumexp

# SEM
from sem.hrr import embed_gaussian
from cswSEM import SEM 
from cswSEM import sem_run_with_boundaries, no_split_sem_run_with_boundaries




def get_boundaries(e_hat):
    return np.array(
        np.concatenate([np.array([0]), e_hat[0:-1] != e_hat[1:]]), dtype=bool
    )

def logsumexp_mean(x):
    """ return the log of the mean, given a 1-d array of log values"""
    return logsumexp(x) - np.log(len(x))


def generate_exp(condition, n_train=160, n_test=40, embedding_library=None, 
    actor_weight=0.0, instructions_weight=0.0, err=0.1):
    """
    :param condition: (str), either 'blocked', 'interleaved', 'early', 'middle', or 'late'
    :param seed: (int), random seed for consistency
    :param err: (float, default=0.1) error probability in Plate's forumla
    :param n_train: (int, default=160), number of train events
    :param n_test: (int, default=40), number of test events
    :param embedding_library: (dictionary, default=None) contains the word embeddings,
        to allow a single library to be reused mutliple times
    :param actor_weight: (float, default=1) how much to weight the actor in the representation
    :param actor_weight: (float, default=0) how much to weight the schema instructions (A or B) in the representation


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
    # if condition == "instructed_interleaved": # instructed interleaved
    #     stories_kwargs['instructions_weight'] = 1.0
    # elif condition == "instructed_blocked": # instructed blocked
    #     stories_kwargs['instructions_weight'] = 1.0
    # else:
    #     stories_kwargs['instructions_weight'] = 0.0

    # transition functions are defined T(s, s') = p
    # "Brew house" stories
    transition_probs_b = {
        (0, 1): 1.0, 
        (1, 3): 0.5, (1, 4): 0.5, 
        (3, 5): 1.0,
        (4, 6): 1.0, 
        (5, 7): 1.0,
        (6, 8): 1.0,
    }

    # "Deep Ocean Cafe" stories
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
    elif condition == 'interleaved':
        list_transitions = [0, 1] * (n_train // 2)
    elif condition == 'single': ## DEBUG
        list_transitions =  \
            [0] * (n_train) 
    else:
        print('condition not properly specified')
        assert False

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
        schema = 'Schema{}'.format(list_transitions[ii])

        # encode the initial scene
        n = 0  # pick the initial scene
        x0 = [['Verb0', a, schema]]
        y.append(n)
        while n < 7:  # draw all of the scenes until the last scene (scene 7 or 8) is reached
            # draw a random scene, conditioned on the current scene, using the transition function
            n = np.arange(0, 9)[np.sum(np.cumsum(t[n, :]) < np.random.uniform(0, 1))]

            # encode the scenes
            v = 'Verb{}'.format(n)
            
            x0.append([v, a, schema])
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
        embedding_library = get_embedding_library(
            d,n_verbs,n_train,n_test)
        

    keys = list(embedding_library.keys())
    keys.sort()

    # ~~~~~ 
    #### encode the stories  as vectors       
    def encode_scene(s):
        X = embedding_library[s[0]] \
            + embedding_library[s[1]] * actor_weight \
            + embedding_library[s[2]] * instructions_weight
        # divide by sqrt(2), as specified in plate, 1995, to keep expected 
        # length at ~1.0
        return X / np.sqrt(1 + actor_weight + instructions_weight)

    x = []  
    for s in stories:
        x0 = [] # vector for a single story
        for s0 in s:
            x0.append(
                encode_scene(s0)
            )
        x.append(np.concatenate(x0))
    # ~~~~~ 

    return x, np.array(y), e, embedding_library


def get_embedding_library(embed_dim,n_verbs,n_train,n_test):
    """ ELib is dict 
    keys: schema0,schema1, 
        verb 0 - 9, actor 0 - (n_train+n_test)
    """
    verb_property = embed_gaussian(embed_dim)
    agent_property = embed_gaussian(embed_dim)

    # when combining terms, devide by sqrt(n_terms) to keep expected length ~1.0
    embedding_library = {
        'Verb{}'.format(ii): (embed_gaussian(embed_dim) + verb_property) / np.sqrt(2.0) 
        for ii in range(n_verbs)
    }
    embedding_library.update({
        'Actor{}'.format(ii): (embed_gaussian(embed_dim) + agent_property)  / np.sqrt(2.0) 
        for ii in range(n_train + n_test)
    })
    embedding_library.update({
        'Schema{}'.format(ii): embed_gaussian(embed_dim)  for ii in [0, 1]
    })
    return embedding_library


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


def score_results(results, e, y, n_train=160, n_test=40, condensed=False):
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

    if not condensed:
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
    else:

        scence_counter = 0
        trial_by_trial = []

        scenes = np.arange(0, n_trials * 5)

        for t in range(n_trials):

            # average pe, accuracy across relevant scenes
            acc = 0
            pe = 0
            for kk in [3, 4]:
                scene_idx = (scenes % 5 == kk) & (scenes // 5 == t)
                acc += float(prob_corr_2afc[scene_idx]) / 2.
                pe += float(pes[scene_idx]) / 2. 
            trial_by_trial.append(
                {
                    't': t,
                    'e_hat': int(e_hat[t]),
                    'accuracy': acc,
                    'pe': pe,
                }
            )
        return results, trial_by_trial


def seed_exp(sem_kwargs, stories_kwargs=None, n_train=160, n_test=40, 
    model_type='SEM', seed=99, condition='blocked'):
    """
    Function generates random tasks and runs the model on them.  
    Returns relevant performance metrics, and can write these to file.

    :param sem_kwargs: (dictionary) contains optimizer and nn params
    :param stories_kwargs: (dictionary) specify the parameters for the stories
    :param n_train: (int, default=160)
    :param n_test: (int, default=40)
    :param no_split: (bool, default=False) use the no-split version of the code (i.e. run as a NN model)
    :param condition: (str), {'blocked','interleaved','early',
        'middle','late','instructed_interleaved','instructed_blocked'}
    """
    np.random.seed(seed)

    if stories_kwargs == None:
        stories_kwargs = {}
    stories_kwargs['n_train'] = n_train
    stories_kwargs['n_test'] = n_test
    
    results = []
    boundaries = []
    prediction_err = []
    trialXtrial = []

    print('seed',seed,'condition',condition)

    ## helper function, used later ##
    # add batch number and condition to all of the results
    def add_batch_cond(json_data):
        for ii in range(len(json_data)):
            json_data[ii]['seed'] = seed
            json_data[ii]['condition'] = condition
        return json_data
    ##  ~~~~~~~~~~~~~~~~~~~~~~~~  ##

    # generate experiment
    x, y, e, _ = generate_exp(condition, **stories_kwargs)

    ## run the model
    run_kwargs = dict()
    """ task is predict next scene, 
    therefore only pass x for training
    """
    if model_type == 'SEM':
        _sem_results = sem_run_with_boundaries(
            x, sem_kwargs, run_kwargs)
    elif model_type == 'LSTM':
        _sem_results = no_split_sem_run_with_boundaries(
            x, sem_kwargs, run_kwargs)
    _sem_results.x_orig = np.concatenate(x)

    ## scores results
    _res, _trialX = score_results(_sem_results, e, y, 
        n_train=n_train, n_test=n_test, condensed=True)
    
    results += add_batch_cond(_res)
    trialXtrial += add_batch_cond(_trialX)


    output = (results, trialXtrial, None)
    return output


if __name__ == "__main__":
    pass