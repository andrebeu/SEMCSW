import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from tqdm import tqdm
# from local_event_models import GRUEvent
from sem.utils import delete_object_attributes

# https://github.com/nicktfranklin/SEM2

# there are a ~ton~ of tf warnings from Keras, suppress them here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NTF tensorflow implementation
from cswRNN import CSWEvent
# AB pytorch implementation
from CSW import CSWNet
import torch as tr

""" 

defn: schema is an rnn
defn: event is A or B
defn: state refers to csw states {1-8}
defn: instance of state vector is obs

- k is max number of NTF models
- - number of `event models` ?
- `active` is an array of keys to 'active event models' ?
- does this implementation initialize one model per trial?

resolve:
- currently contents of Results object used to execute model
    fix this so that anything necessary for model execution
    is passed between funs or object attribute 
    results object should be only for recording/analysis

procedure:
- mark what methods are ready for tr implementation 
2
"""



class BaseSEM(object):
    
    def __init__(self):
        # save_x_hat: bool
        #     save the MAP scene predictions?
        # :param save_x_hat: boolean (default False) normally, we don't save this as the interpretation can be tricky
        #     N.b: unlike the posterior calculation, this is done at the level of individual scenes within the
        #     events (and not one per event)
        
        return None

    def _update_state(self, x, k=None):
        """
        Update internal state based on input 
        data X and max # of event types (clusters) K
        """
        # get dimensions of data
        [n, d] = np.shape(x)
        if self.d is None:
            self.d = d
        else:
            assert self.d == d  # scenes must be of same dimension

        # get max # of clusters / event types
        if k is None:
            k = n
        self.k = max(self.k, k)

        # initialize CRP prior = running count of the clustering process
        if self.c.size < self.k:
            self.c = np.concatenate((self.c, np.zeros(self.k - self.c.size)), axis=0)
        assert self.c.size == self.k

    def clear_event_models(self):
        if self.event_models is not None:
            for _, e in self.event_models.items():
                e.clear()
                e.model = None
            
        self.event_models = None
        self.model = None
        tf.compat.v1.reset_default_graph()  # for being sure
        tf.keras.backend.clear_session()

    def init_for_boundaries(self, list_events):
        # update internal state
        print('i_f_bound')
        k = 0
        self._update_state(np.concatenate(list_events, axis=0), k)
        del k  # use self.k and self.d
        # store a compiled version of the model and session for reuse
        if self.k_prev is None:
            # initialize the first event model
            new_model = self.f_class(self.d, **self.f_opts)
            self.model = new_model.init_model()
            self.event_models[0] = new_model

    def run_w_boundaries(self, list_events):
        """
        This method is the same as the above except the 
        event boundaries are pre-specified by the experimenter
        as a list of event tokens (the event/schema type is still inferred).

        One difference is that the event token-type 
        association is bound at the last scene of an event type.
        N.B. ! also, all of the updating is done at 
        the event-token level.  There is no updating within an event!

        evaluate the probability of each event over the whole token


        Parameters
        ----------
        list_events: input data; list of n x d arrays -- each an event

        Return
        ------
        post: n_e by k array of posterior probabilities

        """
        print('r_w_bound')
        # print(type(list_events),len(list_events),
        #     type(list_events[0]),list_events[0].shape)

        # loop over events

        self.init_for_boundaries(list_events)

        for x in list_events:
            self.update_single_event(x)
        return None

    def update_prior_and_posterior_of_event_model(self,x):
        """ 
        only a few models are 'active', but here NTF expands dimension
        of prior for every new event 
        """
        event_len = np.shape(x)[0]
        ## update prior and posterior of event model
        self.k += 1
        self._update_state(x, self.k)

        # pull the relevant items from the results
        if self.results is None:
            self.results = Results()
            post = np.zeros((1, self.k))
            log_like = np.zeros((1, self.k)) - np.inf
            log_prior = np.zeros((1, self.k)) - np.inf
            x_hat = np.zeros((event_len, self.d))
            sigma = np.zeros((event_len, self.d))
            scene_log_like = np.zeros((event_len, self.k)) - np.inf # for debugging
        
        else:
            post = self.results.post
            log_like = self.results.log_like
            log_prior = self.results.log_prior
            x_hat = self.results.x_hat
            sigma = self.results.sigma
            scene_log_like = self.results.scene_log_like  # for debugging

            # extend the size of the posterior, etc

            n, k0 = np.shape(post)
            while k0 < self.k:
                post = np.concatenate([post, np.zeros((n, 1))], axis=1)
                log_like = np.concatenate([log_like, np.zeros((n, 1)) - np.inf], axis=1)
                log_prior = np.concatenate([log_prior, np.zeros((n, 1)) - np.inf], axis=1)
                n, k0 = np.shape(post)

                scene_log_like = np.concatenate([
                    scene_log_like, np.zeros(
                        (np.shape(scene_log_like)[0], 1)) - np.inf
                    ], axis=1)

            # extend the size of the posterior, etc
            post = np.concatenate([post, np.zeros((1, self.k))], axis=0)
            log_like = np.concatenate([log_like, np.zeros((1, self.k)) - np.inf], axis=0)
            log_prior = np.concatenate([log_prior, np.zeros((1, self.k)) - np.inf], axis=0)
            x_hat = np.concatenate([x_hat, np.zeros((event_len, self.d))], axis=0)
            sigma = np.concatenate([sigma, np.zeros((event_len, self.d))], axis=0)
            scene_log_like = np.concatenate([scene_log_like, np.zeros((event_len, self.k)) - np.inf], axis=0)
        
        return log_like,log_prior,post,x_hat,sigma,scene_log_like
    
    def verify_active_models(self,active):
        """ loop through each potentially active event model and verify 
            a model has been initialized
        """
        for k0 in active:
            if k0 not in self.event_models.keys():
                new_model = self.f_class(self.d, **self.f_opts)
                if self.model is None:
                    self.model = new_model.init_model()
                else:
                    new_model.set_model(self.model)
                self.event_models[k0] = new_model
        return None

    def clear(self):
        """ This function deletes sem from memory"""
        self.clear_event_models()
        delete_object_attributes(self.results)
        delete_object_attributes(self)


class SEM(BaseSEM):

    def __init__(self, lmda=1., alfa=10.0, f_class=CSWEvent, f_opts=None, seed=99):
        """
        lmda: float
            sCRP stickiness parameter
        alfa: float
            sCRP concentration parameter
        f_class: class
            object class that has the functions "predict" and "update".
            used as the event model
        f_opts: dictionary
            kwargs for initializing f_class
        """
        # NTF: SEM internal state
        self.f_class = f_class
        self.f_opts = f_opts
        self.lmda = lmda
        self.alfa = alfa
        self.k = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type
        self.model = None # this is the tensorflow model that gets used
        self.x_prev = None  # last scene
        self.k_prev = None  # last event type
        self.x_history = np.zeros(())
        self.results = None
        ## AB
        self.seed = seed
        self.obs_dim = 8 # state vector
        self.sch_stsize = 25
        self.lr = 0.1
        self.schlib = [CSWNet(self.sch_stsize,self.seed)] # library of schemas
        self.schema_active = self.schlib[0]


    def _calculate_unnormed_sCRP(self, prev_cluster=None):
        """ 
        
        """
        # internal function for consistency across "run" methods

        # calculate sCRP prior
        prior = self.c.copy()
        idx = len(np.nonzero(self.c)[0])  # get number of visited clusters

        if idx <= self.k:
            prior[idx] += self.alfa  # set new cluster probability to alpha

        # add stickiness parameter for n>0, only for the previously chosen event
        if prev_cluster is not None:
            prior[prev_cluster] += self.lmda

        # prior /= np.sum(prior)
        return prior
    
    def update_single_event(self, x, update=True):
        """
        x: num_events x obs_dim
        `lik` :array num_events x len(schema_lib) with log_likelihoods. 
        `prior` :1d-array with nonzero elements corresponding to active models
            size grows at every event
        `active` :list of len(schema_lib). contains ints to active models.
            nb this is a range so self.n_schemas = active[-1]
        `prior`, `log_prior`, `log_like` all grow with num of events
            should grow with num of schemas
        """
        print('=update single event')

        event = x 
        event_len = np.shape(x)[0]

        (log_like,log_prior,post,x_hat,sigma,scene_log_like
            ) = self.update_prior_and_posterior_of_event_model(x)

        """ 
        todo: currently prior grows with number of obs
        should only grow with number of schemas
        """
        prior = self._calculate_unnormed_sCRP(self.k_prev)
        print('-prior ',prior)

        # initialize likelihoods
        """ active is a list (range) of integers of models with nonzero priors 
        """
        active = np.nonzero(prior)[0]
        self.n_schemas = len(active)

        ## ~~ SEM select winning model
        # calculate likelihood of obs under each active model
        lik,_x_hat,_sigma = self.calculate_likelihoods(x,active,prior)
        # calculate log like and prior, used for deciding on event_model
        log_like[-1, :self.n_schemas] = np.sum(lik, axis=0)
        log_prior[-1, :self.n_schemas] = np.log(prior[:self.n_schemas])
        # at the end of the event, find the winning model!
        k = active_model_idx = self.get_winning_model(post,log_prior,log_like)
        print('active_model_idx',active_model_idx)
        print('lik',lik.shape)
        print('len active',self.n_schemas)
        ## ~\~ SEM calculate eventmodel likes and select winning model

        # cache for next event/story
        self.k_prev = k
        # update the prior
        self.c[k] += event_len
        
        ## ~~ gradient update winning model weights
        # update with first observation
        self.event_models[k].update_f0(x[0])
        # update with subsequent observations
        x_prev = x[0]
        for X0 in x[1:]:
            self.event_models[k].update(x_prev, X0)
            x_prev = X0
        ## ~/~ gradient update winning model weights

        # collect RESULTS 
        self.results.log_like = log_like
        self.results.log_prior = log_prior
        self.results.e_hat = np.argmax(post, axis=1)
        self.results.log_loss = logsumexp(log_like + log_prior, axis=1)
        scene_log_like[-event_len:, :self.n_schemas] = lik
        self.results.scene_log_like = scene_log_like
        x_hat[-event_len:, :] = _x_hat
        sigma[-event_len:, :] = _sigma
        self.results.x_hat = x_hat
        self.results.sigma = sigma
        return None

    def get_winning_model(self,post,log_prior,log_like):
        """ 
        splitting SEM uses argmax of posterior log probability
        nonsplitting SEM takes single event
        """
        log_post = log_prior[-1, :self.n_schemas] + log_like[-1, :self.n_schemas]
        post[-1, :self.n_schemas] = np.exp(log_post - logsumexp(log_post))
        k = np.argmax(log_post)
        self.results.post = post
        return k

    def calculate_likelihoods(self,x,active,prior):
        """ 
        updates `lik` which contains log likelihood of active models
        """
        event_len = np.shape(x)[0]
        # again, this is a readout of the model only and not used for updating,
        # but also keep track of the within event posterior
        _x_hat = np.zeros((event_len, self.d))  # temporary storre
        _sigma = np.zeros((event_len, self.d))

        ### initialize array 
        """ iteratively calculate likelihood of each obs
        """        
        lik = np.zeros((event_len, self.n_schemas))

        for ii, x_curr in enumerate(x):
            # print('new obs')

            # we need to maintain a distribution over possible event types for the current events --
            # this gets locked down after termination of the event.
            # Also: none of the event models can be updated until *after* the event has been observed

            # special case the first scene within the event
            if ii == 0:
                event_boundary = True
            else:
                event_boundary = False

            # can i pull this out of the loop?
            self.verify_active_models(active)
            
            #~# using active schema, calculate xhats. 
            """ this might just be for evaluation 
            - the reason this needs to be in this loop is `lik` 
                is updating. does that allow for different models
                to predict different timesteps within event?
            - 
            """
            if ii == 0:
                # prior to the first scene within an event having been observed
                k_within_event = np.argmax(prior) 
                _x_hat[ii, :] = self.event_models[k_within_event].predict_f0() 
            else:
                # otherwise, use previously observed scenes
                k_within_event = np.argmax(
                    np.sum(lik[:ii, :self.n_schemas], axis=0) + np.log(prior[:self.n_schemas]))
                _x_hat[ii, :] = self.event_models[k_within_event].predict_next_generative(x[:ii, :])
                
            _sigma[ii, :] = self.event_models[k_within_event].get_variance()

            #~\# using active schema, calculate xhats. 

            """
            Inference: calculate likelihood of each active event model
            `log_likelihood_sequence` makes prediction using 
            and evaluates likelihood of prediction
            """
            for k0 in active:
                model = self.event_models[k0]

                if not event_boundary:
                    lik[ii, k0] = model.log_likelihood_sequence(
                        x[:ii, :].reshape(-1, self.d), x_curr
                        )
                else:
                    lik[ii, k0] = model.log_likelihood_f0(x_curr)

        return lik,_x_hat,_sigma





class Results(object):
    """ placeholder object to store results """
    pass
