import os
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from tqdm import tqdm
# from local_event_models import GRUEvent
from sem.utils import delete_object_attributes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# NTF tensorflow implementation
from cswRNN import CSWEvent
# AB pytorch implementation
from CSW import CSWNet
import torch as tr


DEBUG = True


""" 

defn: schema is an rnn
defn: event_type is A or B
defn: state refers to csw states {1-8}
defn: obs instance of state vector 
defn: event sequence of obs. corresponds to story

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

        # loop over events

        self.init_for_boundaries(list_events)
        if DEBUG:
            event_idx_L = [0,0,0,1,1,2,2,1,1,0]
            # event_idx_L = [0,0,0,1,1,1,1,1,1,0]
            for idx,event in enumerate(list_events):
                self.update_single_event(event,event_idx=event_idx_L[idx])
        else:
            for event in list_events:
                self.update_single_event(event)
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
        # schlib always has one "inactive" schema  
        # similarly. dimensions of prior,likelihood,posterior
        ## will be len(schlib) == 1+num_active_schemas
        self.prior = np.array([self.alfa,self.alfa])
        self.schema_count = np.array([0,0])
        self.schlib = [
            CSWNet(self.sch_stsize,self.seed),
            CSWNet(self.sch_stsize,self.seed+1)
            ] 
        self.active_schema_idx = 0
        self.prev_schema_idx = None
        None 

    def get_crp_prior(self):
        """ 
        - prior calculation relies on the `schema_count`
            which is an array that counts number of times (#obs)
            each schema was used 
        - len(prior) == len(schlib)
        """
        ## case not previous not new
        prior = self.schema_count.copy().astype(float)
        ## case previous event
        if self.prev_schema_idx!=None:
            prior[self.prev_schema_idx] += self.lmda
        else: # tstep0 
            prior[0] = self.alfa
        ## case new event
        prior[-1] = self.alfa
        assert len(prior) == len(self.schlib)
        return prior

    def _calculate_unnormed_sCRP(self, prev_cluster=None):
        """ 
        self.k is max number of clusters
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

    def update_single_event(self, x, event_idx=None, update=True):
        """
        x: num_events x obs_dim
        `lik` :array num_events x len(schema_lib) with log_likelihoods. 
        `prior` :1d-array with nonzero elements corresponding to active models
            size grows at every event
        `active` :list of len(schema_lib). contains ints to active models.
            nb this is a range so self.n_schemas = active[-1]
        `prior`, `log_prior`, `log_like` all grow with num of events
            should grow with num of schemas

        NF keeps track of distributions over time
            every epoch a new row is added
            when a new model is forked, a new column is added
        """
        print('\n**process new event')

        event = x 
        event_len = np.shape(event)[0]
        
        (log_like,log_prior,post,x_hat,sigma,scene_log_like
            ) = self.update_prior_and_posterior_of_event_model(x)

        ## PRIOR CALCULATION
        prior = self._calculate_unnormed_sCRP(self.k_prev)
        prior_AB = self.get_crp_prior()

        # active
        active = np.nonzero(prior)[0]
        self.n_schemas = len(active)
        assert active[-1]==len(active)-1
        
        ## log prior
        log_prior[-1, :self.n_schemas] = np.log(prior[:self.n_schemas])
        log_prior_AB = np.log(prior_AB)
        # print('NF-logprior\n',log_prior[-1, :self.n_schemas])
        # print('AB-logprior\n',log_prior_AB,'\n')
        assert np.alltrue(log_prior_AB==log_prior[-1, :self.n_schemas])

        ## LIKELIHOOD CALCULATION
        # likelihood is calculated for each step (tsteps x nschemas)
        print('\n-NF start likelihood calc')
        lik,_x_hat,_sigma = self.calculate_likelihoods(x,active,prior)
        print('\n-AB start likelihood calc')
        log_like_AB = self.calc_likelihood(event) # (tsteps,schemas)
        print('-end likelihood calc')
        # print('NF-like',lik)
        # print('AB-like',log_like_AB,'\n')
        if self.prev_schema_idx!=None:
            assert (np.alltrue(log_like_AB==lik))
        
        ## collapse log_like over obs 
        log_like[-1, :self.n_schemas] = np.sum(lik, axis=0) 
        log_like_AB = np.sum(log_like_AB,axis=0)
        # print('NF-sum_loglike',log_like[-1, :self.n_schemas])
        # print('AB-sum_loglike',log_like_AB,'\n')
        if self.prev_schema_idx !=None:
            assert np.alltrue(log_like_AB == log_like[-1, :self.n_schemas])

        ## USE PRIOR AND LIKELIHOOD TO SELECT MODEL
        k = self.active_schema_idx = self.get_winning_model(post,log_prior,log_like)
        self.active_schema_idx = self.get_active_schema_idx(log_prior_AB,log_like_AB)
        # print('NF-schemaidx',k)
        # print('AB-schemaidx',self.active_schema_idx,'\n')
        assert (self.active_schema_idx == k)
        ## ~\~ SEM calculate eventmodel likes and select winning model

        ## specify model debug 
        if DEBUG:
            self.active_schema_idx = k = event_idx

        print(
            'num schemas',len(self.schlib),
            'schema_idx',self.active_schema_idx
            )

        ## cache for next event/story
        self.k_prev = k
        self.c[k] += event_len
        self.prev_schema_idx = self.active_schema_idx

        ## if new active_schema, update schlib
        if self.active_schema_idx == len(self.schema_count)-1:
            # print('new active schema')
            self.schema_count = np.concatenate([self.schema_count,[0]])
            self.schlib.append(CSWNet(self.sch_stsize,self.seed))
        self.schema_count[self.active_schema_idx] += event_len


        ### NF GRADIENT STEP: UPDATE WINNING MODEL WEIGHTS
        # update with first observation
        def update_model(model,event):
            print('updatef0')
            model.update_f0(event[0])
            obs_prev = event[0]
            # update with subsequent observations
            for obs in event[1:]:
                print('update')
                model.update(obs_prev, obs)
                obs_prev = obs
            return None
        print('-')
        NF_active_model = self.event_models[k]
        update_model(NF_active_model,event)
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

    def get_active_schema_idx(self,log_prior,log_like):
        """ 
        splitting SEM uses argmax of posterior log probability
        nonsplitting SEM takes single event
        """
        if self.prev_schema_idx==None:
            # handle differences with NF indexing
            return 0
        log_post = log_prior + log_like
        # post[-1, :self.n_schemas] = np.exp(log_post - logsumexp(log_post))
        k = np.argmax(log_post)
        # self.results.post = post
        return k

    #NF
    def calculate_likelihoods(self,x,active,prior):
        """ 
        updates `lik` which contains log likelihood of *all schemas*
        NB NTF calculate lik of all active schemas and an extra schema 
        """
        # print('\n\n == NF like')

        event = x
        event_len = np.shape(event)[0]
        """ NTF
        this is a readout of the model only and not used for updating,
            but also keep track of the within event posterior
        """
        _x_hat = np.zeros((event_len, self.d))  # temporary storre
        _sigma = np.zeros((event_len, self.d))

        ### initialize array 
        """ iteratively calculate likelihood of each obs
        """        
        lik = np.zeros((event_len, self.n_schemas))

        for ii, x_curr in enumerate(event):
            obs = x_curr
            # print('new obs')
            """ NTF
            need to maintain a distribution over 
            possible event types for the current events --
            this gets locked down after termination of the event.
            Also: none of the event models can be updated until 
            *after* the event has been observed
            """
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
            # print('NTF: eval lik of models',active)
            for k0 in active:
                # print('NF-sch',k0)
                model = self.event_models[k0]

                if not event_boundary:
                    lik[ii, k0] = model.log_likelihood_sequence(
                        x[:ii, :].reshape(-1, self.d), x_curr
                        )
                else:
                    lik[ii, k0] = model.log_likelihood_f0(x_curr)

        return lik,_x_hat,_sigma
    
    #AB
    def calc_likelihood(self,event):
        """ calculate likelihood for all schemas 
            - active schemas + one inactive schema
        """
        event_len = event.shape[0]
        num_schemas = len(self.event_models)
        log_like = np.zeros((event_len, num_schemas))
        for sch_idx in np.arange(num_schemas):
            print('AB-sch',sch_idx)
            model = self.event_models[sch_idx]
            log_like[:,sch_idx] = model.log_likelihood(event)
        return log_like





class Results(object):
    """ placeholder object to store results """
    pass
