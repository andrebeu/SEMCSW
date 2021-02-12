import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from tqdm import tqdm
# from local_event_models import GRUEvent
from sem.utils import delete_object_attributes
from multiprocessing import Queue, Process
# https://github.com/nicktfranklin/SEM2

# there are a ~ton~ of tf warnings from Keras, suppress them here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cswRNN import CSWEvent

class Results(object):
    """ placeholder object to store results """
    pass

""" 

defn: "event" is "story"
defn: "state" is "state"

- what is k? 
- - number of `event models` ?
- `active` is an array of keys to 'active event models' ?
"""

class BaseSEM(object):
    
    def __init__(self):
        # save_x_hat: bool
        #     save the MAP scene predictions?
        # :param save_x_hat: boolean (default False) normally, we don't save this as the interpretation can be tricky
        #     N.b: unlike the posterior calculation, this is done at the level of individual scenes within the
        #     events (and not one per event)
        self.save_x_hat = False
        None

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

    def run_w_boundaries(self, list_events, progress_bar=True, 
        leave_progress_bar=True, 
        generative_predicitons=False, minimize_memory=False):
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

        progress_bar: bool
            use a tqdm progress bar?

        leave_progress_bar: bool
            leave the progress bar after completing?

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
        if minimize_memory:
            self.clear_event_models()

    def update_prior_and_posterior_of_event_model(self,x):
        """ 
        """
        n_scene = np.shape(x)[0]
        ## update prior and posterior of event model
        self.k += 1
        self._update_state(x, self.k)

        # pull the relevant items from the results
        if self.results is None:
            self.results = Results()
            post = np.zeros((1, self.k))
            log_like = np.zeros((1, self.k)) - np.inf
            log_prior = np.zeros((1, self.k)) - np.inf
            if self.save_x_hat:
                x_hat = np.zeros((n_scene, self.d))
                sigma = np.zeros((n_scene, self.d))
                scene_log_like = np.zeros((n_scene, self.k)) - np.inf # for debugging
        
        else:
            post = self.results.post
            log_like = self.results.log_like
            log_prior = self.results.log_prior

            if self.save_x_hat:
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

                if self.save_x_hat:
                    scene_log_like = np.concatenate([
                        scene_log_like, np.zeros((np.shape(scene_log_like)[0], 1)) - np.inf
                        ], axis=1)

            # extend the size of the posterior, etc
            post = np.concatenate([post, np.zeros((1, self.k))], axis=0)
            log_like = np.concatenate([log_like, np.zeros((1, self.k)) - np.inf], axis=0)
            log_prior = np.concatenate([log_prior, np.zeros((1, self.k)) - np.inf], axis=0)
            if self.save_x_hat:
                x_hat = np.concatenate([x_hat, np.zeros((n_scene, self.d))], axis=0)
                sigma = np.concatenate([sigma, np.zeros((n_scene, self.d))], axis=0)
                scene_log_like = np.concatenate([scene_log_like, np.zeros((n_scene, self.k)) - np.inf], axis=0)
        
        return log_like,log_prior,post,x_hat,sigma,scene_log_like
    
    def verify_active_models(self,active):
        """# loop through each potentially active event model and verify 
            a model has been initialized"""
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

    def __init__(self, lmda=1., alfa=10.0, f_class=CSWEvent, f_opts=None):
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
        self.save_x_hat = True

        self.lmda = lmda
        self.alfa = alfa

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_opts = f_opts

        # SEM internal state
        #
        self.k = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type
        self.model = None # this is the tensorflow model that gets used

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type

        self.x_history = np.zeros(())

        # instead of dumping the results, store them to the object
        self.results = None

    def _calculate_unnormed_sCRP(self, prev_cluster=None):
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

        :param x: this is an n x d array of the n scenes in an event
        :param update: boolean (default True) update the prior and posterior of the event model
        
        :return:

        NB this is called once for every story/event (i.e. number of trials = n_train+n_test)
        """
        print('update single event')

        n_scene = np.shape(x)[0]

        (log_like,log_prior,post,x_hat,sigma,scene_log_like
            ) = self.update_prior_and_posterior_of_event_model(x)

        # DELTA: calculate un-normed sCRP prior
        prior = self._calculate_unnormed_sCRP(self.k_prev)

        # initialize likelihoods
        active = np.nonzero(prior)[0]

        ## ~~ SEM select winning model
        # calculate likelihood of obs under each active model
        lik,_x_hat,_sigma = self.calculate_likelihoods(x,active,prior)
        # calculate log like and prior, used for deciding on event_model
        log_like[-1, :len(active)] = np.sum(lik, axis=0)
        log_prior[-1, :len(active)] = np.log(prior[:len(active)])
        # at the end of the event, find the winning model!
        k = active_model_idx = self.get_winning_model(post,log_prior,log_like,active)
        ## ~\~ SEM calculate eventmodel likes and select winning model

        # cache for next event/story
        self.k_prev = k
        # update the prior
        self.c[k] += n_scene
        
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
        if self.save_x_hat:
            x_hat[-n_scene:, :] = _x_hat
            sigma[-n_scene:, :] = _sigma
            scene_log_like[-n_scene:, :len(active)] = lik
            self.results.x_hat = x_hat
            self.results.sigma = sigma
            self.results.scene_log_like = scene_log_like

        return None

    def get_winning_model(self,post,log_prior,log_like,active):
        """ splitting SEM uses argmax
        nonsplitting SEM takes single event
        """
        log_post = log_prior[-1, :len(active)] + log_like[-1, :len(active)]
        post[-1, :len(active)] = np.exp(log_post - logsumexp(log_post))
        k = np.argmax(log_post)
        self.results.post = post
        return k

    def calculate_likelihoods(self,x,active,prior):
        """ 
        """
        n_scene = np.shape(x)[0]
        # again, this is a readout of the model only and not used for updating,
        # but also keep track of the within event posterior
        if self.save_x_hat:
            _x_hat = np.zeros((n_scene, self.d))  # temporary storre
            _sigma = np.zeros((n_scene, self.d))

        ### initialize array 
        """ iteratively calculate likelihood of each obs
        """        
        lik = np.zeros((n_scene, len(active)))

        for ii, x_curr in enumerate(x):
            print('new obs')

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
            
            ### ~~~~~ Start ~~~~~~~###

            ## prior to updating, pull x_hat based on the ongoing estimate of the event label
            if ii == 0:
                # prior to the first scene within an event having been observed
                k_within_event = np.argmax(prior)  
            else:
                # otherwise, use previously observed scenes
                k_within_event = np.argmax(
                    np.sum(lik[:ii, :len(active)], axis=0
                        ) + np.log(prior[:len(active)])
                    )
            
            if self.save_x_hat:
                if event_boundary:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_f0()
                else:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_next_generative(x[:ii, :])
                _sigma[ii, :] = self.event_models[k_within_event].get_variance()

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









class NoSplitSEM(BaseSEM):

    def __init__(self, lmda=None, alfa=None, f_class=CSWEvent, f_opts=None):
        """
        lmda: float
            placeholder
        alfa: float
            placeholder
        f_class: class
            object class that has the functions "predict" and "update".
            used as the event model
        f_opts: dictionary
            kwargs for initializing f_class
        """
        self.save_x_hat = True

        self.f_class = f_class
        self.f_opts = f_opts

        # SEM internal state
        #
        self.k = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type
        self.model = None # instance of tensorflow model 

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type

        self.x_history = np.zeros(())

        # instead of dumping the results, store them to the object
        self.results = None

    def update_single_event(self, x, update=True):
        """

        :param x: this is an n x d array of the n scenes in an event
        :param update: boolean (default True) update the prior and posterior of the event model
        :return:
        """

        n_scene = np.shape(x)[0]

        (log_like,log_prior,post,x_hat,sigma,scene_log_like
            ) = self.update_prior_and_posterior_of_event_model(x)

        ## DELTA: SEM calculates calculate un-normed sCRP prior
        prior = [1]

        # likelihood
        active = np.nonzero(prior)[0]
        lik = np.zeros((n_scene, len(active)))

        # again, this is a readout of the model only and not used for updating,
        # but also keep track of the within event posterior
        if self.save_x_hat:
            _x_hat = np.zeros((n_scene, self.d))  # temporary storre
            _sigma = np.zeros((n_scene, self.d))

        ## loop over samples?
        for ii, x_curr in enumerate(x):

            # we need to maintain a distribution over possible event types for the current events --
            # this gets locked down after termination of the event.
            # Also: none of the event models can be updated until *after* the event has been observed

            # special case, first scene within event
            if ii == 0:
                event_boundary = True
            else:
                event_boundary = False

            # 
            self.verify_active_models(active)

            ### ~~~~~ Start ~~~~~~~###

            ## prior to updating, pull x_hat based 
            # on the ongoing estimate of the event label
            k_within_event = 0
            
            if self.save_x_hat:
                if event_boundary:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_f0()
                else:
                    _x_hat[ii, :] = self.event_models[k_within_event].predict_next_generative(x[:ii, :])
                _sigma[ii, :] = self.event_models[k_within_event].get_variance()


            ## Update the model, inference first!
            for k0 in active:
                # get the log likelihood for each event model
                model = self.event_models[k0]

                if not event_boundary:
                    # log_likelihood sequence makes the model prediction internally
                    # using predict_next_generative, and evaluates the likelihood of the prediction
                    lik[ii, k0] = model.log_likelihood_sequence(x[:ii, :].reshape(-1, self.d), x_curr)
                else:
                    lik[ii, k0] = model.log_likelihood_f0(x_curr)
            

        # cache the diagnostic measures
        log_like[-1, 0] = np.sum(lik, axis=0)

        # calculate the log prior
        log_prior[-1, :len(active)] = np.log(prior[:len(active)])
        
        ## delta: at the end of the event, find the winning model!
        k = self.get_winning_model(post,log_prior,log_like,active)
        
        ## ~common
        # update the prior
        self.c[k] += n_scene
        # cache for next event
        self.k_prev = k

        # update the winning model's estimate
        self.event_models[k].update_f0(x[0])
        x_prev = x[0]
        for X0 in x[1:]:
            self.event_models[k].update(x_prev, X0)
            x_prev = X0
        self.results.log_like = log_like
        self.results.log_prior = log_prior
        self.results.e_hat = np.argmax(post, axis=1)
        self.results.log_loss = logsumexp(log_like + log_prior, axis=1)

        if self.save_x_hat:
            x_hat[-n_scene:, :] = _x_hat
            sigma[-n_scene:, :] = _sigma
            scene_log_like[-n_scene:, :len(active)] = lik
            self.results.x_hat = x_hat
            self.results.sigma = sigma
            self.results.scene_log_like = scene_log_like

        ## \~common
        return None

    def get_winning_model(self,post,log_prior,log_like,active):
        log_post = log_prior[-1, :len(active)] + log_like[-1, :len(active)]
        post[-1, :len(active)] = np.exp(log_post - logsumexp(log_post))
        k = 0
        self.results.post = post
        return k

    


def no_split_sem_worker(queue, x, sem_init_kwargs=None, run_kwargs=None):
    if sem_init_kwargs is None:
        sem_init_kwargs=dict()
    if run_kwargs is None:
        run_kwargs=dict()
    
    sem_model = NoSplitSEM(**sem_init_kwargs)
    sem_model.run_w_boundaries(x, **run_kwargs)
    queue.put(sem_model.results)


def no_split_sem_run_with_boundaries(x, sem_init_kwargs=None, run_kwargs=None):
    """ this initailizes SEM, runs the main function 'run_w_boundaries', and
    returns the results object within a seperate process.
    
    See help on SEM class and on subfunction 'run_w_boundaries' for more detail on the 
    parameters contained in 'sem_init_kwargs'  and 'run_kwargs', respectively.

    """
    
    q = Queue()
    p = Process(target=no_split_sem_worker, args=[q, x], 
                kwargs=dict(sem_init_kwargs=sem_init_kwargs, run_kwargs=run_kwargs))
    p.start()
    return q.get()


def sem_worker(queue, x, sem_init_kwargs=None, run_kwargs=None):
    if sem_init_kwargs is None:
        sem_init_kwargs=dict()
    if run_kwargs is None:
        run_kwargs=dict()
    
    sem_model = SEM(**sem_init_kwargs)
    sem_model.run_w_boundaries(x, **run_kwargs)
    queue.put(sem_model.results)


def sem_run_with_boundaries(x, sem_init_kwargs=None, run_kwargs=None):
    """ this initailizes SEM, runs the main function 'run_w_boundaries', and
    returns the results object within a seperate process.
    
    See help on SEM class and on subfunction 
    'run_w_boundaries' for more detail on the 
    parameters contained in 'sem_init_kwargs'  
    and 'run_kwargs', respectively.

    """
    
    q = Queue()
    p = Process(target=sem_worker, args=[q, x], 
                kwargs=dict(
                    sem_init_kwargs=sem_init_kwargs, 
                    run_kwargs=run_kwargs)
                )
    p.start()
    return q.get()



