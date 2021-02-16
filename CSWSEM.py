
import os
import numpy as np
import torch as tr



DEBUG = True


""" 
notes: 
- implement conceptual replication
    - only diff is LSTM training procedure
    - todo next: prepare assertion script
        - imports current csw task
        - init from existing nb
        - prediction error / variance estimate
- run sem vs lstm
    - small gridsearch 
        - goal find dynamic range
        - caution/study "sem variance" problem

"""

class CSWNet(tr.nn.Module):

    def __init__(self,stsize,seed):

        self.seed = seed
        super().__init__()
        tr.manual_seed(seed)
        ## network parameters
        self.stsize = stsize
        self.smdim = 12
        ## init setting
        self.is_trained = False
        self._build()

        return None

    def _build(self):
        ## architecture setup
        # embedding handled within
        self.input_embed = tr.nn.Embedding(self.smdim,self.stsize)
        self.lstm = tr.nn.LSTMCell(self.stsize,self.stsize)
        self.init_lstm = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
        self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.smdim,bias=False)
        return None

    # def embed(events_int):
    #     return events_vec

""" 
    def update(self)
    def compute_loglike(self,events)

"""

    def forward(self,event):
        ''' main wrapper 
        takes event, returns event_hat
        event is [tsteps,scene_dim]
        embed ints
        '''
        state_emb = self.input_embed(state_int)
        h_lstm,c_lstm = self.init_lstm # rnn state
        outputs = -tr.ones(len(state_emb),self.stsize)
        # explicit unroll not necessary
        for tstep in range(len(state_emb)):
            h_lstm,c_lstm = self.lstm(state_emb[tstep],(h_lstm,c_lstm))
            outputs[tstep] = h_lstm
        outputs = self.ff_hid2ulog(outputs)
        return outputs


    def fast_mvnorm_diagonal_logprob(self, x, variances):
        """
        Assumes a zero-mean mulitivariate normal with a diagonal covariance function
        Parameters:
            x: array, shape (D,)
                observations
            variances: array, shape (D,)
                Diagonal values of the covariance function
        output
        ------
            log-probability: float
        """
        log_2pi = np.log(2.0 * np.pi)
        return -0.5 * (log_2pi * np.shape(x)[0] + np.sum(np.log(variances) + (x**2) / variances ))

    def map_variance(self, samples, nu0, var0):
        """
        This estimator assumes an scaled inverse-chi squared prior over the
        variance and a Gaussian likelihood. The parameters d and scale
        of the internal function parameterize the posterior of the variance.
        Taken from Bayesian Data Analysis, ch2 (Gelman)

        samples: N length array or NxD array, where N is the number of 
                 samples and D is the dimensions
        nu0: prior degrees of freedom
        var0: prior scale parameter

        returns: float or D-length array, mode of the posterior

        ## Calculation ##

        the posterior of the variance is thus (Gelman, 2nd edition, page 50):
            
            p(var | y) ~ Inv-X^2(nu0 + n, (nu0 * var0 + n * v) / (nu0 + n) )

        where n is the sample size and v is the empirical variance.  The 
        mode of this posterior simplifies to:

            mode(var|y) = (nu0 * var0 + n * v) / (v0 + n + 2)

        which is just a weighted average of the two modes

        """

        # get n and v from the data
        n = np.shape(samples)[0]
        v = np.var(samples, axis=0)

        mode = (nu0 * var0 + n * v) / (nu0 + n + 2)
        return mode

    def _update_variance(self):
        if np.shape(self.prediction_errors)[0] > 1:
            self.Sigma = self.map_variance(self.prediction_errors, self.var_df0, self.var_scale0)





""" 
"""

class SEM(SEMBase):

    def __init__(self, lmda=1., alfa=10.0, f_opts=None, seed=99, nosplit=True):
        """
        """
        SEMBase.__init__()
        # NTF: SEM internal state
        self.seed = seed
        self.lmda = lmda
        self.alfa = alfa
        self.obs_dim = 8
        self.sch_stsize = 25
        self.lr = 0.1
        self.nosplit = nosplit
        """
        schlib always has one "inactive" schema  
         similarly, dimensions of prior,likelihood,posterior
         will be len(schlib) == 1+num_active_schemas
        """
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

    def get_active_schema_idx(self,log_prior,log_like):
        """ 
        splitting SEM uses argmax of posterior log probability
        nonsplitting SEM takes single event
        """
        if self.nosplit:
            return 0
        # edge case first trial
        if self.prev_schema_idx==None:
            return 0
        log_post = log_prior + log_like
        return np.argmax(log_post)

    def calc_likelihood(self,event):
        """ calculate likelihood for all schemas 
            - active schemas + one inactive schema
        """
        print('\n\n == AB like')
        event_len = event.shape[0]
        num_schemas = len(self.schlib)
        log_like = np.zeros((event_len, num_schemas))
        for sch_idx in np.arange(num_schemas):
            log_like[:,sch_idx] = self.calc_likelihood_model(event,sch_idx)
        return log_like

    def calc_likelihood_model(self, event, sch_idx):
        """ 
        calculates likelihood of event under model
        Xp : current observation (target)
        X  : observation history (input)

        NB: NF calculated logprob for each tstep separately
            and then summed over tsteps. that would require doing 
            multiple unrolls of the RNN. 
            Instead I calcualte model output sequence and sum within here
        """

        model = self.schlib[sch_idx]

        ## case: new inactive schema
        if not model.is_trained:
            return norm(0, self.variance_prior_mode ** 0.5).logpdf(Xp).sum()
        
        ## case: reused schema
        event_hat = model.forward(event)

        # calculate probability
        logprob = 0
        for scene,scene_hat in zip(event,event_hat):
            logprob += self.fast_mvnorm_diagonal_logprob(
                            scene.reshape(-1) - scene_hat.reshape(-1), 
                        self.Sigma)
        return logprob


    # run functions

    def run_trial(self, event, event_idx=None):
        """ 
        todo: change event to event_int
        given an event, runs a single trial
        """
        print('\n**process new event')

        event_len = np.shape(event)[0]

        # prior
        prior = self.get_crp_prior()
        log_prior = np.log(prior)

        # likelihood
        log_like = self.calc_likelihood(event) # (tsteps,schemas)
        log_like = np.sum(log_like,axis=0) 

        # select model
        self.active_schema_idx = self.get_active_schema_idx(log_prior,log_like)
        self.prev_schema_idx = self.active_schema_idx

        if DEBUG:
            self.active_schema_idx = event_idx
        
        print(
            'num schemas',len(self.schlib),
            'schema_idx',self.active_schema_idx
            )

        ## if new active_schema, update schlib
        if self.active_schema_idx == len(self.schema_count)-1:
            # print('new active schema')
            self.schema_count = np.concatenate([self.schema_count,[0]])
            self.schlib.append(CSWNet(self.sch_stsize,self.seed))
        self.schema_count[self.active_schema_idx] += event_len

        ### GRADIENT STEP: UPDATE WINNING MODEL WEIGHTS
      
        return None

    def run_exp(self, events):
        """
        wrapper for run_trial
        """
        # loop over events
        for event in events:
            _ = self.run_trial(event)
        return None
  



