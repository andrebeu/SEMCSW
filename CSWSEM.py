import time
import os
import numpy as np
import torch as tr

from scipy.stats import norm


"""
event (len6): B,L,2,3,4,E
event_hat (len5): Lh,2h,3h,4h,Eh
event_target (len5): Lt,...,Et
"""

## common across all objects
OBSDIM = 10



class SEM(object):

    def __init__(self, nosplit, lmda, alfa, rnn_kwargs, seed):
        """
        """
        self.seed = seed
        tr.manual_seed(seed)
        np.random.seed(seed)
        ## event RNN
        self.rnn_kwargs = rnn_kwargs
        self.stsize = rnn_kwargs['stsize']
        self.learn_rate = rnn_kwargs['learn_rate']
        self.pdim = rnn_kwargs['pdim']
        # SEMBase.__init__()
        self.nosplit = nosplit
        # params
        self.lmda = lmda
        self.alfa = alfa
        # hopefully do not need obsdim and stsize
        self.obsdim = OBSDIM
        # collect sem data; locals() returns kwargs dict
        sem_kwargs = locals()
        sem_params = {**sem_kwargs,**rnn_kwargs}
        self.data = SEMData(sem_params)
        """
        schlib always has one "inactive" schema  
         similarly, dimensions of prior,likelihood,posterior
         will be len(schlib) == 1+num_active_schemas
        """
        self.active_schema_idx = 0
        self.prev_schema_idx = None
        self._init_schlib()
        self.in_layer = tr.nn.Linear(self.obsdim,self.obsdim)
        return None 

    def _init_schlib(self):
        self.prior = np.array([self.alfa,self.alfa])
        self.schema_count = np.array([0,0])
        self.schlib = [
            CSWSchema(**self.rnn_kwargs),
            CSWSchema(**self.rnn_kwargs)
            ] 
        return None

    def get_logpriors(self):
        """ 
        3 cases:
            1) new/inactive schema:
                prior = alfa
            2) old schema, active on previous trial:
                prior = count + lamda
            3) old schema, not active on previous trial:
                prior = count
        - CRP calculation relies on `schema_count`
            which is an array that counts number of times (#obs)
            each schema was used 
        - len(prior) == len(schlib)
        """
        ## init prior
        prior = self.schema_count.copy().astype(float)
        ## active schemas
        if type(self.prev_schema_idx)==int:
            prior[self.prev_schema_idx] += self.lmda
        elif self.prev_schema_idx==None: # tstep0 
            prior[0] = self.alfa
        ## new/inactive schema 
        prior[-1] = self.alfa
        assert len(prior) == len(self.schlib)
        return np.log(prior)

    def get_loglikes(self,event):
        """ wrapper around schema.calc_like
            - active schemas + one inactive schema
        """
        loglikes = -99*np.ones(len(self.schlib))
        for sch_idx,schema in enumerate(self.schlib):
            loglikes[sch_idx] = schema.calc_event_loglike(event)
        return loglikes

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
        # print(log_prior,log_like,log_post)
        return np.argmax(log_post)

    def select_schema(self,log_prior,log_like):
        """ returns index of active schema
        updates previous_schema_index
        if new schema, append to library
        """
        # select model
        active_schema_idx = self.get_active_schema_idx(log_prior,log_like)
        # update previous schema index
        self.prev_schema_idx = active_schema_idx
        # if new active_schema, update schlib (need to wrap this)
        if active_schema_idx == len(self.schema_count)-1:
            self.schema_count = np.concatenate([self.schema_count,[0]])
            self.schlib.append(CSWSchema(**self.rnn_kwargs))
        return active_schema_idx

    # run functions

    def forward_trial(self, event):
        """ 
        wrapper for within trial calculations 
        records trial data
        """
        # embed
        event = tr.Tensor(event).unsqueeze(1) # include batch dim
        assert event.shape == (6,1,10)
        # prior & likelihood of each schema
        log_priors = self.get_logpriors() 
        log_likes = self.get_loglikes(event) 
        assert len(log_priors) == len(log_likes) == len(self.schlib)
        # select schema and update count
        active_schema_idx = self.select_schema(log_priors,log_likes)
        # NB count update
        self.schema_count[active_schema_idx] += len(event) 
        active_schema = self.schlib[active_schema_idx]
        #
        print('sch',active_schema_idx,'\nlpriors',log_priors,'\nllikes',log_likes)
        #
        self.data.record_trial('active_schema',active_schema_idx)
        # gradient step
        loss = active_schema.backprop(event)
        self.data.record_trial('loss',loss)
        return None

    def forward_exp(self,exp,curr):
        """
        wrapper for run_trial
        """
        # loop over events
        lossL = []
        for trial_num,event in enumerate(exp):
            print('\n\nTRIAL',trial_num,'nsc',len(self.schlib))
            self.data.new_trial(trial_num)
            self.data.record_trial('curriculum',curr[trial_num])
            self.forward_trial(event)
        # exp recording sem params 
        # close data
        exp_data = self.data.finalize()
        return exp_data
  


class CSWSchema(tr.nn.Module):

    def __init__(self,pdim,stsize,learn_rate):
        super().__init__()
        ## network parameters
        self.obsdim = OBSDIM
        self.pdim = pdim
        self.stsize = stsize
        self.learn_rate = learn_rate
        # setup
        self._build()
        # backprop
        self.lossop = tr.nn.MSELoss()
        self.optiop = tr.optim.Adam(
            self.parameters(), lr=self.learn_rate)
        # init settings
        self.is_active = False
        # loglike params (from NF)
        # NB a function of obs dimension
        self.variance_prior_mode = 1/self.obsdim
        self.var_df0 = 1
        self.var_scale0 = 0.3
        self.sigma = np.ones(self.obsdim)/self.obsdim
        return None

    def _build(self):
        ## architecture setup
        # embedding handled within
        self.in_layer = tr.nn.Linear(self.obsdim,self.pdim)
        self.lstm = tr.nn.LSTM(self.pdim,self.stsize)
        self.init_lstm = tr.nn.Parameter(tr.rand(2,1,1,self.stsize),requires_grad=True)
        self.out_layer = tr.nn.Linear(self.stsize,self.obsdim)
        return None


    def forward(self,event):
        ''' 
        receives full event, returns ehat
            len(event)
        main wrapper 
        takes event, returns event_hat
        event is [tsteps,scene_dim]
        embed ints
        '''
        eventx = event[:-1]
        # prop
        ehat = eventx
        h_lstm,c_lstm = self.init_lstm 
        ehat = self.in_layer(ehat)
        ehat,(h_lstm,c_lstm) = self.lstm(ehat,(h_lstm,c_lstm))
        ehat = self.out_layer(ehat)
        # numpy mode for like
        assert len(ehat) == len(event)-1
        return ehat

    def backprop(self,event):
        """ update weights 
        """
        ehat = self.forward(event)
        etarget = event[1:]
        assert ehat.shape == etarget.shape == (5,1,10)
        ## 
        self.update_variance(ehat,etarget)
        ## back prop
        loss = 0
        self.optiop.zero_grad()
        for tstep in range(len(ehat)):
            loss += self.lossop(ehat[tstep],etarget[tstep])
            loss.backward(retain_graph=True)
        self.optiop.step()
        self.is_active = True
        # update variance
        return loss.detach().numpy()

    # like funs

    def calc_event_loglike_inactive(self,event):
        """ 
        - evaluate likelihood of each scene under normal_pdf
            summing over components (obsdim)
        - NB currently evaluating Begin,...,4
            i.e. leaveing out end
        returns int, loglike of entire event
        """
        eventx = event[:-1]
        assert self.is_active == False
        loglike_event = 0
        for scene in eventx:
            normal_pdf = norm(0, self.variance_prior_mode ** 0.5)
            scene_loglike = normal_pdf.logpdf(scene).sum()
            loglike_event += scene_loglike
        return loglike_event

    def calc_event_loglike(self,event):
        """ 
        NB self.sigma is a function of prediction error 
            and thus changes over time
        - not fully confident with handling of inactive schema 
        returns int, loglike of entire event
        """
        ## inactive schema:
        # if not self.is_active:
        #     return self.calc_event_loglike_inactive(event)
        # like of schema is function of eventPE
        ehat = self.forward(event).detach().numpy() 
        etarget = event[1:].detach().numpy() # rm BEGIN
        assert ehat.shape == etarget.shape == (5,1,10) # t,batch,obs
        event_pes = np.sum([etarget,-ehat],0).squeeze() # t,obs 
        # loop over scenes of event        
        loglike_event = 0
        for scene_pe in event_pes:
            loglike_event += self.fast_mvnorm_diagonal_logprob(scene_pe, self.sigma)
        return loglike_event

    # NF misc

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

    def update_variance(self,ehat,etarget):
        ehat = ehat.detach().numpy().squeeze()
        etarget = etarget.detach().numpy().squeeze()
        PE = ehat - etarget
        ##
        self.sigma = self.map_variance(PE, self.var_df0, self.var_scale0)
        return None



class CSWTask():
    """ 
    """

    def __init__(self   ):
        A1,A2,B1,B2 = self._init_paths()
        self.paths = [[A1,A2],[B1,B2]]
        # keep obs dim fixed: NF plate's formula 
        # calculations assumes 10 
        self.obsdim = OBSDIM
        self.tsteps = len(self.paths[0][0])
        self.exp_int = None
        return None

    # helper init
    def _init_paths(self):
        """ 
        begin -> locA -> node11, node 21, node 31, end
        begin -> locA -> node12, node 22, node 32, end
        begin -> locB -> node11, node 22, node 31, end
        begin -> locB -> node12, node 21, node 32, end
        """
        begin,locA,locB = 0,1,2
        node11,node12 = 3,4
        node21,node22 = 5,6
        node31,node32 = 7,8
        end = 9
        A1 = np.array([begin,locA,
            node11,node21,node31,end
            ])
        A2 = np.array([begin,locA,
            node12,node22,node32,end
            ])
        B1 = np.array([begin,locB,
            node11,node22,node31,end
            ])
        B2 = np.array([begin,locB,
            node12,node21,node32,end
            ])
        return A1,A2,B1,B2

    def _init_emat(self):
        self.n_obs = 11
        self.embed_mat = np.random.normal(
            loc=0., scale=1./np.sqrt(self.obsdim), 
            size=(self.n_obs, self.obsdim)
            )
        return None

    # used to generate exp
    def get_curriculum(self,condition,n_train,n_test):
        """ 
        order of events
        NB blocked: ntrain needs to be divisible by 4
        """
        curriculum = []   
        if condition == 'blocked':
            assert n_train%4==0
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4 )
        elif condition == 'early':
            curriculum =  \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 4)
        elif condition == 'middle':
            curriculum =  \
                [0, 1] * (n_train // 8) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4) + \
                [0, 1] * (n_train // 8)
        elif condition == 'late':
            curriculum =  \
                [0, 1] * (n_train // 4) + \
                [0] * (n_train // 4) + \
                [1] * (n_train // 4)
        elif condition == 'interleaved':
            curriculum = [0, 1] * (n_train // 2)
        elif condition == 'single': ## DEBUG
            curriculum =  \
                [0] * (n_train) 
        else:
            print('condition not properly specified')
            assert False
        # 
        curriculum += [int(np.random.rand() < 0.5) for _ in range(n_test)]
        return np.array(curriculum)

    def embed_path(self,path_int):
        """ 
        given exp_int (tsteps)
        returns exp (ntrials,tsteps,obsdim)
        """
        return self.embed_mat[path_int]

    # main wrapper
    def generate_experiment(self,condition,n_train,n_test):
        """ 
        exp arr of events (vec)
        returns [n_train+n_test,tsteps,obsdim]
        """
        self._init_emat()
        # get curriculum
        n_trials = n_train+n_test
        curr = self.get_curriculum(condition,n_train,n_test)
        # generate trials
        exp = -np.ones([n_trials,self.tsteps,self.obsdim])
        self.exp_int = -np.ones([n_trials,self.tsteps],dtype=int)
        for trial_idx in range(n_train+n_test):
            # select A1,A2,B1,B2
            event_type = curr[trial_idx]
            path_type = np.random.randint(2)
            path_int = self.paths[event_type][path_type]
            # embed
            self.exp_int[trial_idx] = path_int
            exp[trial_idx] = self.embed_path(path_int)
        return exp,curr




class SEMData(object):

    def __init__(self, semparams):
        self.semparams = semparams
        self.sem_data = []
        return None

    def new_trial(self,trial_num):
        """ 
        init dict containing trial data 
        NB** trial_data is first appended to sem_data
            and is then modified at runtime.
        """
        self.trial_data = {'trial':trial_num}
        self.sem_data.append(self.trial_data) 

    def record_trial(self,key,value):
        """ populate trial dict 
        NB** trial_data is a dict that has already been 
        appended to sem_data. this method modifies self.trial_data 
        """
        self.trial_data[key] = value

    def record_exp(self,key,value):
        """ record experiment wide keyvalue in sem_data """
        for trial_dict in self.sem_data:
            trial_dict.update({key:value})
        return None

    def record_semparams(self):
        """ update each dict in sem_data
        to include sem semparams
        """
        semparams = {k:v for k,v in self.semparams.items() if k!='self'}
        for trial_dict in self.sem_data:
            trial_dict.update(semparams)
        return None

    def finalize(self):
        self.record_semparams()
        return self.sem_data




