import os
import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, Lambda, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import l2_normalize
from sem.utils import unroll_data, get_prior_scale, delete_object_attributes

### there are a ~ton~ of tf warnings from Keras, suppress them here
print("TensorFlow Version: {}".format(tf.__version__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SharedObj(object):
    """
    NF helper methods
    """
    def __init__(self, d, var_df0=None, var_scale0=None, 
        optimizer=None, n_epochs=10, init_model=False,
        kernel_initializer='glorot_uniform', l2_regularization=0.00, 
        batch_size=32, prior_log_prob=None,reset_weights=False, 
        batch_update=True, optimizer_kwargs=None, 
        variance_prior_mode=None, variance_window=None):
        """

        :param d: dimensions of the input space
        """
        self.d = d
        self.f_is_trained = False
        self.f0_is_trained = False
        self.f0 = np.zeros(d)

        #### ~~~ Variance Prior Parameters ~~~~ ###
        # in practice, only the mode of the variance prior
        # matters at all. Changing the df/scale but maintaining
        # the mode has no effect on the model behavior or the 
        # implied variance. (It does facilitate magnitude of the 
        # log likelihood, but not the dynamic range of the 
        # log-likelihoods). As such, it is convient to fix the
        # prior df and solve for the scale for some desired variance.
        
        
        # allow for set DF and set scale to override the variance prior mode
        if (var_df0 is not None) and (var_scale0 is not None):
            variance_prior_mode = var_df0 / (var_df0 + 2) * var_scale0
        
        elif variance_prior_mode is None:
            # in simulations, it is often convient to approximately 
            # normalize the stim to unit length, or 
            # X ~ N(0, (1/d) * I)
            variance_prior_mode = 1 / d 
        self.variance_prior_mode = variance_prior_mode

        if var_df0 is None:
            var_df0 = 1
        self.var_df0 = var_df0

        if var_scale0 is None:
            var_scale0 = get_prior_scale(self.var_df0, variance_prior_mode)
        self.var_scale0 = var_scale0

        # also set a default prior log probability, inferred from the prior variance
        # new way!! evaluate the probability of the trace as given a zero mean gaussian
        # vector with the variance prior mode
        # if prior_log_prob is None:
            # # this is a decent approximation of what a random normalized vector would
            # # under the generative process of X ~ N(0, var_scale0 * I),
            # # which gives (in expectation) unit vectors
            # 
            # # note, norm uses standard deviation, not variance
            # prior_log_prob = norm(0, variance_prior_mode ** 0.5).logpdf(
            #     variance_prior_mode ** 0.5) * d
            
        self.prior_probability = prior_log_prob

        # how many observations do we consider in calculating the variance?
        if variance_window is None:
            variance_window = int(1e6) # this is plausiblely large...
        self.variance_window = variance_window
        
        #### ~~~ END Variance Prior Parameters ~~~~ ###

        self.x_history = [np.zeros((0, self.d))]

        if (optimizer is None) and (optimizer_kwargs is None):
            optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
        elif (optimizer is None) and not (optimizer_kwargs is None):
            optimizer = Adam(**optimizer_kwargs)
        elif (optimizer is not None) and (type(optimizer) != str):
            optimizer = optimizer()

        self.compile_opts = dict(optimizer=optimizer, loss='mean_squared_error')
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = regularizers.l2(l2_regularization)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)

        self.d = d
        self.reset_weights = reset_weights
        self.batch_update = batch_update
        self.training_pairs = []
        self.prediction_errors = np.zeros((0, self.d), dtype=np.float)
        self.model_weights = None

        # initialize the covariance with the mode of the prior distribution
        self.Sigma = np.ones(d) * var_df0 * var_scale0 / (var_df0 + 2)

        self.is_visited = False  # governs the special case of model's first prediction (i.e. with no experience)

        # switch for inheritance -- don't want to init the model for sub-classes
        if init_model:
            self.init_model()

        # generate a vector to act as a placeholder for time-points
        # prior to the start of the event so as to allow the network 
        # to implicitly learn a hidden state. We'll assume this
        # vector has length appox equal to 1
        self.filler_vector = np.random.randn(self.d) / np.sqrt(self.d)


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


class TFobj(SharedObj):
    """ 
    anything tensorflow specific
    """
    def __init__(self, d, var_df0=None, var_scale0=None, t=3,
        optimizer=None, n_epochs=10, l2_regularization=0.00, 
        batch_size=32,kernel_initializer='glorot_uniform', 
        init_model=False, prior_log_prob=None, reset_weights=False,
        batch_update=True, optimizer_kwargs=None,variance_prior_mode=None, 
        variance_window=None):

        SharedObj.__init__(self, d, var_df0=var_df0, var_scale0=var_scale0,
            optimizer=optimizer, n_epochs=n_epochs,init_model=False, 
            kernel_initializer=kernel_initializer,l2_regularization=l2_regularization, 
            prior_log_prob=prior_log_prob,reset_weights=reset_weights, 
            batch_update=batch_update, optimizer_kwargs=optimizer_kwargs, 
            variance_prior_mode=variance_prior_mode,variance_window=variance_window)

        self.t = t
        self.n_epochs = n_epochs

        # list of clusters of scenes:
        # each element of list = history of scenes for given cluster
        # history = N x D tensor, N = # of scenes in cluster, D = dimension of single scene
        #
        self.x_history = [np.zeros((0, self.d))]
        self.batch_size = batch_size

        if init_model:
            self.init_model()

        # cache the initial weights for retraining speed
        self.init_weights = None
        # generate a vector to act as a placeholder for time-points
        # prior to the start of the event so as to allow the network 
        # to implicitly learn a hidden state. We'll assume this
        # vector has length appox equal to 1
        self.filler_vector = np.random.randn(self.d) / np.sqrt(self.d)

    def init_model(self):
        self._compile_model()
        self.model_weights = self.model.get_weights()
        return self.model

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(LSTM(self.n_hidden, input_shape=(None, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(Dense(self.d, activation=None, kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)

    def do_reset_weights(self):
        # # self._compile_model()
        if self.init_weights is None:
            new_weights = [
                self.model.layers[0].kernel_initializer(w.shape)
                 for w in self.model.get_weights()
            ]
            self.model.set_weights(new_weights)
            self.model_weights = self.model.get_weights()
            self.init_weights = self.model.get_weights()
        else:
            self.model.set_weights(self.init_weights)

    def _unroll(self, x_example):
        """ put data into format expected by model
        concatenate current example with 
        the history of the last t-1 examples
        this is for the recurrent layer
        """
        x_train = np.concatenate([self.x_history[-1][-(self.t - 1):, :], x_example], axis=0)
        # x_train = np.concatenate([self.filler_vector, x_train], axis=0)
        x_train = x_train.reshape((1, np.min([x_train.shape[0], self.t]), self.d))
        return x_train

    def get_variance(self):
        # Sigma is stored as a vector corresponding to the entries of the diagonal covariance matrix
        return self.Sigma

    def do_reset_weights(self):
        new_weights = [
            self.model.layers[0].kernel_initializer(w.shape)
                for w in self.model.get_weights()
        ]
        self.model.set_weights(new_weights)
        self.model_weights = self.model.get_weights()

    def set_model(self, model):
        self.model = model
        self.do_reset_weights()

    def clear(self):
        delete_object_attributes(self)



class CSWEvent(TFobj):
    """ 

    """
    def __init__(self, d, var_df0=None, var_scale0=None, t=3, 
        n_hidden=None, optimizer=None,n_epochs=10, dropout=0.50, 
        l2_regularization=0.00, batch_size=32,
        kernel_initializer='glorot_uniform', init_model=False, 
        prior_log_prob=None, reset_weights=False,batch_update=True, 
        optimizer_kwargs=None, variance_prior_mode=None,
        variance_window=None):

        TFobj.__init__(self, d, var_df0=var_df0, 
            var_scale0=var_scale0, t=t,optimizer=optimizer, 
            n_epochs=n_epochs,l2_regularization=l2_regularization, 
            batch_size=batch_size,kernel_initializer=kernel_initializer, 
            init_model=False,prior_log_prob=prior_log_prob, 
            reset_weights=reset_weights,batch_update=batch_update, 
            optimizer_kwargs=optimizer_kwargs,variance_prior_mode=variance_prior_mode, 
            variance_window=variance_window)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _predict_f0(self):
        return self.predict_next_generative(self.filler_vector)

    def predict_f0(self):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        N.B. This answer is cached for speed

        """
        return self.f0

    def _predict_next(self, X):
        self.model.set_weights(self.model_weights)
        # Note: this function predicts the next conditioned on the training data the model has seen

        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert np.ndim(X) == 1
        assert X.shape[0] == self.d

        x_test = X.reshape((1, self.d))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        x_test = self._unroll(x_test)
        return self.model.predict(x_test)

    def predict_next(self, X):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        """
        if not self.f_is_trained:
            if np.ndim(X) > 1:
                return np.copy(X[-1, :]).reshape(1, -1)
            return np.copy(X).reshape(1, -1)

        return self._predict_next(X)

    def predict_next_generative(self, X):
        """ set weights and data reshape """
        self.model.set_weights(self.model_weights)
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.d))
        return self.model.predict(X0)

    # likelihood 
    
    def log_likelihood_f0(self, Xp):
        if not self.f0_is_trained:
            if self.prior_probability:
                return self.prior_probability
            else: 
                return norm(0, self.variance_prior_mode ** 0.5).logpdf(Xp).sum()

        # predict the initial point (# this has been precomputed for speed)
        Xp_hat = self.predict_f0()

        # calculate probability
        logprob = self.fast_mvnorm_diagonal_logprob(
                        Xp.reshape(-1) - Xp_hat.reshape(-1), 
                    self.Sigma)
        return logprob

    def log_likelihood_sequence(self, X, Xp):
        """ 
        Xp :current observation (target)
        X  :observation history (input)
        """
        print('log_like_seq')
        ## case: inactive schema
        if not self.f_is_trained:
            if self.prior_probability:
                return self.prior_probability
            else: 
                return norm(0, self.variance_prior_mode ** 0.5).logpdf(Xp).sum()
        ## case: reused schema
        Xp_hat = self.predict_next_generative(X)

        # calculate probability
        logprob = self.fast_mvnorm_diagonal_logprob(
                        Xp.reshape(-1) - Xp_hat.reshape(-1), 
                    self.Sigma)
        return logprob

    # WRAPPER

    def estimate(self):
        """ optional: run batch gradient 
        descent on all past event clusters """
        if self.reset_weights:
            self.do_reset_weights()
        else:
            self.model.set_weights(self.model_weights)

        # get predictions errors for variance estimate *before* updating the 
        # neural networks.  For an untrained model, the prediction should be the
        # origin, deterministically

        # Update Sigma
        x_train_0, xp_train_0 = self.training_pairs[-1]
        xp_hat = self.model.predict(x_train_0)
        self.prediction_errors = np.concatenate([self.prediction_errors, xp_train_0 - xp_hat], axis=0)
                
        # remove old observations from consideration of the variance
        t = np.max([0, np.shape(self.prediction_errors)[0] - self.variance_window])
        self.prediction_errors = self.prediction_errors[t:, :]

        # update the variance
        self._update_variance()


        ## then update the NN
        n_pairs = len(self.training_pairs)

        if self.batch_update:
            def draw_sample_pair():
                # draw a random cluster for the history
                idx = np.random.randint(n_pairs)
                return self.training_pairs[idx]
        else:
            # for online sampling, just use the last training sample
            def draw_sample_pair():
                return self.training_pairs[-1]

        # run batch gradient descent on all of the past events!
        # print('=')
        for _ in range(self.n_epochs):
            # print('==')

            # draw a set of training examples from the history
            x_batch = np.zeros((0, self.t, self.d))
            xp_batch = np.zeros((0, self.d))
            for _ in range(self.batch_size):

                x_sample, xp_sample = draw_sample_pair()
                # print('===',x_sample)

                x_batch = np.concatenate([x_batch, x_sample], axis=0)
                xp_batch = np.concatenate([xp_batch, xp_sample], axis=0)
            # print('train')
            self.model.train_on_batch(x_batch, xp_batch)
        self.model_weights = self.model.get_weights()

    # called in CSWSEM

    def update_f0(self, Xp, update_estimate=True):
        """ 
        called in CSWNET
        """
        self.update(self.filler_vector, Xp, update_estimate=update_estimate)
        self.f0_is_trained = True

        # precompute f0 for speed
        self.f0 = self._predict_f0()

    def update(self, X, Xp, update_estimate=True):
        """ 
        called in CSWNET
        """
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.d
        assert Xp.ndim == 1
        assert Xp.shape[0] == self.d

        x_example = X.reshape((1, self.d))
        xp_example = Xp.reshape((1, self.d))

        # concatenate the training example to the active event token
        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)

        # also, create a list of training pairs (x, y) for efficient sampling
        #  picks  random time-point in the history
        _n = np.shape(self.x_history[-1])[0]
        x_train_example = np.reshape(
                    unroll_data(self.x_history[-1][max(_n - self.t, 0):, :], self.t)[-1, :, :], (1, self.t, self.d)
                )
        self.training_pairs.append(tuple([x_train_example, xp_example]))

        if update_estimate:
            self.estimate()
            self.f_is_trained = True


    # def run_generative(self, n_steps, initial_point=None):
    #     self.model.set_weights(self.model_weights)
    #     if initial_point is None:
    #         x_gen = self._predict_f0()
    #     else:
    #         x_gen = np.reshape(initial_point, (1, self.d))
    #     for ii in range(1, n_steps):
    #         x_gen = np.concatenate([x_gen, self.predict_next_generative(x_gen[:ii, :])])
    #     return x_gen



import torch as tr

class CSWNet(tr.nn.Module):

  def __init__(self,stsize,seed):
    super().__init__()
    tr.manual_seed(seed)
    self.seed = seed
    self.stsize = stsize
    self.smdim = 12
    self.input_embed = tr.nn.Embedding(self.smdim,self.stsize)
    self.lstm = tr.nn.LSTMCell(self.stsize,self.stsize)
    self.init_lstm = tr.nn.Parameter(tr.rand(2,1,self.stsize),requires_grad=True)
    self.ff_hid2ulog = tr.nn.Linear(self.stsize,self.smdim,bias=False)
    return None

  def forward(self,state_int):
    ''' state_int contains ints [time,1] '''
    state_emb = self.input_embed(state_int)
    h_lstm,c_lstm = self.init_lstm
    outputs = -tr.ones(len(state_emb),self.stsize)
    for tstep in range(len(state_emb)):
      h_lstm,c_lstm = self.lstm(state_emb[tstep],(h_lstm,c_lstm))
      outputs[tstep] = h_lstm
    outputs = self.ff_hid2ulog(outputs)
    return outputs





""" 

"""