from sem.event_models import LinearEvent, map_variance

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, Lambda, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import l2_normalize
from sem.utils import fast_mvnorm_diagonal_logprob, unroll_data, get_prior_scale, delete_object_attributes
from scipy.stats import norm

class RecurrentLinearEvent(LinearEvent):

    # RNN which is initialized once and then trained using stochastic gradient descent
    # i.e. each new scene is a single example batch of size 1

    def __init__(self, d, var_df0=None, var_scale0=None, t=3,
                 optimizer=None, n_epochs=10, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=None, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None,variance_prior_mode=None,variance_window=None):

        LinearEvent.__init__(self, d, var_df0=var_df0, var_scale0=var_scale0,
                             optimizer=optimizer, n_epochs=n_epochs,
                             init_model=False, kernel_initializer=kernel_initializer,
                             l2_regularization=l2_regularization, prior_log_prob=prior_log_prob,
                             reset_weights=reset_weights, batch_update=batch_update, 
                             optimizer_kwargs=optimizer_kwargs, variance_prior_mode=variance_prior_mode,
                             variance_window=variance_window)

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
        self.filler_vector = np.random.randn(1, self.d) / np.sqrt(self.d)

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

    # initialize model once so we can then update it online
    def _compile_model(self):
        self.model = Sequential()
        self.model.add(SimpleRNN(self.d, input_shape=(self.t, self.d),
                                 activation=None, kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)

    # concatenate current example with the history of the last t-1 examples
    # this is for the recurrent layer
    #
    def _unroll(self, x_example):
        x_train = np.concatenate([self.x_history[-1][-(self.t - 1):, :], x_example], axis=0)
        
#         x_train = np.concatenate([np.zeros((self.t - x_train.shape[0], self.d)), x_train], axis=0)
        n_pad = self.t - x_train.shape[0]
        x_train = np.concatenate([np.tile(self.filler_vector, (n_pad, 1)), x_train], axis=0)
        x_train = x_train.reshape((1, self.t, self.d))
        return x_train

    # predict a single example
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

    def _predict_f0(self):
        return self.predict_next_generative(np.zeros(self.d))

    def _update_variance(self):
        if np.shape(self.prediction_errors)[0] > 1:
            self.Sigma = map_variance(self.prediction_errors, self.var_df0, self.var_scale0)

    def update(self, X, Xp, update_estimate=True):
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

    def predict_next_generative(self, X):
        self.model.set_weights(self.model_weights)
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.d))
        return self.model.predict(X0)

    # optional: run batch gradient descent on all past event clusters
    def estimate(self):
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
        for _ in range(self.n_epochs):

            # draw a set of training examples from the history
            x_batch = np.zeros((0, self.t, self.d))
            xp_batch = np.zeros((0, self.d))
            for _ in range(self.batch_size):

                x_sample, xp_sample = draw_sample_pair()

                x_batch = np.concatenate([x_batch, x_sample], axis=0)
                xp_batch = np.concatenate([xp_batch, xp_sample], axis=0)

            self.model.train_on_batch(x_batch, xp_batch)
        self.model_weights = self.model.get_weights()
        
        
class GRUEvent(RecurrentLinearEvent):

    def __init__(self, d, var_df0=None, var_scale0=None, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=None, reset_weights=False,
                 batch_update=True, optimizer_kwargs=None,variance_prior_mode=None,variance_window=None):

        RecurrentLinearEvent.__init__(self, d, var_df0=var_df0, var_scale0=var_scale0, t=t,
                                      optimizer=optimizer, n_epochs=n_epochs,
                                      l2_regularization=l2_regularization, batch_size=batch_size,
                                      kernel_initializer=kernel_initializer, init_model=False,
                                      prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                      batch_update=batch_update, optimizer_kwargs=optimizer_kwargs,
                                      variance_prior_mode=variance_prior_mode, variance_window=variance_window)

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
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class LSTMEvent(RecurrentLinearEvent):

    def __init__(self, d, var_df0=None, var_scale0=None, t=3, n_hidden=None, optimizer=None,
                 n_epochs=10, dropout=0.50, l2_regularization=0.00,
                 batch_size=32, kernel_initializer='glorot_uniform', init_model=False, prior_log_prob=None,
                 reset_weights=False, batch_update=True, optimizer_kwargs=None, variance_prior_mode=None,
                 variance_window=None):

        RecurrentLinearEvent.__init__(self, d, var_df0=var_df0, var_scale0=var_scale0, t=t,
                                      optimizer=optimizer, n_epochs=n_epochs,
                                      l2_regularization=l2_regularization, batch_size=batch_size,
                                      kernel_initializer=kernel_initializer, init_model=False,
                                      prior_log_prob=prior_log_prob, reset_weights=reset_weights,
                                      batch_update=batch_update, optimizer_kwargs=optimizer_kwargs,
                                      variance_prior_mode=variance_prior_mode, variance_window=variance_window)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.dropout = dropout

        if init_model:
            self.init_model()

    def _compile_model(self):
        self.model = Sequential()
        # input_shape[0] = time-steps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(LSTM(self.n_hidden, input_shape=(self.t, self.d),
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(rate=self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)
