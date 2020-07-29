from sem.event_models import LinearEvent, map_variance, RecurrentLinearEvent

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, Lambda, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import l2_normalize
from sem.utils import fast_mvnorm_diagonal_logprob, unroll_data, get_prior_scale, delete_object_attributes
from scipy.stats import norm
import numpy as np


class AndreRNN(RecurrentLinearEvent):

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
        self.model.add(LSTM(self.n_hidden, input_shape=(None, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(Dense(self.d, activation=None, kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)

