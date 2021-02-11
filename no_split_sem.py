import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from tqdm import tqdm
# from sem.event_models import GRUEvent
from sem.utils import delete_object_attributes
from multiprocessing import Queue, Process

# there are a ~ton~ of tf warnings from Keras, suppress them here
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Results(object):
    """ placeholder object to store results """
    pass

