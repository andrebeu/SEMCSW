import numpy as np
from schema_prediction_task_6_10_20 import generate_exp, batch_exp
from sem.event_models import GRUEvent, LSTMEvent

output_file_path = './json_files_v061020/'

## sem parameters
dropout           = 0.0
l2_regularization = 0.0
batch_size        = 25.0
# n_epochs          = 28
# batch_size        = 50
# lr                = 0.005
# epsilon           = 1e-5
# log_alpha         = 10.0
# log_lambda        = 0.0
# n_hidden          = 40

def main(n_epochs=28,batch_size=25,lr=0.001,log_alpha=0.0,
    log_lambda=0.0, n_hidden=10, epsilon=1e-5, no_split=False,
     batch_n=0, LSTM=False, batch_update=True,actor_weight=1.0):
    """ 
    :param n_epochs:    int     (28)
    :param batch_size:  int     (25)
    :param lr:          float   (0.001)
    :param log_alpha:   float   (0.0)
    :param log_lamda:   float   (0.0)
    :param n_hidden:    int     (10)
    :param epsilon      float   (1e-5)
    :param no_split     bool    (False) Prevent the model from segmenting?
    :param batch_update bool    (True)  update the RNNs with sampled batches?

    this function runs a single itteration of the model, (both blocked and interleaved)
    given a set of parameters
    """

    # use an LSTM or a GRU as the event model?
    if LSTM:
        f_class = LSTMEvent
        model = 'LSTM'
    else:
        f_class = GRUEvent
        model = 'GRU'

    # set the kwargs for the story
    story_kwargs = dict(seed=None, actor_weight=actor_weight)

    # set the sem_kwargs
    optimizer_kwargs = dict(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=False)
    f_opts=dict(
        dropout=dropout,
        batch_size=batch_size,
        batch_update=batch_update,
        l2_regularization=l2_regularization, 
        n_epochs=n_epochs,
        optimizer_kwargs=optimizer_kwargs,
        )
    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts, f_class=f_class)

    # mark whether we run th
    tag = ''
    if no_split:
        tag += '_nosplit'
    if batch_update == False:
        tag += '_online' 

    # for now, we only consider the basic blocked vs interleaved set up
    json_tag = '_{}_nhidden{}_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}'.format(
        model, n_hidden, epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n, tag)

    # run a single batch
    _ = batch_exp(
        sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
        progress_bar=False, block_only=False,
        run_mixed=False, 
        save_to_json=True,
        json_tag=json_tag,
        json_file_path=output_file_path, 
        no_split=no_split)


def parse_args(args):

    # these are the defaults for the function
    main_kwargs = dict(n_epochs=28,batch_size=25,lr=0.001,log_alpha=0.0,
    log_lambda=0.0, n_hidden=10, epsilon=1e-5, no_split=False,
    batch_n=0, LSTM=False, batch_update=True, actor_weight=1.0)

    for arg in args:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        
        # set the option as a kwarg
        if k in main_kwargs:
            print("Setting parameter: {} = {}".format(k, v))
            main_kwargs[k] = v
        else:
            print("Parameter: {} not found!".format(k))
            
    main_kwargs['n_epochs']     = int(float(main_kwargs['n_epochs']))
    main_kwargs['batch_size']   = int(float(main_kwargs['batch_size']))
    main_kwargs['lr']           = float(main_kwargs['lr'])
    main_kwargs['log_alpha']    = float(main_kwargs['log_alpha'])
    main_kwargs['log_lambda']   = float(main_kwargs['log_lambda'])
    main_kwargs['n_hidden']     = int(float(main_kwargs['n_hidden']))
    main_kwargs['epsilon']      = float(main_kwargs['epsilon'])
    main_kwargs['no_split']     = main_kwargs['no_split'] == 'True'
    main_kwargs['batch_n']        = int(float(main_kwargs['batch_n']))
    main_kwargs['LSTM']         = main_kwargs['LSTM'] == 'True'
    main_kwargs['batch_update'] = main_kwargs['batch_update'] == 'True'
    main_kwargs['actor_weight']      = float(main_kwargs['actor_weight'])

    return main_kwargs



if __name__ == "__main__":
    import sys

    kwargs = parse_args(sys.argv[1:])
    print(kwargs)
    main(**kwargs) 