import numpy as np
from schema_prediction_task_7_22_20 import generate_exp, batch_exp
from sem.event_models import GRUEvent, LSTMEvent


output_file_path = './json_files_mergecopy/'

n_batch = 250


def main():
    # from sem.event_models import LSTMEvent
    from custom_event_models import GRUEvent_normed

    seed = None
    err = 0.2; # error probability for plate's formula

    # story_generator = generate_exp_balanced
    story_kwargs = dict(seed=seed, err=err, actor_weight=1.0)

    ## sem parameters
    dropout           = 0.0
    l2_regularization = 0.0
    n_epochs          = 32
    batch_size        = 25
    lr                = 0.009
    epsilon           = 1e-5
    log_alpha         = 4.0
    log_lambda        = 16.0
    n_hidden          = None
    var_df0           = None
    var_scale0        = None

    batch_update = True

    optimizer_kwargs = dict(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=epsilon, amsgrad=False
    )

    f_class = GRUEvent_normed

    f_opts=dict(
        dropout=dropout,
        batch_size=batch_size,batch_update=batch_update,
        l2_regularization=l2_regularization,
        n_epochs=n_epochs,\
        optimizer_kwargs=optimizer_kwargs,
        var_scale0=var_scale0, var_df0=var_df0)

    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts, f_class=f_class)

    # for now, we only consider the basic blocked vs interleaved set up
    json_tag = '_AndreTask_v072220_mergcopy'

    # run a single batch
    _ = batch_exp(
        sem_kwargs, story_kwargs, n_batch=n_batch, sem_progress_bar=True,
        progress_bar=False, block_only=False,
        run_mixed=False, 
        save_to_json=True,
        json_tag=json_tag,
        json_file_path=output_file_path, 
        no_split=False)




if __name__ == "__main__":
    main() 