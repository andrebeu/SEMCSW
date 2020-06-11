

def make_random_queue(batch_n, lrs, n_epochs_, log_alphas, log_lambdas, tag='', mixed=False, epsilon=1e-5):
    # go through these lists in a random order for better sampling
    parameter_tuples = []
    for lr in lrs:
        for n_epochs in n_epochs_:
            for log_alpha in log_alphas:
                for log_lambda in log_lambdas:
                    parameter_tuples.append((lr, n_epochs, log_alpha, log_lambda))

    n = len(parameter_tuples)  # total number of simulations to be run

    # loop through the parameters in a random order, look for corresponding files,
    # and if none are found, add to the cue.
    parameters_queue = []
    for lr, n_epochs, log_alpha, log_lambda in np.random.permutation(parameter_tuples):
        
        # # list of files that specify the simulation experiment exactly
        # args = [epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, int(batch_n)]
        # files = glob('./json_files/*{}*{}*{}*{}*{}*{}*{}*'.format(*args))

        args = [epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, int(batch_n)]

        json_tag = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}'.format(
            epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, batch_n, tag)
        json_tag_mixed = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}_mixed'.format(
            epsilon, lr, int(n_epochs), dropout, log_alpha, log_lambda, batch_n, tag)

        file_string = './json_files/results{}.json'.format(json_tag)
        
        if mixed:
            file_string = './json_files/results{}.json'.format(json_tag_mixed)

        files = glob(file_string)

        if not files:
            parameters_queue.append((lr, int(n_epochs), log_alpha, log_lambda))

    t = len(parameters_queue)
    print('Found {} of {} simulations previous completed'.format(n-t, n))
    return parameters_queue


def run_single_batch(batch_n, lr, n_epochs, log_alpha, log_lambda, mixed=False, tag='', 
                    no_split=False, batch_update=True):

    # sem prior params (dosen't rely on mutability)
    sem_kwargs = dict(lmda=np.exp(log_lambda), alfa=np.exp(log_alpha), f_opts=f_opts)

    f_opts['n_epochs'] = int(n_epochs)
    optimizer_kwargs['lr'] = lr # this relies on mutability to change the f_opts dictionary

    # run batch update or online training?
    f_opts['batch_update'] = batch_update

    json_tag = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n, tag)
    json_tag_mixed = '_e{}_lr{}_n{}_d{}_logalfa_{}_loglmda_{}_batch_{}{}_mixed'.format(
        epsilon, lr, n_epochs, dropout, log_alpha, log_lambda, batch_n, tag)

    if not mixed:
        print(json_tag)
        _, _, _ = batch_exp(
            sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
            progress_bar=False, block_only=False,
            run_mixed=False,
            save_to_json=True,
            json_tag=json_tag,
            json_file_path='./json_files/', 
            no_split=no_split)
    else:
        print(json_tag_mixed)
        _, _, _ = batch_exp(
            sem_kwargs, story_kwargs, n_batch=1, sem_progress_bar=True,
            progress_bar=False, block_only=False,
            run_mixed=True,
            save_to_json=True,
            json_tag=json_tag_mixed,
            json_file_path='./json_files/', 
            no_split=no_split)
