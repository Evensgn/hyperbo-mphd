from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import gp_added_v3
from hyperbo.gp_utils import utils
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
import jax
import jax.numpy as jnp
import numpy as np
import os
from tensorflow_probability.substrates.jax.distributions import Gamma, LogNormal
import optax
from experiment_defs import *


def fit_gp_params(key, dataset, cov_func, mean_func, mlp_features, objective, opt_method, fit_gp_maxiter,
                  fit_gp_batch_size, fit_gp_adam_learning_rate):
    n_dim = list(dataset.values())[0].x.shape[1]
    if 'mlp' in mean_func.__name__ or 'mlp' in cov_func.__name__:
        last_layer_dim = mlp_features[-1]
    else:
        last_layer_dim = n_dim

    # minimize nll
    init_params = defs.GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * last_layer_dim),
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method': opt_method,
            'maxiter': fit_gp_maxiter,
            'logging_interval': 100,
            'objective': objective,
            'batch_size': fit_gp_batch_size,
            'learning_rate': fit_gp_adam_learning_rate,
        })

    if 'mlp' in mean_func.__name__ or 'mlp' in cov_func.__name__:
        init_params.config['mlp_features'] = mlp_features
    if cov_func == kernel.dot_product_mlp:
        init_params.model['dot_prod_sigma'] = 0.1
        init_params.model['dot_prod_bias'] = 0.

    warp_func = utils.single_gp_default_warp_func

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    init_key, key = jax.random.split(key)

    model.initialize_params(init_key)

    new_key, key = jax.random.split(key)
    init_nll = model.neg_log_marginal_likelihood_batch(new_key, fit_gp_batch_size)
    inferred_params = model.train()
    inferred_nll = model.neg_log_marginal_likelihood_batch(new_key, fit_gp_batch_size)

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, params_utils.retrieve_params(inferred_params, param_keys, warp_func=warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))

    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def split_fit_gp_params_id(dir_path, key, setup, train_id, dataset_func_combined, dataset_func_split,
                           base_cov_func, base_mean_func, objective, opt_method, fit_gp_maxiter, fit_gp_batch_size,
                           fit_gp_adam_learning_rate, method_name):
    if setup == 'a':
        dataset = dataset_func_combined(train_id)
    elif setup == 'b':
        dataset = dataset_func_split(train_id)['train']  # only use training set
    else:
        raise ValueError('setup = {} not supported'.format(setup))

    if method_name == 'base':
        cov_func = base_cov_func
        mean_func = base_mean_func
        mlp_features = None
    elif method_name == 'hyperbo':
        cov_func = HYPERBO_KERNEL_TYPE
        mean_func = HYPERBO_MEAN_TYPE
        mlp_features = HYPERBO_MLP_FEATURES
    elif method_name == 'ablr':
        cov_func = ABLR_KERNEL_TYPE
        mean_func = ABLR_MEAN_TYPE
        mlp_features = ABLR_MLP_FEATURES
    elif method_name == 'fsbo':
        cov_func = FSBO_KERNEL_TYPE
        mean_func = FSBO_MEAN_TYPE
        mlp_features = FSBO_MLP_FEATURES
    else:
        raise ValueError('method_name = {} not supported'.format(method_name))

    new_key, key = jax.random.split(key)
    gp_params, nll_logs = fit_gp_params(new_key, dataset, cov_func, mean_func, mlp_features, objective, opt_method,
                                        fit_gp_maxiter, fit_gp_batch_size, fit_gp_adam_learning_rate)
    results = {'gp_params': gp_params, 'nll_logs': nll_logs}
    method_suffix = '' if method_name == 'base' else '_{}'.format(method_name)
    np.save(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}{}.npy'.format(setup, train_id, method_suffix)),
            results)


def split_fit_direct_hgp_two_step(dir_path, setup, train_id_list, distribution_type, leaveout_id=None, extra_suffix=''):
    constant_list = []
    lengthscale_list = []
    signal_variance_list = []
    noise_variance_list = []

    results = {}

    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    for train_id in train_id_list:
        gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)),
                            allow_pickle=True).item()['gp_params']
        constant_list.append(gp_params['constant'])
        lengthscale_list += list(gp_params['lengthscale'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])

    gp_distribution_params = {}
    gp_distribution_params['constant'] = utils.normal_param_from_samples(np.array(constant_list))
    if distribution_type == 'gamma':
        gp_distribution_params['lengthscale'] = utils.gamma_param_from_samples(np.array(lengthscale_list))
        gp_distribution_params['signal_variance'] = utils.gamma_param_from_samples(np.array(signal_variance_list))
        gp_distribution_params['noise_variance'] = utils.gamma_param_from_samples(np.array(noise_variance_list))
    elif distribution_type == 'lognormal':
        gp_distribution_params['lengthscale'] = utils.lognormal_param_from_samples(np.array(lengthscale_list))
        gp_distribution_params['signal_variance'] = utils.lognormal_param_from_samples(np.array(signal_variance_list))
        gp_distribution_params['noise_variance'] = utils.lognormal_param_from_samples(np.array(noise_variance_list))
    else:
        raise ValueError('distribution_type = {} not supported'.format(distribution_type))

    results['gp_distribution_params'] = gp_distribution_params
    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_{}.npy'.format(setup))
    save_file_name = save_file_name.replace('.npy', '{}.npy'.format(extra_suffix))
    np.save(save_file_name, results)


def split_fit_hpl_hgp_two_step(dir_path, key, setup, train_id_list, fit_two_step_maxiter, fit_two_step_learning_rate,
                               distribution_type, dataset_dim_feature_values_path, leaveout_id=None, extra_suffix=''):
    dim_feature_values = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()
    model_params = {}

    constant_list = []
    signal_variance_list = []
    noise_variance_list = []

    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    model_params['search_space_params'] = {}
    fit_gp_results = {}
    for train_id in train_id_list:
        gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_{}_id_{}.npy'.format(setup, train_id)),
                            allow_pickle=True).item()['gp_params']
        fit_gp_results[train_id] = gp_params
        model_params['search_space_params'][train_id] = gp_params
        constant_list.append(gp_params['constant'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])

    # fit mlp for lengthscale
    lengthscale_dist_mlp_features = HPL_LENGTHSCALE_MLP_FEATURES
    new_key, key = jax.random.split(key)
    init_val = jnp.ones((0, 4), jnp.float32)
    lengthscale_dist_mlp_params = bf.MLP(lengthscale_dist_mlp_features).init(new_key, init_val)['params']

    # optimization with adam
    optimizer = optax.adam(fit_two_step_learning_rate)
    opt_state = optimizer.init(lengthscale_dist_mlp_params)

    @jax.jit
    def loss_func(lengthscale_dist_mlp_params):
        lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
        loss = 0.
        for train_id in train_id_list:
            gp_params = fit_gp_results[train_id]
            dim_feature_value = dim_feature_values[train_id]
            lengthscale = gp_params['lengthscale']
            lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params},
                                                              dim_feature_value)

            # making sure the correct dim_feature_value is used
            assert len(lengthscale) == dim_feature_value.shape[0]

            for dim in range(len(lengthscale)):
                if distribution_type == 'gamma':
                    lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
                    print(lengthscale_dist_params_dim)
                    lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                    gamma_dist = Gamma(lengthscale_a, rate=lengthscale_b)
                    loss += -gamma_dist.log_prob(lengthscale[dim])
                elif distribution_type == 'lognormal':
                    lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
                    lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], \
                                                        lengthscale_dist_params_dim[1]
                    lognormal_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
                    loss += -lognormal_dist.log_prob(lengthscale[dim])
                else:
                    raise ValueError('distribution_type = {} not supported'.format(distribution_type))
        return loss

    for iter in range(fit_two_step_maxiter):
        current_loss, grad = jax.value_and_grad(loss_func)(lengthscale_dist_mlp_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        lengthscale_dist_mlp_params = optax.apply_updates(lengthscale_dist_mlp_params, updates)
        if iter % 100 == 0:
            print('iter:', iter, ', loss:', current_loss)

    if distribution_type == 'gamma':
        print('lengthscale_gamma_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_gamma_mlp_params'] = lengthscale_dist_mlp_params
    elif distribution_type == 'lognormal':
        print('lengthscale_lognormal_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_lognormal_mlp_params'] = lengthscale_dist_mlp_params
    else:
        raise ValueError('distribution_type = {} not supported'.format(distribution_type))

    # fit other parameters
    constant_list = jnp.array(constant_list)
    constant_mu, constant_sigma = utils.normal_param_from_samples(constant_list)
    model_params['constant_normal_params'] = (constant_mu, constant_sigma)
    print('constant: Normal(mu={}, sigma={})'.format(constant_mu, constant_sigma))

    signal_variance_list = jnp.array(signal_variance_list)
    noise_variance_list = jnp.array(noise_variance_list)
    if distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = utils.gamma_param_from_samples(signal_variance_list)
        model_params['signal_variance_gamma_params'] = (signal_variance_a, signal_variance_b)
        print('signal_variance: Gamma(alpha={}, beta={})'.format(signal_variance_a, signal_variance_b))

        noise_variance_a, noise_variance_b = utils.gamma_param_from_samples(noise_variance_list)
        model_params['noise_variance_gamma_params'] = (noise_variance_a, noise_variance_b)
        print('noise_variance: Gamma(alpha={}, beta={})'.format(noise_variance_a, noise_variance_b))
    elif distribution_type == 'lognormal':
        signal_variance_mu, signal_variance_sigma = utils.lognormal_param_from_samples(signal_variance_list)
        model_params['signal_variance_lognormal_params'] = (signal_variance_mu, signal_variance_sigma)
        print('signal_variance: LogNormal(mu={}, sigma={})'.format(signal_variance_mu, signal_variance_sigma))

        noise_variance_mu, noise_variance_sigma = utils.lognormal_param_from_samples(noise_variance_list)
        model_params['noise_variance_lognormal_params'] = (noise_variance_mu, noise_variance_sigma)
        print('noise_variance: LogNormal(mu={}, sigma={})'.format(noise_variance_mu, noise_variance_sigma))
    else:
        raise ValueError('distribution_type = {} not supported'.format(distribution_type))

    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_{}.npy'.format(setup))
    save_file_name = save_file_name.replace('.npy', '{}.npy'.format(extra_suffix))
    np.save(save_file_name, model_params)


def fit_hpl_hgp_end_to_end(key, super_dataset, dim_feature_values, cov_func, mean_func, objective, opt_method,
                           fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                           init_params_value):
    # minimize nll
    init_params = defs.GPParams(
        model = {},
        config = {
            'method': opt_method,
            'maxiter': fit_hgp_maxiter,
            'logging_interval': 10,
            'objective': objective,
            'batch_size': fit_hgp_batch_size,
            'learning_rate': fit_hgp_adam_learning_rate,
            'distribution_type': distribution_type,
        })

    single_gp_warp_func = utils.single_gp_default_warp_func
    single_gp_inverse_warp_func = utils.single_gp_default_inverse_warp_func
    hgp_warp_func, hgp_inverse_warp_func = utils.get_e2e_v3_warp_func(
        distribution_type=distribution_type,
        single_gp_warp_func=single_gp_warp_func,
        single_gp_inverse_warp_func=single_gp_inverse_warp_func,
    )

    init_params.config['lengthscale_{}_mlp_features'.format(distribution_type)] = HPL_LENGTHSCALE_MLP_FEATURES
    if init_params_value is None:
        # initialize constant, signal variance, and noise variance distribution parameters
        new_key, key = jax.random.split(key)
        init_params.model['constant_normal_params'] = jax.random.normal(new_key, (2,))
        new_key, key = jax.random.split(key)
        init_params.model['signal_variance_{}_params'.format(distribution_type)] = jax.random.normal(new_key, (2,))
        new_key, key = jax.random.split(key)
        init_params.model['noise_variance_{}_params'.format(distribution_type)] = jax.random.normal(new_key, (2,))

        # initialize per-search-space gp parameters
        init_params.model['search_space_params'] = {}
        for dataset_id, dataset in super_dataset.items():
            n_dim = list(dataset.values())[0].x.shape[1]
            init_params.model['search_space_params'][dataset_id] = {
                'constant': 1.0,
                'lengthscale': jnp.array([1.0] * n_dim),
                'signal_variance': 0.,
                'noise_variance': -4.,
            }

        # initialize the lengthscale distribution mlp
        new_key, key = jax.random.split(key)
        init_val = jnp.ones((0, 4), jnp.float32)
        init_params.model['lengthscale_{}_mlp_params'.format(distribution_type)] = \
            bf.MLP(init_params.config['lengthscale_{}_mlp_features'.format(distribution_type)]).init(
            new_key, init_val)['params']
    else:
        init_params_value = utils.apply_warp_func(init_params_value, hgp_inverse_warp_func)
        init_params.model = init_params_value

    model = gp_added_v3.HGP_E2E_v3(
        super_dataset=super_dataset,
        dim_feature_values=dim_feature_values,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        hgp_warp_func=hgp_warp_func,
        single_gp_warp_func=single_gp_warp_func,
    )

    init_key, key = jax.random.split(key)
    model.initialize_params(init_key)

    new_key, key = jax.random.split(key)
    init_nll = model.neg_log_marginal_likelihood_batch(key, fit_hgp_batch_size)

    new_key, key = jax.random.split(key)
    inferred_params = model.train(key=new_key)

    new_key, key = jax.random.split(key)
    inferred_nll = model.neg_log_marginal_likelihood_batch(key, fit_hgp_batch_size)

    param_keys = init_params.model.keys()
    retrieved_inferred_params = dict(
        zip(param_keys, params_utils.retrieve_params(inferred_params, param_keys, warp_func=hgp_warp_func)))
    print('retrieved_inferred_params = {}'.format(retrieved_inferred_params))

    print('init_nll = {}, inferred_nll = {}'.format(init_nll, inferred_nll))
    nll_logs = (init_nll, inferred_nll)

    return retrieved_inferred_params, nll_logs


def split_fit_hpl_hgp_end_to_end(dir_path, key, setup, train_id_list, dataset_func_combined, dataset_func_split,
                                 dataset_dim_feature_values_path, cov_func, mean_func, objective, opt_method,
                                 fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                 use_init_params_value=True, leaveout_id=None, extra_suffix=''):
    if leaveout_id is not None:
        assert (setup == 'b')
        train_id_list = [train_id for train_id in train_id_list if train_id != leaveout_id]

    print('split_fit_hpl_hgp_end_to_end: train_id_list = {}'.format(train_id_list))

    super_dataset = {}
    for train_id in train_id_list:
        print('read train_id = {}'.format(train_id))
        if setup == 'a':
            dataset = dataset_func_combined(train_id)
        elif setup == 'b':
            dataset = dataset_func_split(train_id)['train']  # only use training set
        else:
            raise ValueError('setup = {} not supported'.format(setup))
        super_dataset[train_id] = dataset

    dim_feature_values = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()
    if use_init_params_value:
        if leaveout_id is not None:
            init_params_value_path = os.path.join(
                dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(leaveout_id))
        else:
            init_params_value_path = os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_{}.npy'.format(setup))
        init_params_value = np.load(init_params_value_path, allow_pickle=True).item()
    else:
        init_params_value = None

    print('read dim_feature_values done')

    new_key, key = jax.random.split(key)
    print('start fit_hpl_hgp_end_to_end')
    gp_params, nll_logs = fit_hpl_hgp_end_to_end(new_key, super_dataset, dim_feature_values, cov_func, mean_func,
                                                 objective, opt_method, fit_hgp_maxiter, fit_hgp_batch_size,
                                                 fit_hgp_adam_learning_rate, distribution_type, init_params_value)
    results = {'gp_params': gp_params, 'nll_logs': nll_logs}
    if leaveout_id is not None:
        save_file_name = os.path.join(dir_path,
                                      'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(leaveout_id))
    else:
        save_file_name = os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_{}.npy'.format(setup))
    if not use_init_params_value:
        save_file_name = save_file_name.replace('.npy', '_from_scratch.npy')
    save_file_name = save_file_name.replace('.npy', '{}.npy'.format(extra_suffix))
    np.save(save_file_name, results)


def eval_training_loss(dir_path, key, super_dataset, dim_feature_values, cov_func, mean_func,
                       objective, eval_loss_batch_size, eval_loss_n_batches, distribution_type, method_name):
    if method_name == 'hpl_hgp_end_to_end':
        hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
                                 allow_pickle=True).item()['gp_params']
    elif method_name == 'hpl_hgp_end_to_end_from_scratch':
        hpl_hgp_params = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
            allow_pickle=True).item()['gp_params']
    elif method_name == 'hpl_hgp_two_step':
        hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'),
                                 allow_pickle=True).item()
    else:
        raise ValueError('Unknown method name: {}'.format(method_name))

    # minimize nll
    model_params = defs.GPParams(
        model = {},
        config = {
            'objective': objective,
            'distribution_type': distribution_type,
        })

    model_params.config['lengthscale_{}_mlp_features'.format(distribution_type)] = HPL_LENGTHSCALE_MLP_FEATURES
    model_params.model = hpl_hgp_params

    model = gp_added_v3.HGP_E2E_v3(
        super_dataset=super_dataset,
        dim_feature_values=dim_feature_values,
        mean_func=mean_func,
        cov_func=cov_func,
        params=model_params,
        hgp_warp_func=None,
        single_gp_warp_func=None,
    )

    init_key, key = jax.random.split(key)
    model.initialize_params(init_key)

    total_nll_batches = []
    nll_data_batches = []
    nll_distribution_params_batches = []
    for i in range(eval_loss_n_batches):
        new_key, key = jax.random.split(key)
        total_nll_i, nll_data_i, nll_distribution_params_i = model.neg_log_marginal_likelihood_batch(
            new_key, eval_loss_batch_size)
        total_nll_batches.append(total_nll_i)
        nll_data_batches.append(nll_data_i)
        nll_distribution_params_batches.append(nll_distribution_params_i)

    results = {
        'total_nll_batches': total_nll_batches,
        'total_nll_mean': np.mean(total_nll_batches),
        'total_nll_std': np.std(total_nll_batches),

        'nll_data_batches': nll_data_batches,
        'nll_data_mean': np.mean(nll_data_batches),
        'nll_data_std': np.std(nll_data_batches),

        'nll_distribution_params_batches': nll_distribution_params_batches,
        'nll_distribution_params_mean': np.mean(nll_distribution_params_batches),
        'nll_distribution_params_std': np.std(nll_distribution_params_batches),
    }
    return results


def split_eval_hpl_hgp_loss_setup_b(dir_path, key, train_or_test, setup_b_id_list, dataset_func_split,
                                    dataset_dim_feature_values_path, cov_func, mean_func, objective,
                                    eval_loss_batch_size, eval_loss_n_batches, distribution_type,
                                    method_name, extra_suffix=''):
    super_dataset = {}
    for dataset_id in setup_b_id_list:
        if train_or_test == 'train':
            dataset = dataset_func_split(dataset_id)['train']  # only use train set
        elif train_or_test == 'test':
            dataset = dataset_func_split(dataset_id)['test']  # only use test set
        else:
            raise ValueError('Unknown train_or_test: {}'.format(train_or_test))
        super_dataset[dataset_id] = dataset

    dim_feature_values = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()

    new_key, key = jax.random.split(key, 2)

    results = eval_training_loss(dir_path, new_key, super_dataset, dim_feature_values, cov_func, mean_func,
                                 objective, eval_loss_batch_size, eval_loss_n_batches, distribution_type, method_name)

    np.save(os.path.join(dir_path, 'split_eval_loss_setup_b_{}_{}{}.npy'.format(train_or_test, method_name, extra_suffix)),
            results)
