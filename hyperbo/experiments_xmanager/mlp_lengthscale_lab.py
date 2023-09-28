import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.basics import data_utils
from hyperbo.bo_utils import bayesopt
from hyperbo.bo_utils import data
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
import hyperbo.bo_utils.acfun as acfun
import jax
import jax.numpy as jnp
import numpy as np
from pathos.multiprocessing import ProcessingPool
import os
import plot
import datetime
import argparse
import math
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma, LogNormal
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
import gc
import subprocess
from experiment_defs import HPOB_DATA_PATH, RESULTS_DIR, GROUP_ID, RANDOM_SEED, TRAIN_ID_LIST, TEST_ID_LIST, FULL_ID_LIST, HPOB_DATA_ANALYSIS_PATH
import optax


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


if __name__ == '__main__':
    output_subdir = 'mlp_lengthscale_lab'
    gp_params_distribution_type = 'gamma'
    assert gp_params_distribution_type in ['gamma', 'lognormal']
    random_seed = 0
    learning_rate = 1e-2
    max_iter = 1000

    group_id = GROUP_ID
    results_dir = RESULTS_DIR

    train_id_list = TRAIN_ID_LIST
    test_id_list = TEST_ID_LIST
    setup_b_id_list = FULL_ID_LIST

    hpob_negative_y = False
    normalize_x = True
    normalize_y = True
    hpob_data_path = HPOB_DATA_PATH
    dataset_func_combined = partial(data.hpob_dataset_v2, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                    normalize_x=normalize_x, normalize_y=normalize_y)
    dataset_func_split = partial(data.hpob_dataset_v3, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                 normalize_x=normalize_x, normalize_y=normalize_y)

    dim_feature_values = np.load(HPOB_DATA_ANALYSIS_PATH, allow_pickle=True).item()

    results = np.load(os.path.join(results_dir, 'hpob_normalize_xy_only_fitting_matern52', 'results.npy'), allow_pickle=True).item()
    fit_gp_results = results['setup_b']['fit_gp_params']

    model_params = {}
    key = jax.random.PRNGKey(random_seed)

    constant_list = []
    signal_variance_list = []
    noise_variance_list = []
    continuous_lengthscale_list = []
    discrete_lengthscale_list = []

    model_params['search_space_params'] = {}
    for id in setup_b_id_list:
        gp_params = fit_gp_results[id]['gp_params']
        model_params['search_space_params'][id] = gp_params
        constant_list.append(gp_params['constant'])
        signal_variance_list.append(gp_params['signal_variance'])
        noise_variance_list.append(gp_params['noise_variance'])
        dim_feature_value = dim_feature_values[id]
        lengthscale = gp_params['lengthscale']
        for dim in range(len(lengthscale)):
            if dim_feature_value[dim][0] == 1:
                discrete_lengthscale_list.append(lengthscale[dim])
            else:
                continuous_lengthscale_list.append(lengthscale[dim])

    # fit mlp for lengthscale
    lengthscale_dist_mlp_features = (2,)
    new_key, key = jax.random.split(key)
    init_val = jnp.ones((0, 4), jnp.float32)
    lengthscale_dist_mlp_params = bf.MLP(lengthscale_dist_mlp_features).init(new_key, init_val)['params']

    # optimization with adam
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(lengthscale_dist_mlp_params)

    for iter in range(max_iter):
        def loss_func(lengthscale_dist_mlp_params):
            lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
            loss = 0.
            for id in setup_b_id_list:
                gp_params = fit_gp_results[id]['gp_params']
                dim_feature_value = dim_feature_values[id]
                lengthscale = gp_params['lengthscale']
                lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params}, dim_feature_value)
                for dim in range(len(lengthscale)):
                    if gp_params_distribution_type == 'gamma':
                        lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
                        lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                        gamma_dist = Gamma(lengthscale_a, rate=lengthscale_b)
                        loss += -gamma_dist.log_prob(lengthscale[dim])
                    elif gp_params_distribution_type == 'lognormal':
                        lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
                        lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
                        lognormal_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
                        loss += -lognormal_dist.log_prob(lengthscale[dim])
            return loss

        current_loss, grad = jax.value_and_grad(loss_func)(lengthscale_dist_mlp_params)
        updates, opt_state = optimizer.update(grad, opt_state)
        lengthscale_dist_mlp_params = optax.apply_updates(lengthscale_dist_mlp_params, updates)
        print('iter:', iter, ', loss:', current_loss)

    if gp_params_distribution_type == 'gamma':
        print('lengthscale_gamma_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_gamma_mlp_params'] = lengthscale_dist_mlp_params
    elif gp_params_distribution_type == 'lognormal':
        print('lengthscale_lognormal_mlp_params', lengthscale_dist_mlp_params)
        model_params['lengthscale_lognormal_mlp_params'] = lengthscale_dist_mlp_params

    # fit other parameters
    constant_list = jnp.array(constant_list)
    constant_mu = jnp.mean(constant_list)
    constant_sigma = jnp.std(constant_list)
    model_params['constant_normal_params'] = (constant_mu, constant_sigma)
    print('constant: Normal(mu={}, sigma={})'.format(constant_mu, constant_sigma))

    signal_variance_list = jnp.array(signal_variance_list)
    if gp_params_distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = utils.gamma_param_from_samples(signal_variance_list)
        model_params['signal_variance_gamma_params'] = (signal_variance_a, signal_variance_b)
        print('signal_variance: Gamma(alpha={}, beta={})'.format(signal_variance_a, signal_variance_b))
    elif gp_params_distribution_type == 'lognormal':
        signal_variance_mu, signal_variance_sigma = utils.lognormal_param_from_samples(signal_variance_list)
        model_params['signal_variance_lognormal_params'] = (signal_variance_mu, signal_variance_sigma)
        print('signal_variance: LogNormal(mu={}, sigma={})'.format(signal_variance_mu, signal_variance_sigma))

    noise_variance_list = jnp.array(noise_variance_list)
    if gp_params_distribution_type == 'gamma':
        noise_variance_a, noise_variance_b = utils.gamma_param_from_samples(noise_variance_list)
        model_params['noise_variance_gamma_params'] = (noise_variance_a, noise_variance_b)
        print('noise_variance: Gamma(alpha={}, beta={})'.format(noise_variance_a, noise_variance_b))
    elif gp_params_distribution_type == 'lognormal':
        noise_variance_mu, noise_variance_sigma = utils.lognormal_param_from_samples(noise_variance_list)
        model_params['noise_variance_lognormal_params'] = (noise_variance_mu, noise_variance_sigma)
        print('noise_variance: LogNormal(mu={}, sigma={})'.format(noise_variance_mu, noise_variance_sigma))

    # save model params
    if not os.path.exists(os.path.join(results_dir, output_subdir)):
        os.makedirs(os.path.join(results_dir, output_subdir))
    np.save(os.path.join(results_dir, output_subdir, 'model_params_two_step_setup_b_2.npy'), model_params)
