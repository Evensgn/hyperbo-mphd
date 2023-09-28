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
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma
from functools import partial
import matplotlib.pyplot as plt
import gc
import subprocess
from experiment_defs import *


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit

if __name__ == '__main__':
    group_id = GROUP_ID
    results_dir = RESULTS_DIR

    is_hpob = IS_HPOB

    train_id_list = TRAIN_ID_LIST
    test_id_list = TEST_ID_LIST
    setup_b_id_list = FULL_ID_LIST

    if is_hpob:
        normalize_x = True
        normalize_y = True
        hpob_negative_y = False
        hpob_data_path = HPOB_DATA_PATH
        pd1_data_path = PD1_DATA_PATH
        dataset_func_combined = partial(data.hpob_dataset_v2, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                        normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hpob_dataset_v3, hpob_data_path=hpob_data_path, pd1_data_path=pd1_data_path,
                                     negative_y=hpob_negative_y,
                                     normalize_x=normalize_x, normalize_y=normalize_y, read_init_index=True)
        dataset_dim_feature_values_path = HPOB_DATA_ANALYSIS_PATH
        extra_info = 'hpob_negative_y: {}, noramlize_x: {}, normalize_y: {}'.format(hpob_negative_y, normalize_x, normalize_y)
    else:
        normalize_x = False
        normalize_y = False
        synthetic_data_path = SYNTHETIC_DATA_PATH
        dataset_func_combined = partial(data.hpl_bo_synthetic_dataset_combined, synthetic_data_path,
                                        normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_func_split = partial(data.hpl_bo_synthetic_dataset_split, synthetic_data_path,
                                     normalize_x=normalize_x, normalize_y=normalize_y)
        dataset_dim_feature_values_path = SYNTHETIC_DATA_ANALYSIS_PATH
        extra_info = 'synthetic_data_path = \'{}\', noramlize_x: {}, normalize_y: {}'.format(synthetic_data_path, normalize_x, normalize_y)

    random_seed = RANDOM_SEED

    fit_gp_method = 'adam'
    fit_gp_maxiter = 20000
    fit_gp_batch_size = 50
    fit_gp_adam_learning_rate = 0.001

    fit_hgp_maxiter = 10000
    fit_hgp_batch_size = 50
    fit_hgp_adam_learning_rate = 0.001

    fit_two_step_maxiter = 10000
    fit_two_step_learning_rate = 0.001

    gp_retrain_maxiter = 100
    gp_retrain_method = 'lbfgs'

    n_init_obs = 5
    budget = 100  # 50
    n_bo_runs = 5

    eval_loss_batch_size = 50
    eval_loss_n_batches = 20

    eval_nll_gp_retrain_maxiter = 500
    eval_nll_gp_retrain_method = 'lbfgs'
    eval_nll_gp_retrain_n_observations = 10
    eval_nll_gp_retrain_n_batches = 20
    eval_nll_batch_size = 300
    eval_nll_n_batches = 10
    ac_func_type_list = AC_FUNC_TYPE_LIST

    distribution_type = DISTRIBUTION_TYPE

    if distribution_type == 'gamma':
        hand_hgp_params = {
            'constant': (0.5, 0.5),
            'lengthscale': (1.0, 0.1),
            'signal_variance': (1.0, 5.0),
            'noise_variance': (1.0, 100.0),
        }
    elif distribution_type == 'lognormal':
        hand_hgp_params = {
            'constant': (0.5, 0.5),
            'lengthscale': (1.0, 1.5),
            'signal_variance': (-2.0, 1.0),
            'noise_variance': (-6.0, 1.0),
        }
    else:
        raise NotImplementedError

    uniform_hgp_params = {
        'constant': (0., 1.),
        'lengthscale': (0.00001, 30.0),
        'signal_variance': (0.00001, 1.0),
        'noise_variance': (0.00001, 0.1),
    }

    log_uniform_hgp_params = {
        'constant': (0., 1.),
        'lengthscale': (-5.0, 5.0),
        'signal_variance': (-5.0, 5.0),
        'noise_variance': (-5.0, 5.0),
    }

    method_name_list = METHOD_NAME_LIST
    pd1_method_name_list = PD1_METHOD_NAME_LIST

    if is_hpob:
        gt_hgp_params = None
        gt_gp_params = None
    else:
        # ground truth for synthetic alpha 1
        synthetic_data_configs = np.load(SYNTHETIC_DATA_CONFIG_PATH, allow_pickle=True).item()
        gt_hgp_params = synthetic_data_configs['distribution_params']
        gt_gp_params = synthetic_data_configs['gp_params']

    kernel_name = 'matern52 adam'
    cov_func = kernel.matern52
    mean_func = mean.constant
    gp_objective = obj.nll
    hgp_objective = obj.neg_log_marginal_likelihood_hgp_v3

    fitting_node_cpu_count = FITTING_NODE_CPU_COUNT
    bo_node_cpu_count = BO_NODE_CPU_COUNT
    nll_node_cpu_count = NLL_NODE_CPU_COUNT

    split_dir = os.path.join(results_dir, 'hpl_bo_split')
    dir_path = os.path.join(split_dir, group_id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # write configs
    configs = {
        'random_seed': random_seed,
        'train_id_list': train_id_list,
        'test_id_list': test_id_list,
        'setup_b_id_list': setup_b_id_list,
        'dataset_func_combined': dataset_func_combined,
        'dataset_func_split': dataset_func_split,
        'dataset_dim_feature_values_path': dataset_dim_feature_values_path,
        'extra_info': extra_info,

        'synthetic_smaller_train_id_lists': SYNTHETIC_SMALLER_TRAIN_ID_LISTS,

        'fit_gp_maxiter': fit_gp_maxiter,
        'fit_gp_batch_size': fit_gp_batch_size,
        'fit_gp_adam_learning_rate': fit_gp_adam_learning_rate,

        'fit_hgp_maxiter': fit_hgp_maxiter,
        'fit_hgp_batch_size': fit_hgp_batch_size,
        'fit_hgp_adam_learning_rate': fit_hgp_adam_learning_rate,

        'fit_two_step_maxiter': fit_two_step_maxiter,
        'fit_two_step_learning_rate': fit_two_step_learning_rate,

        'gp_retrain_maxiter': gp_retrain_maxiter,
        'gp_retrain_method': gp_retrain_method,

        'n_init_obs': n_init_obs,
        'budget': budget,
        'n_bo_runs': n_bo_runs,

        'eval_loss_batch_size': eval_loss_batch_size,
        'eval_loss_n_batches': eval_loss_n_batches,

        'eval_nll_gp_retrain_maxiter': eval_nll_gp_retrain_maxiter,
        'eval_nll_gp_retrain_method': eval_nll_gp_retrain_method,
        'eval_nll_gp_retrain_n_observations': eval_nll_gp_retrain_n_observations,
        'eval_nll_gp_retrain_n_batches': eval_nll_gp_retrain_n_batches,
        'eval_nll_batch_size': eval_nll_batch_size,
        'eval_nll_n_batches': eval_nll_n_batches,

        'ac_func_type_list': ac_func_type_list,

        'hand_hgp_params': hand_hgp_params,
        'uniform_hgp_params': uniform_hgp_params,
        'log_uniform_hgp_params': log_uniform_hgp_params,
        'gt_hgp_params': gt_hgp_params,
        'gt_gp_params': gt_gp_params,

        'kernel_name': kernel_name,
        'cov_func': cov_func,
        'mean_func': mean_func,
        'gp_objective': gp_objective,
        'hgp_objective': hgp_objective,
        'fit_gp_method': fit_gp_method,
        'distribution_type': distribution_type,

        'fitting_node_cpu_count': fitting_node_cpu_count,
        'bo_node_cpu_count': bo_node_cpu_count,
        'nll_node_cpu_count': nll_node_cpu_count,

        'method_name_list': method_name_list,
        'pd1_method_name_list': pd1_method_name_list,
    }
    np.save(os.path.join(dir_path, 'configs.npy'), configs)

