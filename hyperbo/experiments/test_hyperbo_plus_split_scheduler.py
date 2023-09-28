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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


hpob_full_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                     '6794', '7607', '7609', '5889']

kernel_list = [
    # ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
    ('matern32 adam', kernel.matern32, obj.nll, 'adam'),
    # ('matern32 nll', kernel.matern32, obj.nll, 'lbfgs'),
    # ('matern52 nll', kernel.matern52, obj.nll, 'lbfgs'),
    # ('matern32_mlp nll', kernel.matern32_mlp, obj.nll, 'lbfgs'),
    # ('matern52_mlp nll', kernel.matern52_mlp, obj.nll, 'lbfgs'),
    # ('squared_exponential_mlp nll', kernel.squared_exponential_mlp, obj.nll, 'lbfgs'),
    # ('dot_product_mlp nll', kernel.dot_product_mlp, obj.nll, 'lbfgs'),
    # ('dot_product_mlp nll adam', kernel.dot_product_mlp, obj.nll, 'adam'),
    # ('squared_exponential_mlp nll adam', kernel.squared_exponential_mlp, obj.nll, 'adam'),

    # ('squared_exponential kl', kernel.squared_exponential, obj.kl, 'lbfgs'),
    # ('matern32 kl', kernel.matern32, obj.kl, 'lbfgs'),
    # ('matern52 kl', kerne.matern52, obj.kl, 'lbfgs'),
    # ('matern32_mlp kl', kernel.matern32_mlp, obj.kl, 'lbfgs'),
    # ('matern52_mlp kl', kernel.matern52_mlp, obj.kl, 'lbfgs'),
    # ('squared_exponential_mlp kl', kernel.squared_exponential_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl', kernel.dot_product_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl adam', kernel.dot_product_mlp, obj.kl, 'adam'),
    # ('squared_exponential_mlp kl adam', kernel.squared_exponential_mlp, obj.kl, 'adam')
]


if __name__ == '__main__':
    group_id = 'split_hpob_normalize_xy_only_fitting'  # 'split_hpob_neg_2'
    python_cmd = '/home/zfan/hyperbo/env-pd/bin/python'
    worker_path = '/home/zfan/hyperbo/hyperbo/experiments/test_hyperbo_plus_split_worker_old.py'

    # train_id_list = ['5860', '5906']
    # test_id_list = ['5889']
    # train_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
    # train_id_list = ['6766', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6767', '6794']
    # train_id_list = ['4796', '5527', '5636', '5859', '5891', '5965', '5970', '5971', '6766', '6767', '6794', '7607', '7609']
    # train_id_list = ['4796', '5527', '5636']
    # test_id_list = ['4796', '5860', '5906', '7607', '7609', '5889']
    # test_id_list = ['5860', '5906', '5889']
    # setup_b_id_list = ['4796', '5860', '5906', '7607', '7609', '5889']
    # setup_b_id_list = ['5860', '5906', '5889']

    # train_id_list = ['4796', '5527']
    train_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
    test_id_list = ['6794', '7607', '7609', '5889']
    setup_b_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                       '6794', '7607', '7609', '5889']

    # train_id_list = ['4796', '5527', '5636', '5859', '5860']
    # test_id_list = ['6794', '7607']
    # setup_b_id_list = ['4796', '5527', '5636', '5859', '5860']

    hpob_negative_y = False
    normalize_x = True
    normalize_y = True
    dataset_func_combined = partial(data.hpob_dataset_v2, negative_y=hpob_negative_y, normalize_x=normalize_x,
                                    normalize_y=normalize_y)
    dataset_func_split = partial(data.hpob_dataset_v3, negative_y=hpob_negative_y, normalize_x=normalize_x,
                                 normalize_y=normalize_y)
    extra_info = 'hpob_negative_y_{}_noramlize_x_{}_normalize_y_{}'.format(hpob_negative_y, normalize_x, normalize_y)

    # hpob_converted_data_path = './hpob_converted_data/sub_sample_1000.npy'
    # dataset_func_combined = partial(data.hpob_converted_dataset_combined, hpob_converted_data_path)
    # dataset_func_split = partial(data.hpob_converted_dataset_split, hpob_converted_data_path)
    # extra_info = 'hpob_converted_data_path = \'{}\''.format(hpob_converted_data_path)

    '''
    # train_id_list = [0]
    train_id_list = list(range(16))
    # test_id_list = list(range(4, 6))
    # setup_b_id_list = list(range(6))
    # train_id_list = list(range(16))
    test_id_list = list(range(16, 20))
    setup_b_id_list = list(range(20))

    train_id_list = [str(x) for x in train_id_list]
    test_id_list = [str(x) for x in test_id_list]
    setup_b_id_list = [str(x) for x in setup_b_id_list]
    synthetic_data_path = './synthetic_data/dataset_4.npy'
    dataset_func_combined = partial(data.hyperbo_plus_synthetic_dataset_combined, synthetic_data_path)
    dataset_func_split = partial(data.hyperbo_plus_synthetic_dataset_split, synthetic_data_path)
    extra_info = 'synthetic_data_path = \'{}\''.format(synthetic_data_path)
    '''

    random_seed = 0
    n_init_obs = 5
    budget = 50  # 50
    n_bo_runs = 5
    gp_fit_maxiter = 10000  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    n_bo_gamma_samples = 100  # 100
    n_nll_gamma_samples = 500  # 500
    setup_a_nll_sub_dataset_level = True
    fit_gp_batch_size = 50  # 50 for adam, 300 for lbfgs
    bo_sub_sample_batch_size = 1000  # 1000 for hpob, 300 for synthetic 4, 1000 for synthetic 5
    adam_learning_rate = 0.001
    eval_nll_batch_size = 100  # 300
    eval_nll_n_batches = 10
    ac_func_type_list = ['ucb', 'ei', 'pi']

    fixed_gp_distribution_params = {
        'constant': (0.0, 1.0),
        'lengthscale': (1.0, 10.0),
        'signal_variance': (1.0, 5.0),
        'noise_variance': (10.0, 100.0)
    }

    '''
    # ground truth for synthetic 4
    gt_gp_distribution_params = {
        'constant': (1.0, 1.0),
        'lengthscale': (10.0, 30.0),
        'signal_variance': (1.0, 1.0),
        'noise_variance': (10.0, 100000.0)
    }
    '''
    gt_gp_distribution_params = None

    kernel_type = kernel_list[0]

    dir_path = os.path.join('results', 'test_hyperbo_plus_split', group_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/test_hyperbo_plus_split'):
        os.makedirs('results/test_hyperbo_plus_split')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # construct the jax random key
    key = jax.random.PRNGKey(random_seed)

    # write configs
    configs = {
        'random_seed': random_seed,
        'train_id_list': train_id_list,
        'test_id_list': test_id_list,
        'setup_b_id_list': setup_b_id_list,
        'dataset_func_combined': dataset_func_combined,
        'dataset_func_split': dataset_func_split,
        'extra_info': extra_info,
        'n_init_obs': n_init_obs,
        'budget': budget,
        'n_bo_runs': n_bo_runs,
        'gp_fit_maxiter': gp_fit_maxiter,
        'n_bo_gamma_samples': n_bo_gamma_samples,
        'n_nll_gamma_samples': n_nll_gamma_samples,
        'setup_a_nll_sub_dataset_level': setup_a_nll_sub_dataset_level,
        'fit_gp_batch_size': fit_gp_batch_size,
        'bo_sub_sample_batch_size': bo_sub_sample_batch_size,
        'adam_learning_rate': adam_learning_rate,
        'eval_nll_batch_size': eval_nll_batch_size,
        'eval_nll_n_batches': eval_nll_n_batches,
        'ac_func_type_list': ac_func_type_list,
        'fixed_gp_distribution_params': fixed_gp_distribution_params,
        'gt_gp_distribution_params': gt_gp_distribution_params,
        'kernel_type': kernel_type
      }
    np.save(os.path.join(dir_path, 'configs.npy'), configs)

    time_0 = time.time()

    # fit_gp_params_setup_a_id
    print('fit_gp_params_setup_a_id')
    sub_process_list = []
    for train_id in train_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_gp_params_setup_a_id', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    # for sub_process_i in sub_process_list:
    #     sub_process_i.wait()
    time_1 = time.time()

    # fit_gp_params_setup_b_id
    print('fit_gp_params_setup_b_id')
    # sub_process_list = []
    for train_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'fit_gp_params_setup_b_id', '--dataset_id', train_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    for sub_process_i in sub_process_list:
        sub_process_i.wait()
    time_2 = time.time()

    # alpha_mle_setup_a
    print('alpha_mle_setup_a')
    new_key, key = jax.random.split(key)
    sub_process_0 = subprocess.Popen(
        [python_cmd, worker_path, '--group_id', group_id, '--mode', 'alpha_mle_setup_a',
         '--dataset_id', '', '--key_0', str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_0.wait()
    time_3 = time.time()

    # alpha_mle_setup_b
    print('alpha_mle_setup_b')
    new_key, key = jax.random.split(key)
    sub_process_0 = subprocess.Popen(
        [python_cmd, worker_path, '--group_id', group_id, '--mode', 'alpha_mle_setup_b',
         '--dataset_id', '', '--key_0', str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_0.wait()
    time_4 = time.time()
    '''

    # test_bo_setup_a_id
    print('test_bo_setup_a_id')
    sub_process_list = []
    for test_id in test_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'test_bo_setup_a_id', '--dataset_id', test_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    # for sub_process_i in sub_process_list:
    #     sub_process_i.wait()
    time_5 = time.time()

    # test_bo_setup_b_id
    print('test_bo_setup_b_id')
    # sub_process_list = []
    for test_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'test_bo_setup_b_id', '--dataset_id', test_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    for sub_process_i in sub_process_list:
        sub_process_i.wait()
    time_6 = time.time()

    # eval_nll_setup_a_id
    print('eval_nll_setup_a_id')
    sub_process_list = []
    for dataset_id in (train_id_list + test_id_list):
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_a_id', '--dataset_id', dataset_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    # for sub_process_i in sub_process_list:
    #    sub_process_i.wait()
    time_7 = time.time()

    # eval_nll_setup_b_train_id
    print('eval_nll_setup_b_train_id')
    # sub_process_list = []
    for dataset_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_b_train_id', '--dataset_id', dataset_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    # for sub_process_i in sub_process_list:
    #    sub_process_i.wait()
    time_8 = time.time()

    # eval_nll_setup_b_test_id
    print('eval_nll_setup_b_test_id')
    # sub_process_list = []
    for dataset_id in setup_b_id_list:
        new_key, key = jax.random.split(key)
        sub_process_i = subprocess.Popen([python_cmd, worker_path, '--group_id', group_id,
                                          '--mode', 'eval_nll_setup_b_test_id', '--dataset_id', dataset_id, '--key_0',
                                          str(new_key[0]), '--key_1', str(new_key[1])])
        sub_process_list.append(sub_process_i)
    for sub_process_i in sub_process_list:
        sub_process_i.wait()
    time_9 = time.time()
    '''

    # merge
    print('merge')
    new_key, key = jax.random.split(key)
    sub_process_0 = subprocess.Popen(
        [python_cmd, worker_path, '--group_id', group_id, '--mode', 'merge',
         '--dataset_id', '', '--key_0', str(new_key[0]), '--key_1', str(new_key[1])])
    sub_process_0.wait()
    time_10 = time.time()

    '''
    
    time_fit_gp = (time_4 - time_0) / 3600
    time_run_bo = (time_6 - time_4) / 3600
    time_eval_nll = (time_9 - time_6) / 3600

    print('time_fit_gp', time_fit_gp)
    print('time_run_bo', time_run_bo)
    print('time_eval_nll', time_eval_nll)
    print('done')
    '''
