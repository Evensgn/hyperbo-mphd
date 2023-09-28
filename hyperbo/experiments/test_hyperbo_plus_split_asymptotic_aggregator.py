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
import matplotlib
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
    # ('matern32 adam', kernel.matern32, obj.nll, 'adam'),
    ('matern32 nll', kernel.matern32, obj.nll, 'lbfgs'),
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


def plot_for_one_param(value_list, x_label, x_list, y_label, ground_truth, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # value_list = np.array(value_list).squeeze()
    # mean_list = np.mean(value_list, axis=0)
    # std_list = np.std(value_list, axis=0)

    '''
    for values in value_list:
        ax.plot(x_list, values, color='purple', alpha=0.05)
    '''

    # data = [value_list[:, i] for i in range(value_list.shape[1])]
    mean_list = np.mean(value_list, axis=1)
    data = value_list
    ax.violinplot(data, positions=x_list, showmeans=True, widths=0.7)

    # ax.fill_between(x_list, mean_list - std_list, mean_list + std_list, alpha=0.2, color=line.get_color())
    if ground_truth is not None:
        line = ax.plot(x_list, mean_list, label='Mean Estimate')[0]
        ax.plot(x_list, ground_truth * np.ones_like(x_list), '--', label='Ground-truth')
    else:
        line = ax.plot(x_list, mean_list, label='Mean NLL')[0]
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_for_one_param_distribution(x_list, value_a_list, value_b_list, x_label, y_label, ground_truth_a, ground_truth_b, dist_type, x_range, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.tick_params(axis='y', labelsize=9)

    x = np.linspace(x_range[0], x_range[1], 100)
    dist = dist_type(ground_truth_a, ground_truth_b)
    ax.plot(x, dist.prob(x), color='red', alpha=0.7, label='Ground-truth')

    for i in range(len(value_a_list[0])):
        a, b = value_a_list[0][i], value_b_list[0][i]
        dist = dist_type(a, b)
        if i == 0:
            ax.plot(x, dist.prob(x), color='green', alpha=0.7, label='# of Training Datasets = {}'.format(x_list[0]))
        else:
            ax.plot(x, dist.prob(x), color='green', alpha=0.7)

    for i in range(len(value_a_list[-1])):
        a, b = value_a_list[-1][i], value_b_list[-1][i]
        dist = dist_type(a, b)
        if i == 0:
            ax.plot(x, dist.prob(x), color='blue', alpha=0.7, label='# of Training Datasets = {}'.format(x_list[-1]))
        else:
            ax.plot(x, dist.prob(x), color='blue', alpha=0.7)

    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_results(results, dir_path):
    n_train_datasets_list, n_seeds, mle_distribution_params_list, nll_results_list, gt_gp_distribution_params = \
        results['n_train_datasets_list'], results['n_seeds'], results['mle_distribution_params_list'], \
        results['nll_results_list'], results['gt_gp_distribution_params']

    for param_key in gt_gp_distribution_params.keys():
        param_name = param_names[param_key]
        ground_truth_a = gt_gp_distribution_params[param_key][0]
        ground_truth_b = gt_gp_distribution_params[param_key][1]

        param_mle_a_list = []
        param_mle_b_list = []
        for i in range(len(n_train_datasets_list)):
            param_mle_a_list.append([x[param_key][0] for x in mle_distribution_params_list[i]])
            param_mle_b_list.append([x[param_key][1] for x in mle_distribution_params_list[i]])

        param_distribution = param_distributions[param_key]

        plot_for_one_param(param_mle_a_list, 'Number of Training Datasets', n_train_datasets_list, 'Estimated {} of {} Prior'.format(param_distribution[1], param_name), ground_truth_a, os.path.join(dir_path, f'mle_{param_key}_n_train_datasets.pdf'))
        plot_for_one_param_distribution(n_train_datasets_list, param_mle_a_list, param_mle_b_list, '{}'.format(param_name), 'Probability Density', ground_truth_a, ground_truth_b, param_distribution[0], param_distribution[2], os.path.join(dir_path, f'mle_distribution_{param_key}_n_train_datasets.pdf'))

    test_nll_mean_list = []
    for i in range(len(n_train_datasets_list)):
        test_nll_mean_list_i = []
        for j in range(n_seeds):
            test_nll_mean_list_i.append(nll_results_list[i][j]['gamma_nll_on_test_mean'])
        test_nll_mean_list.append(test_nll_mean_list_i)
    plot_for_one_param(test_nll_mean_list, 'Number of Training Datasets', n_train_datasets_list, 'NLL of Test Datasets',
                       None, os.path.join(dir_path, f'test_nll_n_train_datasets.pdf'))


param_names = {
    'constant': 'Constant',
    'lengthscale': 'Length-scale',
    'signal_variance': 'Signal Variance',
    'noise_variance': 'Noise Variance',
}

param_distributions = {
    'constant': [Normal, '$c$', [-1, 1]],
    'lengthscale': [Gamma, '$a$', [0, 2]],
    'signal_variance': [Gamma, '$a$', [0.5, 2]],
    'noise_variance': [Gamma, '$a$', [0, 1e-3]]
}

if __name__ == '__main__':
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20}
    axes = {}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)

    n_train_datasets_list = list(range(2, 17))
    n_seeds = 5
    aggregation_dir_path = os.path.join('../results', 'test_hyperbo_plus_split_asymptotic_s4_2')

    # separation

    '''
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
    dataset_func_combined = partial(data.hpob_dataset_v2, negative_y=hpob_negative_y)
    dataset_func_split = partial(data.hpob_dataset_v3, negative_y=hpob_negative_y)
    extra_info = 'hpob_negative_y_{}'.format(hpob_negative_y)

    # hpob_converted_data_path = './hpob_converted_data/sub_sample_1000.npy'
    # dataset_func_combined = partial(data.hpob_converted_dataset_combined, hpob_converted_data_path)
    # dataset_func_split = partial(data.hpob_converted_dataset_split, hpob_converted_data_path)
    # extra_info = 'hpob_converted_data_path = \'{}\''.format(hpob_converted_data_path)
    '''

    # train_id_list = [0]
    # test_id_list = list(range(4, 6))
    # setup_b_id_list = list(range(6))
    # train_id_list = list(range(16))
    test_id_list = list(range(16, 20))
    setup_b_id_list = list(range(20))

    test_id_list = [str(x) for x in test_id_list]
    setup_b_id_list = [str(x) for x in setup_b_id_list]
    synthetic_data_path = './synthetic_data/dataset_4.npy'
    dataset_func_combined = partial(data.hyperbo_plus_synthetic_dataset_combined, synthetic_data_path)
    dataset_func_split = partial(data.hyperbo_plus_synthetic_dataset_split, synthetic_data_path)
    extra_info = 'synthetic_data_path = \'{}\''.format(synthetic_data_path)

    n_workers = 25
    n_init_obs = 5
    budget = 50  # 50
    n_bo_runs = 5
    gp_fit_maxiter = 500  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    n_bo_gamma_samples = 100  # 100
    n_nll_gamma_samples = 500  # 500
    setup_a_nll_sub_dataset_level = True
    fit_gp_batch_size = 50  # 50 for adam, 300 for lbfgs
    bo_sub_sample_batch_size = 1000  # 2000 for hpob, 300 for synthetic 4, 1000 for synthetic 5
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

    # ground truth for synthetic 4
    gt_gp_distribution_params = {
        'constant': (1.0, 1.0),
        'lengthscale': (10.0, 30.0),
        'signal_variance': (1.0, 1.0),
        'noise_variance': (10.0, 100000.0)
    }
    '''
    gt_gp_distribution_params = None
    '''

    '''
    kernel_type = kernel_list[0]
    # construct the jax random key
    key = jax.random.PRNGKey(0)
    
    mle_distribution_params_list = []
    nll_results_list = []
    for n_train_datasets in n_train_datasets_list:
        mle_distribution_params_list.append([])
        nll_results_list.append([])

        group_id_list = []
        train_id_list_list = []
        for seed_id in range(n_seeds):
            group_id = 'split_asymptotic_s4_2_n_train_datasets_{}_seed_{}'.format(n_train_datasets, seed_id)  # 'split_synthetic_2' 'split_hpob_neg_2'
            group_id_list.append(group_id)

            train_id_list = list(np.random.choice(16, n_train_datasets, replace=False))  # train_id_list = list(range(16))
            train_id_list = [str(x) for x in train_id_list]
            train_id_list_list.append(train_id_list)

            dir_path = os.path.join('results', 'test_hyperbo_plus_split', group_id)
            gp_distribution_params = np.load(os.path.join(dir_path, 'split_alpha_mle_setup_a.npy'), allow_pickle=True).item()['gp_distribution_params']
            mle_distribution_params_list[-1].append(gp_distribution_params)

            gamma_nll_on_test_list = []
            for k in range(eval_nll_n_batches):
                gamma_nll_on_test_list.append([])

            for test_id in test_id_list:
                nll_results_test_id = np.load(
                    os.path.join(dir_path, 'split_eval_nll_setup_a_id_{}.npy'.format(test_id)),
                    allow_pickle=True).item()

                gamma_nll_on_test_batches_i, _ = nll_results_test_id['gamma_nll_on_batches_i'], nll_results_test_id[
                    'gamma_n_for_sdl']
                for k in range(eval_nll_n_batches):
                    gamma_nll_on_test_list[k].append(gamma_nll_on_test_batches_i[k])


            gamma_nll_on_test_batches = []
            for k in range(eval_nll_n_batches):
                gamma_nll_on_test_batches.append(np.mean(gamma_nll_on_test_list[k]))

            nll_results = {
                'gamma_nll_on_test_mean': np.mean(gamma_nll_on_test_batches),
                'gamma_nll_on_test_std': np.std(gamma_nll_on_test_batches)
            }

            nll_results_list[-1].append(nll_results)

    results = {
        'n_train_datasets_list': n_train_datasets_list,
        'n_seeds': n_seeds,
        'mle_distribution_params_list': mle_distribution_params_list,
        'nll_results_list': nll_results_list,
        'gt_gp_distribution_params': gt_gp_distribution_params
    }

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(aggregation_dir_path):
        os.mkdir(aggregation_dir_path)

    np.save(os.path.join(aggregation_dir_path, 'results.npy'), results)
    '''
    results = np.load(os.path.join(aggregation_dir_path, 'results.npy'), allow_pickle=True).item()

    plot_results(results, aggregation_dir_path)

