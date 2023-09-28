import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
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
import matplotlib.pyplot as plt
import matplotlib
from test_hyperbo_plus_split_worker import fit_gp_params
from multiprocessing import Pool


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


kernel_list = [
    # ('squared_exponential nll', kernel.squared_exponential, obj.nll, 'lbfgs'),
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
    # ('matern52 kl', kernel.matern52, obj.kl, 'lbfgs'),
    # ('matern32_mlp kl', kernel.matern32_mlp, obj.kl, 'lbfgs'),
    # ('matern52_mlp kl', kernel.matern52_mlp, obj.kl, 'lbfgs'),
    # ('squared_exponential_mlp kl', kernel.squared_exponential_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl', kernel.dot_product_mlp, obj.kl, 'lbfgs'),
    # ('dot_product_mlp kl adam', kernel.dot_product_mlp, obj.kl, 'adam'),
    # ('squared_exponential_mlp kl adam', kernel.squared_exponential_mlp, obj.kl, 'adam')
]


def run(args):
    key, gp_params, n_discrete_points_fixed, n_sub_datasets_list, n_sub_datasets_fixed, n_discrete_points_list, \
    cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size, adam_learning_rate = args
    mean_func = mean.constant

    '''
    # fix n_discrete_points, increase n_sub_datasets
    gp_params_fit_list_a = []
    for n_sub_datasets in n_sub_datasets_list:
        print('a: ', n_sub_datasets)
        dataset = {}
        for i in range(n_sub_datasets):
            new_key, key = jax.random.split(key)
            vx = jax.random.uniform(new_key, (n_discrete_points_fixed, 1))
            new_key, key = jax.random.split(key)
            vy = gp.sample_from_gp(new_key, mean_func, cov_func, gp_params, vx, num_samples=1)
            dataset[i] = defs.SubDataset(x=vx, y=vy)
        gp_params_fit_now, _ = fit_gp_params(key, dataset, cov_func, objective, opt_method, gp_fit_maxiter,
                                             fit_gp_batch_size, adam_learning_rate)
        gp_params_fit_list_a.append(gp_params_fit_now)

    '''
    gp_params_fit_list_a = None

    # fix n_sub_datasets, increase n_discrete_points
    gp_params_fit_list_b = []
    for n_discrete_points in n_discrete_points_list:
        print('b: ', n_discrete_points)
        dataset = {}
        for i in range(n_sub_datasets_fixed):
            new_key, key = jax.random.split(key)
            vx = jax.random.uniform(new_key, (n_discrete_points, 1))
            new_key, key = jax.random.split(key)
            vy = gp.sample_from_gp(new_key, mean_func, cov_func, gp_params, vx, num_samples=1)
            dataset[i] = defs.SubDataset(x=vx, y=vy)
        gp_params_fit_now, _ = fit_gp_params(key, dataset, cov_func, objective, opt_method, gp_fit_maxiter,
                                             fit_gp_batch_size, adam_learning_rate)
        gp_params_fit_list_b.append(gp_params_fit_now)

    return gp_params_fit_list_a, gp_params_fit_list_b


def plot_for_one_param(value_list, x_label, x_list, y_label, ground_truth, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.tick_params(axis='y', labelsize=8)

    value_list = np.array(value_list).squeeze()
    mean_list = np.mean(value_list, axis=0)
    std_list = np.std(value_list, axis=0)

    '''
    for values in value_list:
        ax.plot(x_list, values, color='purple', alpha=0.05)
    '''

    data = [value_list[:, i] for i in range(value_list.shape[1])]
    ax.violinplot(data, positions=x_list, showmeans=True, widths=3)

    line = ax.plot(x_list, mean_list, label='Mean Estimate')[0]
    # ax.fill_between(x_list, mean_list - std_list, mean_list + std_list, alpha=0.2, color=line.get_color())
    ax.plot(x_list, ground_truth * np.ones_like(x_list), '--', label='Ground-truth')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_results(results, n_discrete_points_fixed, n_sub_datasets_list, n_sub_datasets_fixed, n_discrete_points_list, gp_params, dir_path):
    for param_key in gp_params.model.keys():
        param_name = param_names[param_key]
        ground_truth = gp_params.model[param_key]

        param_value_list_a_list = []
        param_value_list_b_list = []
        for result in results:
            gp_params_fit_list_a, gp_params_fit_list_b = result
            param_value_list_a = [gp_params_fit_now[param_key] for gp_params_fit_now in gp_params_fit_list_a]
            param_value_list_a_list.append(param_value_list_a)
            # param_value_list_b = [gp_params_fit_now[param_key] for gp_params_fit_now in gp_params_fit_list_b]
            # param_value_list_b_list.append(param_value_list_b)

        plot_for_one_param(param_value_list_a_list, 'Number of Sub-datasets', n_sub_datasets_list, param_name, ground_truth, os.path.join(dir_path, f'{param_key}_n_sub_datasets_{n_discrete_points_fixed}_observations_2.pdf'))
        # plot_for_one_param(param_value_list_b_list, '# of observations', n_discrete_points_list, param_name, ground_truth, os.path.join(dir_path, f'{param_name}_n_discrete_points_{n_sub_datasets_fixed}_sub_datasets.pdf'))



param_names = {
    'constant': 'Constant',
    'lengthscale': 'Length-scale',
    'signal_variance': 'Signal Variance',
    'noise_variance': 'Noise Variance',
}


if __name__ == '__main__':
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    axes = {}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)

    key = jax.random.PRNGKey(0)

    n_random_seeds = 50

    save_id = 'asymptotic_analysis_7'
    n_discrete_points_fixed = 25
    n_sub_datasets_list = list(range(5, 105, 5))

    n_sub_datasets_fixed = 50
    n_discrete_points_list = list(range(5, 55, 5))

    gp_fit_maxiter = 100  # 50000 for adam (5000 ~ 6.5 min per id), 500 for lbfgs
    fit_gp_batch_size = 1000  # 50 for adam, 300 for lbfgs
    adam_learning_rate = 0.001

    kernel_type = kernel_list[0]
    kernel_name, cov_func, objective, opt_method = kernel_type

    gp_params = defs.GPParams(
        model={
            'constant': 0.0,
            'lengthscale': 0.3,
            'signal_variance': 1.0,
            'noise_variance': 1e-4
        }
    )

    '''
    p = Pool(n_random_seeds)
    args_list = []
    for i in range(n_random_seeds):
        new_key, key = jax.random.split(key)
        args_list.append((new_key, gp_params, n_discrete_points_fixed, n_sub_datasets_list, n_sub_datasets_fixed,
                          n_discrete_points_list, cov_func, objective, opt_method, gp_fit_maxiter, fit_gp_batch_size,
                          adam_learning_rate))
    results = p.map(run, args_list)
    '''

    # save all results
    dir_path = os.path.join('../results', save_id)
    if not os.path.exists('../results'):
        os.makedirs('../results')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # np.save(os.path.join(dir_path, 'results.npy'), results)
    results = np.load(os.path.join('../results', save_id, 'results.npy'), allow_pickle=True)

    plot_results(results, n_discrete_points_fixed, n_sub_datasets_list, n_sub_datasets_fixed, n_discrete_points_list, gp_params, dir_path)

