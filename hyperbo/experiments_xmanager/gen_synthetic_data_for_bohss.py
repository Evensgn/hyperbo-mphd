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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    for distribution_type in ['gamma', 'lognormal']:
        dataset_id = 'beta_{}'.format(distribution_type)
        key, subkey = jax.random.split(key)

        cov_func = kernel.matern52
        mean_func = mean.constant

        n_search_space = 20
        n_funcs = [20 for i in range(n_search_space)]
        n_func_dims = list(np.random.randint(low=2, high=15, size=n_search_space))
        n_discrete_points = [3000 for i in range(n_search_space)]

        if distribution_type == 'gamma':
            const_params = {'mu': 0.5, 'sigma': 0.2}
            ls_linear_params = {'alpha_w': 0.07692, 'alpha_b': 0.8462, 'beta_w': -0.3539, 'beta_b': 5.7077}
            sig_var_params = {'alpha': 15., 'beta': 100.}
            noise_var_params = {'alpha': 1., 'beta': 10000.}
        elif distribution_type == 'lognormal':
            const_params = {'mu': 0.5, 'sigma': 0.2}
            ls_linear_params = {'mu_w': 0.2692, 'mu_b': -2.7385, 'sigma_w': -0.03846, 'sigma_b': 1.3769}
            sig_var_params = {'mu': -2.0, 'sigma': 0.25}
            noise_var_params = {'mu': -11.6, 'sigma': 1.6}
        else:
            raise NotImplementedError

        distribution_params = {
            'constant': const_params,
            'lengthscale': ls_linear_params,
            'signal_variance': sig_var_params,
            'noise_variance': noise_var_params,
        }

        const_prior = 'normal'
        ls_prior = distribution_type
        sig_var_prior = distribution_type
        noise_var_prior = distribution_type

        distribution_types = {
            'constant': const_prior,
            'lengthscale': ls_prior,
            'signal_variance': sig_var_prior,
            'noise_variance': noise_var_prior,
        }

        dataset, gp_params = data.gen_synthetic_super_dataset_bohss(
            subkey, n_search_space, n_funcs, n_func_dims, n_discrete_points, cov_func, mean_func, const_params,
            ls_linear_params, sig_var_params, noise_var_params, const_prior=const_prior, ls_prior=ls_prior,
            sig_var_prior=sig_var_prior, noise_var_prior=noise_var_prior,
        )

        configs = {
            'cov_func': 'matern52',
            'mean_func': 'constant',
            'dataset_id': dataset_id,
            'n_search_space': n_search_space,
            'n_funcs': n_funcs,
            'n_func_dims': n_func_dims,
            'n_discrete_points': n_discrete_points,
            'distribution_params': distribution_params,
            'distribution_types': distribution_types,
            'gp_params': gp_params,
        }

        np.save('../synthetic_data/dataset_{}.npy'.format(dataset_id), dataset)
        np.save('../synthetic_data/dataset_{}_configs.npy'.format(dataset_id), configs)

        with open(os.path.join('../synthetic_data/dataset_{}_configs.txt'.format(dataset_id)), 'w') as f:
            f.write(str(configs))

        print('dataset saved')

