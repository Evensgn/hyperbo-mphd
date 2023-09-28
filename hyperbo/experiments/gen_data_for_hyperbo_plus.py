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


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    dataset_id = '5'
    n_search_space = 20
    n_funcs = [60 for i in range(n_search_space)]
    n_func_dims = list(np.random.randint(low=5, high=15, size=n_search_space))
    n_discrete_points = [1000 for i in range(n_search_space)]
    cov_func = kernel_list[0][1]
    cov_func_name = kernel_list[0][0]

    const_params = [1, 1]
    ls_params = [10, 30]
    sig_var_params = [1, 1]
    noise_var_params = [10, 100000]

    dataset, gp_params = data.hyperbo_plus_gen_synthetic(key, n_search_space, n_funcs, n_func_dims, n_discrete_points, cov_func, const_params,
                                                         ls_params, sig_var_params, noise_var_params, const_prior='normal', ls_prior='gamma',
                                                         sig_var_prior='gamma', noise_var_prior='gamma')

    # visualize example gps in 1 dimension
    length_scales = gp_params[1]
    mean_func = mean.constant
    print('length_scales:', length_scales)
    print('noise_vars:', gp_params[3])
    for length_scale in length_scales[:5]:
        params_i = defs.GPParams(
            model={
                'constant': 0.0,
                'lengthscale': length_scale,
                'signal_variance': 1.0,
                'noise_variance': 1e-6
        })
        key, _ = jax.random.split(key)
        vx = jnp.expand_dims(jnp.arange(0, 1, 0.01), axis=1)
        key, _ = jax.random.split(key)
        vy = gp.sample_from_gp(key, mean_func, cov_func, params_i, vx, num_samples=5)
        for i in range(5):
            plt.plot(vx, vy[:, i])
        plt.title('length_scale: {}'.format(length_scale))
        plt.show()

    np.save('./synthetic_data/dataset_{}.npy'.format(dataset_id), dataset)

    with open(os.path.join('./synthetic_data/dataset_{}.txt'.format(dataset_id)), 'w') as f:
        f.write('n_search_space: {}\n'.format(n_search_space))
        f.write('n_funcs: {}\n'.format(n_funcs))
        f.write('n_func_dims: {}\n'.format(n_func_dims))
        f.write('n_discrete_points: {}\n'.format(n_discrete_points))
        f.write('cov_func_name: {}\n'.format(cov_func_name))
        f.write('const_params: {}\n'.format(const_params))
        f.write('ls_params: {}\n'.format(ls_params))
        f.write('sig_var_params: {}\n'.format(sig_var_params))
        f.write('noise_var_params: {}\n'.format(noise_var_params))

    print('dataset saved')

