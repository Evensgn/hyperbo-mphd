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


def plot_single_gp_samples(key, length_scale, mean_func, cov_func):
    params = defs.GPParams(
        model={
            'constant': 0.0,
            'lengthscale': length_scale,
            'signal_variance': 1.0,
            'noise_variance': 1e-6
        }
    )
    dir_path = os.path.join('results', 'gen_gp_plots')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    matplotlib.rc('font', size=40)

    colors = ['c', 'b']
    for i in range(2):
        key, _ = jax.random.split(key)
        vx = jax.random.uniform(key, (150, 1))
        key, _ = jax.random.split(key)
        vy = gp.sample_from_gp(key, mean_func, cov_func, params, vx, num_samples=1)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12.5, 5))
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.scatter(vx, vy[:, 0], color=colors[i])
        ax.set(yticklabels=[])
        ax.tick_params(left=False)
        # ax.tick_params(axis='both', which='major')
        fig.savefig(os.path.join(dir_path, 'single_gp_sample_{}.pdf'.format(i)))

        ax.axis('off')
        fig.savefig(os.path.join(dir_path, 'single_gp_sample_{}_noaxis.pdf'.format(i)))
        plt.close(fig)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    length_scale = 0.1
    mean_func = mean.constant
    cov_func = kernel.matern32
    plot_single_gp_samples(key, length_scale, mean_func, cov_func)


