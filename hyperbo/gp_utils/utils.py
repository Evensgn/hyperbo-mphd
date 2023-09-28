# coding=utf-8
# Copyright 2022 HyperBO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utils for gp_utils."""

import logging

from hyperbo.basics import definitions as defs
from hyperbo.basics import linalg
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import scipy
from functools import partial
import numpy as np
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma, LogNormal, Uniform


# import jax._src.dtypes as dtypes

SubDataset = defs.SubDataset

vmap = jax.vmap
EPS = 1e-10

identity_warp = lambda x: x
softplus_warp = jax.nn.softplus
softplus_inverse_warp = tfp.math.softplus_inverse


def sub_sample_dataset_iterator(key, dataset, batch_size):
  """Iterator for subsample a dataset such that each sub_dataset has at most batch_size data points.

  Args:
    key: Jax random state.
    dataset: dict of SubDataset.
    batch_size: int, maximum number of data points per sub dataset in a batch.

  Yields:
    A sub sampled dataset batch.
  """
  while True:
    sub_sampled_dataset = {}
    for sub_dataset_key, sub_dataset in dataset.items():
      if sub_dataset.x.shape[0] >= batch_size:
        key, subkey = jax.random.split(key, 2)
        indices = jax.random.permutation(subkey, sub_dataset.x.shape[0])
        ifaligned = sub_dataset.aligned
        sub_sampled_dataset[sub_dataset_key] = SubDataset(
            x=sub_dataset.x[indices[:batch_size], :],
            y=sub_dataset.y[indices[:batch_size], :],
            aligned=ifaligned)
      else:
        sub_sampled_dataset[sub_dataset_key] = sub_dataset
    yield sub_sampled_dataset


def squareplus_warp(x):
  """Alternative to softplus with nicer properties.

  See https://twitter.com/jon_barron/status/1387167648669048833

  Args:
    x: scalar or numpy array.

  Returns:
    The transformed x.
  """
  return 0.5 * (x + jnp.sqrt(x**2 + 4))


DEFAULT_WARP_FUNC = {
    'constant': identity_warp,
    'lengthscale': lambda x: softplus_warp(x) + EPS,
    'signal_variance': lambda x: softplus_warp(x) + EPS,
    'noise_variance': lambda x: softplus_warp(x) + EPS,
}

DEFAULT_WARP_FUNC_LC_ONLY = {
    'constant': identity_warp,
    'lengthscale': lambda x: softplus_warp(x) + EPS,
    'signal_variance': identity_warp,
    'noise_variance': identity_warp,
}


def apply_warp_func(params, warp_func):
    if warp_func is None:
        return params
    warpped_params = {}
    for key, value in params.items():
        if key in warp_func:
            warpped_params[key] = warp_func[key](value)
        else:
            warpped_params[key] = value
    return warpped_params


single_gp_default_warp_func = {
    'constant': identity_warp,
    'lengthscale': lambda x: softplus_warp(x) + EPS,
    'signal_variance': lambda x: softplus_warp(x) + EPS,
    'noise_variance': lambda x: softplus_warp(x) + EPS,
    'dot_prod_sigma': lambda x: softplus_warp(x) + EPS,
}

single_gp_default_inverse_warp_func = {
    'constant': identity_warp,
    'lengthscale': lambda x: softplus_inverse_warp(x - EPS),
    'signal_variance': lambda x: softplus_inverse_warp(x - EPS),
    'noise_variance': lambda x: softplus_inverse_warp(x - EPS),
    'dot_prod_sigma': lambda x: softplus_inverse_warp(x - EPS),
}

normal_params_warp = lambda x: (x[0], softplus_warp(x[1]) + EPS)
lognormal_params_warp = normal_params_warp
gamma_params_warp = lambda x: (softplus_warp(x[0]) + EPS, softplus_warp(x[1]) + EPS)

normal_params_inverse_warp = lambda x: (x[0], softplus_inverse_warp(x[1] - EPS))
lognormal_params_inverse_warp = normal_params_inverse_warp
gamma_params_inverse_warp = lambda x: (softplus_inverse_warp(x[0] - EPS), softplus_inverse_warp(x[1] - EPS))


def search_space_params_warp(search_space_params, single_gp_warp_func):
    warpped_search_space_params = {}
    for key, value in search_space_params.items():
        warpped_search_space_params[key] = apply_warp_func(value, single_gp_warp_func)
    return warpped_search_space_params


def search_space_params_inverse_warp(search_space_params, single_gp_inverse_warp_func):
    warpped_search_space_params = {}
    for key, value in search_space_params.items():
        warpped_search_space_params[key] = apply_warp_func(value, single_gp_inverse_warp_func)
    return warpped_search_space_params


def get_e2e_v3_warp_func(distribution_type, single_gp_warp_func, single_gp_inverse_warp_func):
    if distribution_type == 'lognormal':
        dist_params_warp = lognormal_params_warp
        dist_params_inverse_warp = lognormal_params_inverse_warp
    elif distribution_type == 'gamma':
        dist_params_warp = gamma_params_warp
        dist_params_inverse_warp = gamma_params_inverse_warp
    else:
        raise ValueError('distribution_type must be lognormal or gamma.')

    e2e_v3_warp_func = {
        'constant_normal_params': normal_params_warp,
        'signal_variance_{}_params'.format(distribution_type): dist_params_warp,
        'noise_variance_{}_params'.format(distribution_type): dist_params_warp,
        'search_space_params': partial(search_space_params_warp, single_gp_warp_func=single_gp_warp_func),
    }

    e2e_v3_inverse_warp_func = {
        'constant_normal_params': normal_params_inverse_warp,
        'signal_variance_{}_params'.format(distribution_type): dist_params_inverse_warp,
        'noise_variance_{}_params'.format(distribution_type): dist_params_inverse_warp,
        'search_space_params': partial(search_space_params_inverse_warp,
                                       single_gp_inverse_warp_func=single_gp_inverse_warp_func),
    }

    return e2e_v3_warp_func, e2e_v3_inverse_warp_func


def kl_multivariate_normal(mu0,
                           cov0,
                           mu1,
                           cov1,
                           weight=1.,
                           partial=True,
                           feat0=None,
                           eps=0.):
  """Computes KL divergence between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
      cov1 must be invertible.
    weight: weight for the returned KL divergence.
    partial: only compute terms in KL involving mu1 and cov1 if True.
    feat0: (optional) feature used to compute cov0 if cov0 = feat0 * feat0.T /
      feat0.shape[1]. For a low-rank cov0, we may have to compute the KL
      divergence for a degenerate multivariate normal.
    eps: (optional) small positive value added to the diagonal terms of cov0 and
      cov1 to make them well behaved.

  Returns:
    KL divergence. The returned value does not include terms that are not
    affected by potential model parameters in mu1 or cov1.
  """
  if not cov0.shape:
    cov0 = cov0[jnp.newaxis, jnp.newaxis]
  if not cov1.shape:
    cov1 = cov1[jnp.newaxis, jnp.newaxis]

  if eps > 0.:
    cov0 = cov0 + jnp.eye(cov0.shape[0]) * eps
    cov1 = cov1 + jnp.eye(cov1.shape[0]) * eps

  mu_diff = mu1 - mu0
  chol1, cov1invmudiff = linalg.solve_linear_system(cov1, mu_diff)
  # pylint: disable=g-long-lambda
  func = lambda x: linalg.inverse_spdmatrix_vector_product(
      cov1, x, cached_cholesky=chol1)
  trcov1invcov0 = jnp.trace(vmap(func)(cov0))
  mahalanobis = jnp.dot(mu_diff, cov1invmudiff)
  logdetcov1 = jnp.sum(2 * jnp.log(jnp.diag(chol1)))
  common_terms = trcov1invcov0 + mahalanobis + logdetcov1
  if partial:
    return 0.5 * weight * common_terms
  else:
    if feat0 is not None and feat0.shape[0] > feat0.shape[1]:
      logging.info('Using pseudo determinant of cov0.')
      sign, logdetcov0 = jnp.linalg.slogdet(
          jnp.divide(jnp.dot(feat0.T, feat0), feat0.shape[1]))
      logging.info(msg=f'Pseudo logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Pseudo determinant of cov0 is 0 or negative.'

      # cov0inv is computed for more accurate pseudo KL. feat0 may be low rank.
      cov0inv = jnp.linalg.pinv(cov0)
      return 0.5 * weight * (
          common_terms - logdetcov0 -
          jnp.linalg.matrix_rank(jnp.dot(cov0inv, cov0)) + jnp.log(2 * jnp.pi) *
          (cov1.shape[0] - feat0.shape[1]))
    else:
      sign, logdetcov0 = jnp.linalg.slogdet(cov0)
      logging.info(msg=f'sign = {sign}; logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Determinant of cov0 is 0 or negative.'
      return 0.5 * weight * (common_terms - logdetcov0 - cov0.shape[0])


def euclidean_multivariate_normal(mu0,
                                  cov0,
                                  mu1,
                                  cov1,
                                  mean_weight=1.,
                                  cov_weight=1.,
                                  **unused_kwargs):
  """Computes Euclidean distance between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
    mean_weight: weight for euclidean distance on the mean vectors.
    cov_weight: weight for euclidean distance on the covariance matrices.

  Returns:
    Reweighted Euclidean distance between two multivariate normal distributions.
  """
  mean_diff = linalg.safe_l2norm(mu0 - mu1)
  cov_diff = linalg.safe_l2norm((cov0 - cov1).flatten())
  return mean_weight * mean_diff + cov_weight * cov_diff


def normal_param_from_samples(samples):
    fit_norm = Normal.experimental_fit(samples)
    fit_mu = fit_norm.loc
    fit_sigma = fit_norm.scale
    return fit_mu, fit_sigma


def gamma_param_from_samples(samples):
    fit_gamma = Gamma.experimental_fit(samples)
    fit_a = fit_gamma.concentration
    fit_b = fit_gamma.rate
    return fit_a, fit_b


def lognormal_param_from_samples(samples):
    fit_lognorm = LogNormal.experimental_fit(samples)
    fit_mu = fit_lognorm.loc
    fit_sigma = fit_lognorm.scale
    return fit_mu, fit_sigma
