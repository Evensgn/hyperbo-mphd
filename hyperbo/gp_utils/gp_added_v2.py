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

"""Inferrence and other util functions for a (multi-task) GP."""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import flax
from flax import linen as nn
from hyperbo.basics import bfgs
from hyperbo.basics import data_utils
from hyperbo.basics import definitions as defs
from hyperbo.basics import lbfgs
from hyperbo.basics import linalg
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
import jax
from jax import flatten_util
import jax.numpy as jnp
import jax.random
import jax.scipy as jsp
import optax
import os
from functools import partial


grad = jax.grad
jit = jax.jit
vmap = jax.vmap

retrieve_params = params_utils.retrieve_params

GPCache = defs.GPCache
SubDataset = defs.SubDataset
GPParams = defs.GPParams


def infer_parameters(mean_func,
                     cov_func,
                     init_params,
                     super_dataset,
                     dim_feature_values,
                     warp_func=None,
                     objective=obj.neg_log_marginal_likelihood_hgp,
                     key=None,
                     params_save_file=None):
  """Posterior inference for a meta GP.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    init_params: GPParams, initial parameters for covariance, mean and noise
      variance, together with config parameters including 'method', a str
      indicating which method should be used. Currently it supports 'bfgs' and
      'momentum'.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    objective: objective loss function to minimize. Curently support
      neg_log_marginal_likelihood or sample_mean_cov_regularizer or linear
      combinations of them.
    key: Jax random state.
    params_save_file: optional file name to save params.

  Returns:
    Dictionary of inferred parameters.
  """
  if key is None:
    key = jax.random.PRNGKey(0)
    logging.info('Using default random state in infer_parameters.')
  if not super_dataset:
    logging.info('No dataset present to train GP.')
    return init_params
  params = init_params
  method = params.config['method']
  batch_size = params.config['batch_size']

  maxiter = init_params.config['maxiter']
  logging_interval = init_params.config['logging_interval']

  if maxiter <= 0 and method != 'slice_sample':
    return init_params

  if method == 'adam':
    @jit
    def loss_func(model_params, batch, key=None):
      return objective(
          key=key,
          mean_func=mean_func,
          cov_func=cov_func,
          params=GPParams(model=model_params, config=init_params.config),
          super_dataset=batch,
          dim_feature_values=dim_feature_values,
          warp_func=warp_func)

    optimizer = optax.adam(params.config['learning_rate'])
    opt_state = optimizer.init(params.model)

    key, new_key = jax.random.split(key, 2)
    dataset_iter = data_utils.sub_sample_super_dataset_iterator(new_key, super_dataset, batch_size)
    model_param = params.model
    for i in range(maxiter):
      batch = next(dataset_iter)
      key, new_key = jax.random.split(key)
      current_loss, grads = jax.value_and_grad(partial(loss_func, key=new_key))(model_param, batch)
      if jnp.isfinite(current_loss):
        params.model = model_param
      else:
        logging.info(msg=f'{method} stopped due to instability.')
        break
      updates, opt_state = optimizer.update(grads, opt_state)
      model_param = optax.apply_updates(model_param, updates)
      if i % logging_interval == 0:
          keys = list(params.model.keys())
          retrieved_params = dict(
              zip(keys, retrieve_params(params, keys, warp_func=warp_func)))
          print('step: {}, loss: {}, params: {}'.format(i, current_loss, retrieved_params))
          jnp.save(params_save_file, retrieved_params)
    key, new_key = jax.random.split(key)
    current_loss = loss_func(model_param, batch, key=new_key)
    if jnp.isfinite(current_loss):
      params.model = model_param
    params_utils.log_params_loss(
        step=maxiter,
        params=params,
        loss=current_loss,
        warp_func=warp_func,
        params_save_file=params_save_file)
  return params


class HGP_E2E_v2:
  """A Gaussian process that supports learning with historical data.

  Attributes:
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: GPParams, parameters for covariance, mean, and noise variance.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    input_dim: dimension of input variables.
    rng: Jax random state.
  """
  dataset: Dict[Union[int, str], SubDataset]

  def __init__(self,
               super_dataset,
               dim_feature_values,
               mean_func: Callable[..., jnp.array],
               cov_func: Callable[..., jnp.array],
               params: GPParams,
               warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None):
    self.mean_func = mean_func
    self.cov_func = cov_func
    if params is not None:
      self.params = params
    else:
      self.params = GPParams()
    self.warp_func = warp_func
    self.set_super_dataset(super_dataset)
    self.dim_feature_values = dim_feature_values
    if 'objective' not in self.params.config:
      self.params.config['objective'] = obj.neg_log_marginal_likelihood
    self.rng = None

  def initialize_params(self, key):
    """Initialize params with a JAX random state."""
    if not self.super_dataset:
      raise ValueError('Cannot initialize GPParams without super-dataset.')
    logging.info(msg=f'super-dataset: {jax.tree_map(jnp.shape, self.super_dataset)}')

    if isinstance(self.params.config['objective'], str):
      self.params.config['objective'] = getattr(obj,
                                                self.params.config['objective'])
    self.rng = key

  def set_super_dataset(self, super_dataset):
    """Reset GP dataset to be dataset.

    Args:
      dataset: a list of vx, vy pairs, i.e. [(vx, vy)_i], where vx is n x d and
        vy is n x 1.
    """
    self.super_dataset = {}
    self.params.cache = {}
    for dataset_key, dataset in super_dataset.items():
        if isinstance(dataset, list):
            dataset = {i: dataset[i] for i in range(len(dataset))}
        for sub_dataset_key, sub_dataset in dataset.items():
            dataset[sub_dataset_key] = SubDataset(*sub_dataset)
        self.super_dataset[dataset_key] = dataset

  def update_sub_dataset(self,
                         sub_dataset: Union[Tuple[jnp.ndarray, ...],
                                            SubDataset],
                         dataset_key: Union[int, str] = 0,
                         sub_dataset_key: Union[int, str] = 0,
                         is_append: bool = False):
    """Update a sub-dataset in dataset.

    Args:
      sub_dataset: the new sub-dataset as in (vx, vy) pair to be updated to
        dataset.
      sub_dataset_key: the key of the sub-dataset in dataset.
      is_append: append to the sub-dataset if True; otherwise replace it.
    """
    sub_dataset = SubDataset(*sub_dataset)

    if is_append:
      new_x = jnp.vstack((self.super_dataset[dataset_key][sub_dataset_key].x, sub_dataset.x))
      new_y = jnp.vstack((self.super_dataset[dataset_key][sub_dataset_key].y, sub_dataset.y))
      self.super_dataset[dataset_key][sub_dataset_key] = SubDataset(x=new_x, y=new_y)
    else:
      self.super_dataset[dataset_key][sub_dataset_key] = sub_dataset
    if dataset_key in self.params.cache:
      self.params.cache[dataset_key].needs_update = True

  def train(self, key=None, params_save_file=None) -> GPParams:
    """Train the GP by fitting it to the dataset.

    Args:
      key: Jax random state.
      params_save_file: optional file name to save params.

    Returns:
      params: GPParams.
    """
    if key is None:
      if self.rng is None:
        self.rng = jax.random.PRNGKey(0)
        logging.info('Using default random state in GP.train.')
      key, subkey = jax.random.split(self.rng, 2)
      self.rng = key
    else:
      key, subkey = jax.random.split(key, 2)
    self.params = infer_parameters(
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        init_params=self.params,
        super_dataset=self.super_dataset,
        dim_feature_values=self.dim_feature_values,
        warp_func=self.warp_func,
        objective=self.params.config['objective'],
        key=subkey,
        params_save_file=params_save_file)
    logging.info(msg=f'params = {self.params}')
    return self.params

  def neg_log_marginal_likelihood_hgp(self) -> float:
    """Compute negative log marginal likelihood for current model."""
    return obj.neg_log_marginal_likelihood_hgp_v2(
        key=self.rng,
        mean_func=self.mean_func,
        cov_func=self.cov_func,
        params=self.params,
        super_dataset=self.super_dataset,
        dim_feature_values=self.dim_feature_values,
        warp_func=self.warp_func,
        n_gamma_samples=self.params.config['n_gamma_samples'],
    )

  def update_model_params(self, model_params: Dict[str, Any]):
    """Update params.model (must clean up params.cache)."""
    self.params.model = model_params
    self.params.cache = {}
