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

"""Bayesian optimization (BO) loop for sequential queries."""
import dataclasses
import time
from typing import Callable, Tuple, Union, Any

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from absl import logging

from hyperbo.basics import definitions as defs
from hyperbo.bo_utils import const
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import priors

# import botorch
# from botorch.models.gp_regression import SingleTaskGP
# from botorch.fit import fit_gpytorch_model
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.optim import optimize_acqf_discrete
# import torch
# from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, ProbabilityOfImprovement


SubDataset = defs.SubDataset
INPUT_SAMPLERS = const.INPUT_SAMPLERS


def bayesopt(key: Any, model: gp.GP, sub_dataset_key: Union[int, str],
             query_oracle: Callable[[Any], Any], ac_func: Callable[...,
                                                                   jnp.array],
             iters: int, input_sampler: Callable[..., jnp.array]) -> SubDataset:
    """Running simulated bayesopt on a set of pre-evaluated inputs x_queries.

    Args:
      key: Jax random state.
      model: gp.GP.
      sub_dataset_key: key of the sub_dataset for testing in dataset.
      query_oracle: evaluation function.
      ac_func: acquisition function handle (see acfun.py).
      iters: number of iterations in BayesOpt sequential queries.
      input_sampler: function for sampling inputs among which the initial point is
        chosen for acquisition function optimization.

    Returns:
      All observations after bayesopt in the form of an (x_observed, y_observed)
      tuple. These observations include those made before bayesopt.
    """
    input_dim = model.input_dim
    for i in range(iters):
        start_time = time.time()
        x_samples = input_sampler(key, input_dim)
        evals = ac_func(
            model=model,
            sub_dataset_key=sub_dataset_key,
            x_queries=x_samples)
        select_idx = evals.argmax()
        x_init = x_samples[select_idx]

        def f(x):
            return -ac_func(
                model=model,
                sub_dataset_key=sub_dataset_key,
                x_queries=jnp.array([x])).flatten()[0]

        opt = jaxopt.ScipyBoundedMinimize(method='L-BFGS-B', fun=f)
        opt_ret = opt.run(
            x_init, bounds=[jnp.zeros(input_dim),
                            jnp.ones(input_dim)])
        eval_datapoint = opt_ret.params, query_oracle(opt_ret.params[None, :])
        logging.info(msg=f'{i}-th iter, x_init={x_init}, '
                         f'eval_datapoint={eval_datapoint}, '
                         f'elpased_time={time.time() - start_time}')
        model.update_sub_dataset(
            eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)
        if 'retrain' in model.params.config and model.params.config['retrain'] > 0:
            if model.params.config['objective'] in [obj.regkl, obj.regeuc]:
                raise ValueError('Objective must include NLL to retrain.')
            else:
                maxiter = model.params.config['retrain']
                logging.info(msg=f'Retraining with maxiter = {maxiter}.')
                model.params.config['maxiter'] = maxiter
                model.train()

    return model.dataset.get(sub_dataset_key,
                             SubDataset(jnp.empty(0), jnp.empty(0)))


def simulated_bayesopt(model: gp.GP, sub_dataset_key: Union[int, str],
                       queried_sub_dataset: SubDataset,
                       ac_func: Callable[...,
                                         jnp.array], iters: int) -> SubDataset:
    """Running simulated bayesopt on a set of pre-evaluated inputs x_queries.

    Args:
      model: gp.GP.
      sub_dataset_key: key of the sub_dataset for testing in dataset.
      queried_sub_dataset: sub_dataset that can be queried.
      ac_func: acquisition function handle (see acfun.py).
      iters: number of iterations in BayesOpt sequential queries.

    Returns:
      All observations after bayesopt in the form of an (x_observed, y_observed)
      tuple. These observations include those made before bayesopt.
    """
    i = 0
    for _ in range(iters):
        # retrain before the initial BO step as well
        if 'retrain' in model.params.config and model.params.config['retrain'] > 0:
            if model.params.config['objective'] in [obj.regkl, obj.regeuc]:
                raise ValueError('Objective must include NLL to retrain.')
            else:
                maxiter = model.params.config['retrain']
                logging.info(msg=f'Retraining with maxiter = {maxiter}.')
                model.params.config['maxiter'] = maxiter
                model.train()

        i += 1
        evals = ac_func(
            model=model,
            sub_dataset_key=sub_dataset_key,
            x_queries=queried_sub_dataset.x)
        select_idx = evals.argmax()
        eval_datapoint = queried_sub_dataset.x[select_idx], queried_sub_dataset.y[
            select_idx]
        model.update_sub_dataset(
            eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)

    return model.dataset.get(sub_dataset_key,
                             SubDataset(jnp.empty(0), jnp.empty(0)))


def simulated_bayesopt_with_gp_params_samples(
        n_dim,
        gp_params_samples,
        cov_func,
        mean_func,
        dataset,
        sub_dataset_key,
        queried_sub_dataset,
        ac_func,
        iters,
        n_bo_gp_params_samples,
        x_batch_size=None):
    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=defs.GPParams(model={}),
        warp_func=None)

    constants, lengthscales, signal_variances, noise_variances = gp_params_samples

    params_sample_list = []
    for j in range(n_bo_gp_params_samples):
        constant_j = constants[j:j + 1]
        lengthscale_j = lengthscales[j]
        signal_variance_j = signal_variances[j:j + 1]
        noise_variance_j = noise_variances[j:j + 1]
        params_sample_list.append(jnp.concatenate((constant_j, lengthscale_j, signal_variance_j, noise_variance_j)))
    params_sample_list = jnp.array(params_sample_list)

    true_len = queried_sub_dataset.x.shape[0]

    # temp
    x_batch_size = None

    if x_batch_size is None:
        print('true_len', true_len)
        x_batch_size = true_len
    if true_len % x_batch_size != 0:
        padded_len = true_len + x_batch_size - true_len % x_batch_size
        gap_len = padded_len - true_len
        x_queries_padded = jnp.concatenate((queried_sub_dataset.x, jnp.zeros((gap_len, n_dim))))
    else:
        padded_len = true_len
        x_queries_padded = queried_sub_dataset.x

    # without jit inside loop: 256s
    # with jit inside loop:

    def hgp_acfun_batch(x_queries_batch):
        evals = ac_func(
            model=model,
            sub_dataset_key=sub_dataset_key,
            x_queries=x_queries_batch)
        return evals

    def hgp_acfun(params_sample):
        params_sample = defs.GPParams(
            model={
                'constant': params_sample[0],
                'lengthscale': params_sample[1:n_dim + 1],
                'signal_variance': params_sample[n_dim + 1],
                'noise_variance': params_sample[n_dim + 2]
            }
        )
        # set params
        model.params = params_sample
        x_queries_batches = []
        for i in range(padded_len // x_batch_size):
            x_queries_batches.append(x_queries_padded[i * x_batch_size:(i + 1) * x_batch_size])
        batch_evals = jax.vmap(hgp_acfun_batch)(jnp.array(x_queries_batches))
        evals = jnp.concatenate(batch_evals, axis=0)

        # P(dataset | theta)
        p_dataset_theta = jnp.exp(
            -obj.neg_log_marginal_likelihood(
                mean_func=model.mean_func,
                cov_func=model.cov_func,
                params=params_sample,
                dataset=model.dataset,
                warp_func=None,
            )
        )
        evals *= p_dataset_theta

        return evals

    for i in range(iters):
        evals = jax.vmap(hgp_acfun)(params_sample_list)
        evals = jnp.sum(jnp.stack(evals), axis=0)

        # print('evals: ', evals[:10])
        assert evals.shape[0] == padded_len

        select_idx = evals[:true_len].argmax()

        eval_datapoint = queried_sub_dataset.x[select_idx], queried_sub_dataset.y[select_idx]
        model.update_sub_dataset(
            eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)

    return model.dataset.get(sub_dataset_key, SubDataset(jnp.empty(0), jnp.empty(0)))


def run_synthetic(dataset,
                  sub_dataset_key,
                  queried_sub_dataset,
                  mean_func,
                  cov_func,
                  init_params,
                  ac_func,
                  iters,
                  warp_func=None,
                  init_random_key=None,
                  method='hyperbo',
                  init_model=False,
                  finite_search_space=True,
                  data_loader_name='',
                  params_save_file=None):
    """Running bayesopt experiment with synthetic data.

    Args:
      dataset: a list of vx, vy pairs, i.e. [(vx, vy)_i], where vx is
        m_points_historical x d and vy is m_points_historical x 1.
      sub_dataset_key: key of the sub_dataset for testing in dataset.
      queried_sub_dataset: sub_dataset that can be queried.
      mean_func: mean function handle that maps from (params, n x d input,
        warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
        more details).
      cov_func: covariance function handle that maps from (params, n1 x d input1,
        n2 x d input2, wrap_func) to a n1 x n2 covariance matrix (see matrix_map
        in kernel.py for more details).
      init_params: initial GP parameters for inference.
      ac_func: acquisition function handle (see acfun.py).
      iters: Number of iterations in sequential bayesopt queries.
      warp_func: optional dictionary that specifies the warping function for each
        parameter.
      init_random_key: random state for jax.random, to be used to initialize
        required parts of GPParams.
      method: BO method.
      init_model: to initialize model if True; otherwise False.
      finite_search_space: use a finite search space if True; otherwise False.
      data_loader_name: data loader name, e.g. pd1, hpob.
      params_save_file: optional file name to save params.

    Returns:
      All observations in (x, y) pairs returned by the bayesopt strategy and all
      the query points in (x, y) pairs. Model params as a dict.
    """
    # print('run_synthetic start')
    logging.info(msg=f'run_synthetic is using method {method}.')
    if method in const.USE_HGP:
        model_class = gp.HGP
        init_params.config.update({
            'objective': 'nll',
            'method': 'slice_sample',
            'burnin': 50,
            'nsamples': 50,
            'priors': priors.DEFAULT_PRIORS,
        })
    else:
        model_class = gp.GP

    model = model_class(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)
    key = init_random_key

    # print('init_model')
    if init_model:
        key, subkey = jax.random.split(key)
        model.initialize_params(subkey)
        # Infer GP parameters.
        key, subkey = jax.random.split(key)
        model.train(subkey, params_save_file)
    if finite_search_space:
        # print('simulated_bayesopt start')
        sub_dataset = simulated_bayesopt(
            model=model,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            iters=iters)
        # print('run_synthetic end')
        return (sub_dataset.x,
                sub_dataset.y), (queried_sub_dataset.x,
                                 queried_sub_dataset.y), model.params.__dict__
    else:
        if data_loader_name not in INPUT_SAMPLERS:
            raise NotImplementedError(
                f'Input sampler for {data_loader_name} not found.')
        _, sample_key = jax.random.split(key)
        sub_dataset = bayesopt(
            key=sample_key,
            model=model,
            sub_dataset_key=sub_dataset_key,
            query_oracle=queried_sub_dataset,
            ac_func=ac_func,
            iters=iters,
            input_sampler=INPUT_SAMPLERS[data_loader_name])
        return (sub_dataset.x,
                sub_dataset.y), None, model.params.__dict__


def _onehot_matrix(shape, idx) -> np.ndarray:
    """Each row is a one-hot vector with idx-th element equal to 1."""
    zeros = np.zeros(shape)
    zeros[:, idx] = 1
    return zeros


def _subdataset_to_arrays(ds: SubDataset, dataset_id: int,
                          num_datasets: int) -> Tuple[np.ndarray, np.ndarray]:
    onehot = _onehot_matrix((ds.y.shape[0], num_datasets), dataset_id)
    return np.concatenate([ds.x, onehot], axis=1), ds.y


@dataclasses.dataclass
class _XYPair:
    """Helper class to keep x,y pair in sync."""
    x: np.ndarray
    y: np.ndarray

    def append_xy(self, other, idx: int) -> None:
        self.x = np.concatenate([self.x, other.x[idx:idx + 1, :]], axis=0)
        self.y = np.concatenate([self.y, other.y[idx:idx + 1, :]], axis=0)

    def delete(self, idx: int) -> None:
        self.x = np.delete(self.x, idx, 0)
        self.y = np.delete(self.y, idx, 0)

    def concat(self, other) -> '_XYPair':
        return _XYPair(
            x=np.concatenate([self.x, other.x]),
            y=np.concatenate([self.y, other.y]))

    def empty_like(self) -> '_XYPair':
        return _XYPair(
            x=np.zeros(0, self.x.shape[1]), y=np.zeros(0, self.y.shape[1]))

    @property
    def size(self):
        return self.x.shape[0]
