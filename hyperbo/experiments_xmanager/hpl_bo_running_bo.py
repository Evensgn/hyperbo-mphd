import gc
import os

import jax
import jax.numpy as jnp
import numpy as np

import hyperbo.bo_utils.acfun as acfun
from hpl_bo_utils import get_gp_priors_from_direct_hgp, get_gp_priors_from_hpl_hgp, get_gp_priors_from_gt_hgp
from hyperbo.basics import definitions as defs
from hyperbo.bo_utils import bayesopt
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import utils
from experiment_defs import *


def run_bo_with_gp(cov_func, mean_func, gp_params, dataset, sub_dataset_key, queried_sub_dataset, ac_func, budget):
    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=gp_params,
        warp_func=None,
    )

    assert 'retrain' not in model.params.config

    sub_dataset = bayesopt.simulated_bayesopt(
        model=model,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        ac_func=ac_func,
        iters=budget,
    )
    return {
        'observations': (sub_dataset.x, sub_dataset.y),
    }


def run_bo_with_gp_retrain(cov_func, mean_func, gp_params, gp_objective, gp_retrain_maxiter, gp_retrain_method, dataset,
                           sub_dataset_key, queried_sub_dataset, ac_func, budget):
    n_init_obs = dataset['history'].x.shape[0]
    single_gp_warp_func = utils.single_gp_default_warp_func
    single_gp_inverse_warp_func = utils.single_gp_default_inverse_warp_func

    gp_params.model = utils.apply_warp_func(gp_params.model, single_gp_inverse_warp_func)
    gp_params.config.update({
        'method': gp_retrain_method,
        'retrain': gp_retrain_maxiter,
        'logging_interval': 10,
        'objective': gp_objective,
        'batch_size': budget + n_init_obs,
    })

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=gp_params,
        warp_func=single_gp_warp_func,
    )

    sub_dataset = bayesopt.simulated_bayesopt(
        model=model,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        ac_func=ac_func,
        iters=budget,
    )
    return {
        'observations': (sub_dataset.x, sub_dataset.y),
    }


def run_bo_with_gp_params_samples(cov_func, mean_func, n_dim, gp_params_samples, n_bo_gp_params_samples, dataset,
                                  sub_dataset_key, queried_sub_dataset, ac_func, budget, x_batch_size):
    sub_dataset = bayesopt.simulated_bayesopt_with_gp_params_samples(
        n_dim=n_dim,
        gp_params_samples=gp_params_samples,
        cov_func=cov_func,
        mean_func=mean_func,
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        ac_func=ac_func,
        iters=budget,
        n_bo_gp_params_samples=n_bo_gp_params_samples,
        x_batch_size=x_batch_size,
    )
    return {
        'observations': (sub_dataset.x, sub_dataset.y),
    }


def run_bo_with_gp_params_prior(cov_func, mean_func, n_dim, gp_priors, gp_objective, gp_retrain_maxiter,
                                gp_retrain_method, dataset, sub_dataset_key, queried_sub_dataset, ac_func, budget):
    n_init_obs = dataset['history'].x.shape[0]
    gp_params = defs.GPParams(
        model={
            'constant': 1.0,
            'lengthscale': jnp.array([1.0] * n_dim),
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method': gp_retrain_method,
            'retrain': gp_retrain_maxiter,
            'logging_interval': 10,
            'objective': gp_objective,
            'batch_size': budget + n_init_obs,
            'priors': gp_priors,
        })

    single_gp_warp_func = utils.single_gp_default_warp_func
    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=gp_params,
        warp_func=single_gp_warp_func,
    )

    sub_dataset = bayesopt.simulated_bayesopt(
        model=model,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        ac_func=ac_func,
        iters=budget,
    )

    final_map_gp_params = utils.apply_warp_func(model.params.model, single_gp_warp_func)

    return {
        'observations': (sub_dataset.x, sub_dataset.y),
        'final_map_gp_params': final_map_gp_params,
    }


def run_bo(run_args):
    (cov_func, mean_func, n_dim, gp_params, gp_priors, gp_objective, gp_retrain_maxiter, gp_retrain_method,
     queried_sub_dataset, init_indices, ac_func, budget, method_name) = run_args

    sub_dataset_key = 'history'
    if init_indices is None:
        init_sub_dataset = defs.SubDataset(x=jnp.empty((0, n_dim)), y=jnp.empty((0, 1)))
    else:
        history_x = queried_sub_dataset.x[init_indices, :]
        history_y = queried_sub_dataset.y[init_indices, :]
        init_sub_dataset = defs.SubDataset(x=history_x, y=history_y)
    dataset = {sub_dataset_key: init_sub_dataset}

    results = {}

    if method_name in ['random', 'base', 'hyperbo', 'ablr', 'fsbo', 'gt_gp']:
        '''
        bo_results = run_bo_with_gp_retrain(
            cov_func=cov_func,
            mean_func=mean_func,
            gp_params=gp_params,
            gp_objective=gp_objective,
            gp_retrain_maxiter=gp_retrain_maxiter,
            gp_retrain_method=gp_retrain_method,
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            budget=budget,
        )
        '''
        bo_results = run_bo_with_gp(
            cov_func=cov_func,
            mean_func=mean_func,
            gp_params=gp_params,
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            budget=budget,
        )
    else:
        bo_results = run_bo_with_gp_params_prior(
            cov_func=cov_func,
            mean_func=mean_func,
            n_dim=n_dim,
            gp_priors=gp_priors,
            gp_objective=gp_objective,
            gp_retrain_maxiter=gp_retrain_maxiter,
            gp_retrain_method=gp_retrain_method,
            dataset=dataset,
            sub_dataset_key=sub_dataset_key,
            queried_sub_dataset=queried_sub_dataset,
            ac_func=ac_func,
            budget=budget,
        )
        results['final_map_gp_params'] = bo_results['final_map_gp_params']

    observations = bo_results['observations']
    results['observations'] = observations

    max_f = jnp.max(queried_sub_dataset.y)
    min_f = jnp.min(queried_sub_dataset.y)
    regrets = []
    max_y = -jnp.inf
    for y in observations[1]:
        if y[0] > max_y:
            max_y = y[0]
        regrets.append((max_f - max_y) / (max_f - min_f))

    results['regrets'] = regrets

    return results


def test_bo(dir_path, key, test_id, pool, dataset, base_cov_func, base_mean_func, n_init_obs, init_indices_values,
            budget, n_bo_runs, gp_objective, gp_retrain_maxiter, gp_retrain_method, ac_func, hand_hgp_params,
            uniform_hgp_params, log_uniform_hgp_params, gt_hgp_params, gt_gp_params, distribution_type,
            dim_feature_value, method_name, extra_suffix):
    n_dim = list(dataset.values())[0].x.shape[1]

    # sample init_indices
    init_indices_map = {}
    for sub_dataset_key, sub_dataset in dataset.items():
        init_indices_map[sub_dataset_key] = {}
        for i in range(n_bo_runs):
            if n_init_obs == 0:
                init_indices_map[sub_dataset_key][i] = None
            else:
                if init_indices_values and 'test{}'.format(i) in init_indices_values[sub_dataset_key] and \
                        len(init_indices_values[sub_dataset_key]['test{}'.format(i)]) == n_init_obs:
                    init_indices_map[sub_dataset_key][i] = init_indices_values[sub_dataset_key]['test{}'.format(i)]
                else:
                    new_key, key = jax.random.split(key)
                    init_indices_map[sub_dataset_key][i] = jax.random.choice(
                        new_key, sub_dataset.x.shape[0], shape=(n_init_obs,), replace=False)

    # construct task list
    task_list = []
    pass_ac_func = ac_func
    if method_name in ['random', 'base', 'hyperbo', 'ablr', 'fsbo', 'gt_gp']:
        gp_priors = None
        if method_name == 'random':
            placeholder_params = defs.GPParams(
                model={
                    'constant': 1.0,
                    'lengthscale': jnp.array([1.0] * n_dim),
                    'signal_variance': 1.0,
                    'noise_variance': 1e-6,
                }
            )
            pass_ac_func = acfun.rand
            gp_params = placeholder_params
            cov_func = base_cov_func
            mean_func = base_mean_func
        elif method_name == 'base':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(test_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            cov_func = base_cov_func
            mean_func = base_mean_func
        elif method_name == 'hyperbo':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_hyperbo.npy'.format(test_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = HYPERBO_MLP_FEATURES
            cov_func = HYPERBO_KERNEL_TYPE
            mean_func = HYPERBO_MEAN_TYPE
        elif method_name == 'ablr':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_ablr.npy'.format(test_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = ABLR_MLP_FEATURES
            cov_func = ABLR_KERNEL_TYPE
            mean_func = ABLR_MEAN_TYPE
        elif method_name == 'fsbo':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_fsbo.npy'.format(test_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = FSBO_MLP_FEATURES
            cov_func = FSBO_KERNEL_TYPE
            mean_func = FSBO_MEAN_TYPE
        elif method_name == 'gt_gp':
            gp_params = gt_gp_params[test_id]
            cov_func = base_cov_func
            mean_func = base_mean_func
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))
    elif method_name == 'gt_hgp':
        gp_params = None
        cov_func = base_cov_func
        mean_func = base_mean_func
        gp_priors = get_gp_priors_from_gt_hgp(gt_hgp_params, distribution_type, n_dim)
    elif method_name in ['hand_hgp', 'uniform_hgp', 'log_uniform_hgp', 'fit_direct_hgp', 'fit_direct_hgp_leaveout']:
        pass_distribution_type = distribution_type
        gp_params = None
        cov_func = base_cov_func
        mean_func = base_mean_func
        if method_name == 'hand_hgp':
            direct_hgp_params = hand_hgp_params
        elif method_name == 'uniform_hgp':
            pass_distribution_type = 'all_uniform'
            direct_hgp_params = uniform_hgp_params
        elif method_name == 'log_uniform_hgp':
            pass_distribution_type = 'log_uniform'
            direct_hgp_params = log_uniform_hgp_params
        elif method_name == 'fit_direct_hgp':
            direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b{}.npy'.format(
                extra_suffix)), allow_pickle=True).item()['gp_distribution_params']
        elif method_name == 'fit_direct_hgp_leaveout':
            direct_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}{}.npy'.format(test_id,
                                                                                                        extra_suffix)),
                allow_pickle=True).item()['gp_distribution_params']
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))
        gp_priors = get_gp_priors_from_direct_hgp(direct_hgp_params, pass_distribution_type, n_dim)
    elif method_name in ['hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_leaveout', 'hpl_hgp_end_to_end_from_scratch',
                         'hpl_hgp_end_to_end_leaveout_from_scratch', 'hpl_hgp_two_step',
                         'hpl_hgp_two_step_leaveout']:
        gp_params = None
        cov_func = base_cov_func
        mean_func = base_mean_func
        if method_name == 'hpl_hgp_end_to_end':
            hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b{}.npy'.format(
                extra_suffix)), allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_leaveout':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}{}.npy'.format(test_id,
                                                                                                       extra_suffix)),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_from_scratch':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch{}.npy'.format(extra_suffix)),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_leaveout_from_scratch':
            hpl_hgp_params = np.load(
                os.path.join(dir_path,
                             'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch{}.npy'.format(test_id,
                                                                                                          extra_suffix)),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_two_step':
            hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b{}.npy'.format(
                extra_suffix)), allow_pickle=True).item()
        elif method_name == 'hpl_hgp_two_step_leaveout':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}{}.npy'.format(test_id,
                                                                                                     extra_suffix)),
                allow_pickle=True).item()
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))
        gp_priors = get_gp_priors_from_hpl_hgp(hpl_hgp_params, distribution_type, n_dim, dim_feature_value)
    else:
        raise ValueError('Unknown method name: {}'.format(method_name))

    for i in range(n_bo_runs):
        for sub_dataset_key, sub_dataset in dataset.items():
            init_indices = init_indices_map[sub_dataset_key][i]
            task_list.append((cov_func, mean_func, n_dim, gp_params, gp_priors, gp_objective, gp_retrain_maxiter,
                              gp_retrain_method, sub_dataset, init_indices, pass_ac_func, budget, method_name))
    print('task_list constructed, number of tasks: {}'.format(len(task_list)))

    if pool is not None:
        print('using pool')
        task_outputs = pool.map(run_bo, task_list)
    else:
        task_outputs = []
        for i, task in enumerate(task_list):
            print('task number {}'.format(i))
            task_outputs.append(run_bo(task))

    print('task_outputs computed')

    n_sub_datasets = len(dataset)
    regrets_list = []
    bo_results_list = []
    for i in range(n_bo_runs):
        regrets_i_list = []
        for j, (sub_dataset_key, sub_dataset) in enumerate(dataset.items()):
            bo_results_ij = task_outputs[i * n_sub_datasets + j]
            bo_results_ij['bo_run_index'] = i
            bo_results_ij['sub_dataset_key'] = sub_dataset_key
            regrets_ij = bo_results_ij['regrets']
            regrets_i_list.append(regrets_ij)
            bo_results_list.append(bo_results_ij)
        regrets_list.append(jnp.mean(jnp.array(regrets_i_list), axis=0))
    regrets_list = jnp.array(regrets_list)
    results = {
        'regrets_list': regrets_list,
        'bo_results_list': bo_results_list,
    }
    return results


def split_test_bo_setup_b_id(dir_path, key, test_id, dataset_func_split, base_cov_func, base_mean_func, n_init_obs,
                             budget, n_bo_runs, gp_objective, gp_retrain_maxiter, gp_retrain_method, ac_func_type,
                             hand_hgp_params, uniform_hgp_params, log_uniform_hgp_params, gt_hgp_params, gt_gp_params,
                             bo_node_cpu_count, distribution_type, dataset_dim_feature_values_path, method_name,
                             extra_suffix=''):
    if bo_node_cpu_count is None or bo_node_cpu_count <= 1:
        pool = None
    else:
        # pool = ProcessingPool(bo_node_cpu_count)
        pool = None

    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[test_id]

    if ac_func_type == 'ucb':
        ac_func = acfun.ucb
    elif ac_func_type == 'ei':
        ac_func = acfun.ei
    elif ac_func_type == 'pi':
        ac_func = acfun.pi
    else:
        raise ValueError('Unknown ac_func_type: {}'.format(ac_func_type))

    dataset_all = dataset_func_split(test_id)
    dataset = dataset_all['test']  # only use test set
    if 'init_index' in dataset_all:
        init_indices_values = dataset_all['init_index']
    else:
        init_indices_values = None

    results = test_bo(
        dir_path, key, test_id, pool, dataset, base_cov_func, base_mean_func, n_init_obs, init_indices_values, budget,
        n_bo_runs, gp_objective, gp_retrain_maxiter, gp_retrain_method, ac_func, hand_hgp_params, uniform_hgp_params,
        log_uniform_hgp_params, gt_hgp_params, gt_gp_params, distribution_type, dim_feature_value, method_name,
        extra_suffix,
    )
    np.save(os.path.join(dir_path, 'split_test_bo_setup_b_id_{}_{}_{}{}.npy'.format(test_id, method_name, ac_func_type,
                                                                                    extra_suffix)),
            results)
