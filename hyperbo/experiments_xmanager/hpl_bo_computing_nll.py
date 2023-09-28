from hyperbo.basics import definitions as defs
from hyperbo.basics import data_utils
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
from hyperbo.gp_utils import gp
import jax
import jax.numpy as jnp
import numpy as np
from pathos.multiprocessing import ProcessingPool
import os
from experiment_defs import *
from hpl_bo_utils import get_gp_priors_from_direct_hgp, get_gp_priors_from_hpl_hgp, get_gp_priors_from_gt_hgp,\
    get_gp_prior_dist_from_gt_hgp, get_gp_prior_dist_from_direct_hgp, get_gp_prior_dist_for_synthetic_from_hpl_hgp
from tensorflow_probability.substrates.jax.distributions import kl_divergence

def nll_on_sub_dataset(gp_params, cov_func, mean_func, sub_dataset, warp_func):
    dataset = {'only': sub_dataset}
    return obj.nll(
        mean_func,
        cov_func,
        gp_params,
        dataset,
        warp_func=warp_func,
        exclude_aligned=True,
    )


def nll_sub_dataset_level_with_gp(key, nll_data_batch_key, dataset, cov_func, mean_func, gp_params, eval_nll_batch_size,
                                  eval_nll_n_batches):
    # sub sample each sub dataset for large datasets
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        nll_data_batch_key, dataset, eval_nll_batch_size)

    nll_loss_mean_per_batch = []
    nll_loss_dict = {}
    for sub_dataset_key in dataset.keys():
        nll_loss_dict[sub_dataset_key] = []
    for k in range(eval_nll_n_batches):
        dataset_k = next(dataset_iter)
        nll_loss_list = []
        for sub_dataset_key, sub_dataset in dataset_k.items():
            nll_i = nll_on_sub_dataset(gp_params, cov_func, mean_func, sub_dataset, warp_func=None)
            nll_loss_list.append(nll_i)
            nll_loss_dict[sub_dataset_key].append(nll_i)
        nll_loss_batch = jnp.mean(jnp.array(nll_loss_list))
        nll_loss_mean_per_batch.append(nll_loss_batch)
    nll_loss_dict['mean_per_batch'] = nll_loss_mean_per_batch
    return nll_loss_dict


def nll_sub_dataset_level_with_gp_params_prior(key, nll_data_batch_key, hgp_retrain_batch_key, dataset, n_dim, cov_func,
                                               mean_func, gp_priors, gp_objective,
                                               gp_retrain_maxiter, gp_retrain_method, gp_retrain_n_observations,
                                               eval_nll_batch_size, eval_nll_n_batches):
    # retrain observations iterator
    retrain_observations_iter = data_utils.sub_sample_dataset_iterator(
        hgp_retrain_batch_key, dataset, gp_retrain_n_observations)

    nll_loss_dict = {}
    observations_i = next(retrain_observations_iter)
    for sub_dataset_key in dataset.keys():
        nll_loss_dict[sub_dataset_key] = []
    nll_loss_mean_per_batch_i = []

    # sub sample each sub dataset for large datasets
    dataset_iter = data_utils.sub_sample_dataset_iterator(
        nll_data_batch_key, dataset, eval_nll_batch_size)
    for k in range(eval_nll_n_batches):
        dataset_k = next(dataset_iter)
        nll_loss_list_ik = []
        for sub_dataset_key in dataset.keys():
            observations = observations_i[sub_dataset_key]
            sub_dataset = dataset_k[sub_dataset_key]

            gp_params = defs.GPParams(
                model={
                    'constant': 1.0,
                    'lengthscale': jnp.array([1.0] * n_dim),
                    'signal_variance': 0.,
                    'noise_variance': -4.
                },
                config={
                    'method': gp_retrain_method,
                    'maxiter': gp_retrain_maxiter,
                    'logging_interval': 1,
                    'objective': gp_objective,
                    'batch_size': gp_retrain_n_observations,
                    'priors': gp_priors,
                })

            single_gp_warp_func = utils.single_gp_default_warp_func
            model = gp.GP(
                dataset={'observations': observations},
                mean_func=mean_func,
                cov_func=cov_func,
                params=gp_params,
                warp_func=single_gp_warp_func,
            )
            model.train()
            del model.params.config['priors']  # remove priors after retraining

            nll_iks = nll_on_sub_dataset(gp_params, cov_func, mean_func, sub_dataset,
                                         warp_func=single_gp_warp_func)
            nll_loss_list_ik.append(nll_iks)
            nll_loss_dict[sub_dataset_key].append(nll_iks)
        nll_loss_mean_ik = jnp.mean(jnp.array(nll_loss_list_ik))
        nll_loss_mean_per_batch_i.append(nll_loss_mean_ik)
    nll_loss_dict['mean_per_batch'] = nll_loss_mean_per_batch_i
    return nll_loss_dict


def eval_nll(dir_path, key, dataset_id, dataset, base_cov_func, base_mean_func, hand_hgp_params,
             uniform_hgp_params, log_uniform_hgp_params, gt_hgp_params, gt_gp_params, distribution_type,
             dim_feature_value, gp_objective, gp_retrain_maxiter, gp_retrain_method, gp_retrain_n_observations,
             eval_nll_batch_size, eval_nll_n_batches, method_name):
    n_dim = list(dataset.values())[0].x.shape[1]
    if method_name in ['base', 'hyperbo', 'ablr', 'fsbo', 'gt_gp']:
        gp_priors = None
        if method_name == 'base':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}.npy'.format(dataset_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            cov_func = base_cov_func
            mean_func = base_mean_func
        elif method_name == 'hyperbo':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_hyperbo.npy'.format(
                dataset_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = HYPERBO_MLP_FEATURES
            cov_func = HYPERBO_KERNEL_TYPE
            mean_func = HYPERBO_MEAN_TYPE
        elif method_name == 'ablr':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_ablr.npy'.format(dataset_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = ABLR_MLP_FEATURES
            cov_func = ABLR_KERNEL_TYPE
            mean_func = ABLR_MEAN_TYPE
        elif method_name == 'fsbo':
            gp_params = np.load(os.path.join(dir_path, 'split_fit_gp_params_setup_b_id_{}_fsbo.npy'.format(dataset_id)),
                                allow_pickle=True).item()['gp_params']
            gp_params = defs.GPParams(model=gp_params)
            gp_params.config['mlp_features'] = FSBO_MLP_FEATURES
            cov_func = FSBO_KERNEL_TYPE
            mean_func = FSBO_MEAN_TYPE
        elif method_name == 'gt_gp':
            gp_params = gt_gp_params[dataset_id]
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
            direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b.npy'),
                                        allow_pickle=True).item()['gp_distribution_params']
        elif method_name == 'fit_direct_hgp_leaveout':
            direct_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
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
            hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
                                     allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_leaveout':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_from_scratch':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_end_to_end_leaveout_from_scratch':
            hpl_hgp_params = np.load(
                os.path.join(dir_path,
                             'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch.npy'.format(dataset_id)),
                allow_pickle=True).item()['gp_params']
        elif method_name == 'hpl_hgp_two_step':
            hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'),
                                     allow_pickle=True).item()
        elif method_name == 'hpl_hgp_two_step_leaveout':
            hpl_hgp_params = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))
        gp_priors = get_gp_priors_from_hpl_hgp(hpl_hgp_params, distribution_type, n_dim, dim_feature_value)
    else:
        raise ValueError('Unknown method name: {}'.format(method_name))

    nll_data_batch_key = jax.random.PRNGKey(0)  # keep this the same for all methods
    hgh_retrain_batch_key, key = jax.random.split(key, 2)
    if gp_params is not None:
        results = nll_sub_dataset_level_with_gp(key, nll_data_batch_key, dataset, cov_func, mean_func, gp_params, eval_nll_batch_size,
                                                eval_nll_n_batches)
    else:
        results = nll_sub_dataset_level_with_gp_params_prior(key, nll_data_batch_key, hgh_retrain_batch_key, dataset, n_dim, cov_func, mean_func, gp_priors,
                                                             gp_objective, gp_retrain_maxiter, gp_retrain_method,
                                                             gp_retrain_n_observations,
                                                             eval_nll_batch_size, eval_nll_n_batches)

    return results


def split_eval_nll_setup_b_id(dir_path, key, train_or_test, dataset_id, dataset_func_split, base_cov_func,
                              base_mean_func, hand_hgp_params, uniform_hgp_params, log_uniform_hgp_params,
                              gt_hgp_params, gt_gp_params, distribution_type,
                              dataset_dim_feature_values_path, gp_objective, gp_retrain_maxiter, gp_retrain_method,
                              gp_retrain_n_observations, eval_nll_gp_retrain_batch_id, eval_nll_batch_size,
                              eval_nll_n_batches, method_name, extra_suffix=''):
    if train_or_test == 'train':
        dataset = dataset_func_split(dataset_id)['train']  # only use train set
    elif train_or_test == 'test':
        dataset = dataset_func_split(dataset_id)['test']  # only use test set
    else:
        raise ValueError('Unknown train_or_test: {}'.format(train_or_test))

    dim_feature_value = np.load(dataset_dim_feature_values_path, allow_pickle=True).item()[dataset_id]

    new_key, key = jax.random.split(key, 2)

    results = eval_nll(dir_path, new_key, dataset_id, dataset, base_cov_func, base_mean_func, hand_hgp_params,
                       uniform_hgp_params, log_uniform_hgp_params, gt_hgp_params, gt_gp_params, distribution_type,
                       dim_feature_value, gp_objective, gp_retrain_maxiter, gp_retrain_method,
                       gp_retrain_n_observations, eval_nll_batch_size, eval_nll_n_batches, method_name)

    np.save(os.path.join(dir_path, 'split_eval_nll_setup_b_{}_id_{}_{}_{}{}.npy'.format(train_or_test,
                                                                                   dataset_id, method_name,
                                                                                   eval_nll_gp_retrain_batch_id,
                                                                                   extra_suffix)), results)


def eval_kl_with_gt(dir_path, gt_hgp_params, distribution_type, n_dim_list, method_name, extra_suffix):
    results = {}
    gp_hyperparam_names = ['constant', 'lengthscale', 'signal_variance', 'noise_variance']
    for gp_hyperparam_name in gp_hyperparam_names:
        results[gp_hyperparam_name] = {}

    for n_dim in n_dim_list:
        gt_hgp_prior_dist = get_gp_prior_dist_from_gt_hgp(gt_hgp_params, distribution_type, n_dim)
        if method_name in ['fit_direct_hgp']:
            if method_name == 'fit_direct_hgp':
                direct_hgp_params = np.load(os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b{}.npy'.format(extra_suffix)),
                                            allow_pickle=True).item()['gp_distribution_params']
            else:
                raise ValueError('Unknown method name: {}'.format(method_name))
            gp_prior_dist = get_gp_prior_dist_from_direct_hgp(direct_hgp_params, distribution_type)
        elif method_name in ['hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch', 'hpl_hgp_two_step']:
            if method_name == 'hpl_hgp_end_to_end':
                hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b{}.npy'.format(extra_suffix)),
                                         allow_pickle=True).item()['gp_params']
            elif method_name == 'hpl_hgp_end_to_end_from_scratch':
                hpl_hgp_params = np.load(
                    os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch{}.npy'.format(extra_suffix)),
                    allow_pickle=True).item()['gp_params']
            elif method_name == 'hpl_hgp_two_step':
                hpl_hgp_params = np.load(os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b{}.npy'.format(extra_suffix)),
                                         allow_pickle=True).item()
            else:
                raise ValueError('Unknown method name: {}'.format(method_name))
            gp_prior_dist = get_gp_prior_dist_for_synthetic_from_hpl_hgp(hpl_hgp_params, distribution_type, n_dim)
        else:
            raise ValueError('Unknown method name: {}'.format(method_name))

        for gp_hyperparam_name in gp_hyperparam_names:
            results[gp_hyperparam_name]['n={}'.format(n_dim)] = kl_divergence(
                gt_hgp_prior_dist[gp_hyperparam_name],
                gp_prior_dist[gp_hyperparam_name])

    for gp_hyperparam_name in gp_hyperparam_names:
        kl_list = []
        for n_dim in n_dim_list:
            kl_list.append(results[gp_hyperparam_name]['n={}'.format(n_dim)])
        kl_list = jnp.array(kl_list)
        results[gp_hyperparam_name]['list'] = kl_list
        results[gp_hyperparam_name]['mean'] = jnp.mean(kl_list)
        results[gp_hyperparam_name]['std'] = jnp.std(kl_list)

    return results


def split_eval_kl_with_gt(dir_path, gt_hgp_params, distribution_type, method_name, extra_suffix):
    n_dim_list = list(range(2, 15))

    results = eval_kl_with_gt(dir_path, gt_hgp_params, distribution_type, n_dim_list, method_name, extra_suffix)

    np.save(os.path.join(dir_path, 'split_eval_kl_with_gt_{}{}.npy'.format(method_name, extra_suffix)), results)
