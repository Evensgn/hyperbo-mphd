import argparse
import os
import jax.numpy as jnp
import numpy as np

from experiment_defs import RESULTS_DIR
from hpl_bo_fitting import split_fit_gp_params_id, split_fit_direct_hgp_two_step, split_fit_hpl_hgp_two_step, \
    split_fit_hpl_hgp_end_to_end, split_eval_hpl_hgp_loss_setup_b
from hpl_bo_running_bo import split_test_bo_setup_b_id
from hpl_bo_computing_nll import split_eval_nll_setup_b_id, split_eval_kl_with_gt
from hpl_bo_merge import split_merge
from hpl_bo_merge_vary_num_dataset import split_merge_vary_num_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPL-BO split worker.')

    parser.add_argument('--group_id', default='split_0', type=str, help='split group id')
    parser.add_argument('--mode', default='', type=str, help='mode')
    parser.add_argument('--dataset_id', default='', type=str, help='dataset id')
    parser.add_argument('--ac_func_type', default='', type=str, help='ac func type')
    parser.add_argument('--method_name', default='', type=str, help='method name')
    parser.add_argument('--key_0', default=0, type=int, help='key 0')
    parser.add_argument('--key_1', default=0, type=int, help='key 1')
    parser.add_argument('--extra_suffix', default='', type=str, help='extra suffix')
    parser.add_argument('--eval_nll_gp_retrain_batch_id', default=-1, type=int, help='eval nll gp retrain batch id')
    args = parser.parse_args()

    dir_path = os.path.join(RESULTS_DIR, 'hpl_bo_split', args.group_id)

    # construct the jax random key
    key = jnp.array([args.key_0, args.key_1], dtype=jnp.uint32)

    # read configs
    configs = np.load(os.path.join(dir_path, 'configs.npy'), allow_pickle=True).item()

    random_seed = configs['random_seed']
    train_id_list = configs['train_id_list']
    test_id_list = configs['test_id_list']

    setup_b_id_list = configs['setup_b_id_list']
    if args.extra_suffix != '':
        setup_b_id_list = configs['synthetic_smaller_train_id_lists'][args.extra_suffix]

    dataset_func_combined = configs['dataset_func_combined']
    dataset_func_split = configs['dataset_func_split']
    dataset_dim_feature_values_path = configs['dataset_dim_feature_values_path']
    extra_info = configs['extra_info']

    fit_gp_maxiter = configs['fit_gp_maxiter']
    fit_gp_batch_size = configs['fit_gp_batch_size']
    fit_gp_adam_learning_rate = configs['fit_gp_adam_learning_rate']

    fit_hgp_maxiter = configs['fit_hgp_maxiter']
    fit_hgp_batch_size = configs['fit_hgp_batch_size']
    fit_hgp_adam_learning_rate = configs['fit_hgp_adam_learning_rate']

    fit_two_step_maxiter = configs['fit_two_step_maxiter']
    fit_two_step_learning_rate = configs['fit_two_step_learning_rate']

    gp_retrain_maxiter = configs['gp_retrain_maxiter']
    gp_retrain_method = configs['gp_retrain_method']

    n_init_obs = configs['n_init_obs']
    budget = configs['budget']
    n_bo_runs = configs['n_bo_runs']

    eval_loss_batch_size = configs['eval_loss_batch_size']
    eval_loss_n_batches = configs['eval_loss_n_batches']

    eval_nll_gp_retrain_maxiter = configs['eval_nll_gp_retrain_maxiter']
    eval_nll_gp_retrain_method = configs['eval_nll_gp_retrain_method']
    eval_nll_gp_retrain_n_observations = configs['eval_nll_gp_retrain_n_observations']
    eval_nll_gp_retrain_n_batches = configs['eval_nll_gp_retrain_n_batches']
    eval_nll_batch_size = configs['eval_nll_batch_size']
    eval_nll_n_batches = configs['eval_nll_n_batches']

    ac_func_type_list = configs['ac_func_type_list']

    hand_hgp_params = configs['hand_hgp_params']
    uniform_hgp_params = configs['uniform_hgp_params']
    log_uniform_hgp_params = configs['log_uniform_hgp_params']
    gt_hgp_params = configs['gt_hgp_params']
    gt_gp_params = configs['gt_gp_params']

    kernel_name = configs['kernel_name']
    cov_func = configs['cov_func']
    mean_func = configs['mean_func']
    gp_objective = configs['gp_objective']
    hgp_objective = configs['hgp_objective']
    fit_gp_method = configs['fit_gp_method']
    distribution_type = configs['distribution_type']

    fitting_node_cpu_count = configs['fitting_node_cpu_count']
    bo_node_cpu_count = configs['bo_node_cpu_count']

    if args.mode == 'fit_gp_params_setup_b_id':
        split_fit_gp_params_id(dir_path, key, 'b', args.dataset_id, dataset_func_combined, dataset_func_split,
                               cov_func, mean_func, gp_objective, fit_gp_method, fit_gp_maxiter, fit_gp_batch_size,
                               fit_gp_adam_learning_rate, args.method_name)

    elif args.mode == 'fit_direct_hgp_two_step_setup_b':
        split_fit_direct_hgp_two_step(dir_path, 'b', setup_b_id_list, distribution_type,
                                      extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_direct_hgp_two_step_setup_b_leaveout':
        split_fit_direct_hgp_two_step(dir_path, 'b', setup_b_id_list, distribution_type, leaveout_id=args.dataset_id,
                                      extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_two_step_setup_b':
        split_fit_hpl_hgp_two_step(dir_path, key, 'b', setup_b_id_list, fit_two_step_maxiter,
                                   fit_two_step_learning_rate, distribution_type, dataset_dim_feature_values_path,
                                   extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_two_step_setup_b_leaveout':
        split_fit_hpl_hgp_two_step(dir_path, key, 'b', setup_b_id_list, fit_two_step_maxiter,
                                   fit_two_step_learning_rate, distribution_type, dataset_dim_feature_values_path,
                                   leaveout_id=args.dataset_id, extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, fit_gp_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=True, extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b_leaveout':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, fit_gp_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=True, leaveout_id=args.dataset_id,
                                     extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b_no_init':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, fit_gp_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=False, extra_suffix=args.extra_suffix)
    elif args.mode == 'fit_hpl_hgp_end_to_end_setup_b_leaveout_no_init':
        split_fit_hpl_hgp_end_to_end(dir_path, key, 'b', setup_b_id_list, dataset_func_combined, dataset_func_split,
                                     dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective, fit_gp_method,
                                     fit_hgp_maxiter, fit_hgp_batch_size, fit_hgp_adam_learning_rate, distribution_type,
                                     use_init_params_value=False, leaveout_id=args.dataset_id,
                                     extra_suffix=args.extra_suffix)

    elif args.mode == 'test_bo_setup_b_id':
        split_test_bo_setup_b_id(dir_path, key, args.dataset_id, dataset_func_split, cov_func, mean_func, n_init_obs,
                                 budget, n_bo_runs, gp_objective, gp_retrain_maxiter, gp_retrain_method,
                                 args.ac_func_type, hand_hgp_params, uniform_hgp_params, log_uniform_hgp_params,
                                 gt_hgp_params, gt_gp_params, bo_node_cpu_count, distribution_type,
                                 dataset_dim_feature_values_path, args.method_name,
                                 extra_suffix=args.extra_suffix)

    elif args.mode == 'eval_nll_setup_b_train_id' or args.mode == 'eval_nll_setup_b_test_id':
        train_or_test = 'train' if args.mode == 'eval_nll_setup_b_train_id' else 'test'
        split_eval_nll_setup_b_id(dir_path, key, train_or_test, args.dataset_id, dataset_func_split, cov_func,
                                  mean_func, hand_hgp_params, uniform_hgp_params, log_uniform_hgp_params, gt_hgp_params,
                                  gt_gp_params, distribution_type, dataset_dim_feature_values_path, gp_objective,
                                  eval_nll_gp_retrain_maxiter, eval_nll_gp_retrain_method,
                                  eval_nll_gp_retrain_n_observations, args.eval_nll_gp_retrain_batch_id,
                                  eval_nll_batch_size, eval_nll_n_batches,
                                  args.method_name, extra_suffix=args.extra_suffix)

    elif args.mode == 'eval_loss_setup_b_train' or args.mode == 'eval_loss_setup_b_test':
        train_or_test = 'train' if args.mode == 'eval_loss_setup_b_train' else 'test'
        split_eval_hpl_hgp_loss_setup_b(dir_path, key, train_or_test, setup_b_id_list, dataset_func_split,
                                        dataset_dim_feature_values_path, cov_func, mean_func, hgp_objective,
                                        eval_loss_batch_size, eval_loss_n_batches, distribution_type,
                                        args.method_name, extra_suffix=args.extra_suffix)

    elif args.mode == 'merge':
        split_merge(dir_path, args.group_id, configs)
    elif args.mode == 'merge_vary_num_dataset':
        split_merge_vary_num_dataset(dir_path, args.group_id, configs)
    elif args.mode == 'eval_kl_with_gt':
        split_eval_kl_with_gt(dir_path, gt_hgp_params, distribution_type, args.method_name,
                              extra_suffix=args.extra_suffix)
    else:
        raise ValueError('Unknown mode: {}'.format(args.mode))
