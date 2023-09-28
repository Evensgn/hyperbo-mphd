import os

import jax.numpy as jnp
import numpy as np
from experiment_defs import *


def split_merge(dir_path, group_id, configs):
    experiment_name = 'hpl_bo_split_group_id_{}_merge'.format(group_id)
    results = {}

    results['experiment_name'] = experiment_name

    # configs
    results['configs'] = configs
    results['configs']['hpl_lengthscale_mlp_features'] = HPL_LENGTHSCALE_MLP_FEATURES

    setup_b_id_list = configs['setup_b_id_list']
    method_name_list = configs['method_name_list']
    ac_func_type_list = configs['ac_func_type_list']
    n_bo_runs = configs['n_bo_runs']
    eval_nll_n_batches = configs['eval_nll_n_batches']

    # setup b
    results['setup_b'] = {}
    results_b = results['setup_b']

    # fit gp parameters
    if IS_HPOB:
        fit_single_gp_list = setup_b_id_list + ['pd1']
    else:
        fit_single_gp_list = setup_b_id_list
    results_b['fit_base_gp_params'] = {}
    for dataset_id in fit_single_gp_list:
        results_b['fit_base_gp_params'][dataset_id] = np.load(
            os.path.join(dir_path,
                         'split_fit_gp_params_setup_b_id_{}.npy'.format(dataset_id)), allow_pickle=True).item()
    results_b['fit_hyperbo_params'] = {}
    for dataset_id in fit_single_gp_list:
        results_b['fit_hyperbo_params'][dataset_id] = np.load(
            os.path.join(dir_path,
                         'split_fit_gp_params_setup_b_id_{}_hyperbo.npy'.format(dataset_id)), allow_pickle=True).item()
    results_b['fit_ablr_params'] = {}
    for dataset_id in fit_single_gp_list:
        results_b['fit_ablr_params'][dataset_id] = np.load(
            os.path.join(dir_path,
                         'split_fit_gp_params_setup_b_id_{}_ablr.npy'.format(dataset_id)), allow_pickle=True).item()
    results_b['fit_fsbo_params'] = {}
    for dataset_id in fit_single_gp_list:
        results_b['fit_fsbo_params'][dataset_id] = np.load(
            os.path.join(dir_path,
                         'split_fit_gp_params_setup_b_id_{}_fsbo.npy'.format(dataset_id)), allow_pickle=True).item()

    # read fit direct hgp params
    if 'fit_direct_hgp' in method_name_list:
        results_b['fit_direct_hgp_params'] = np.load(
            os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b.npy'),
            allow_pickle=True).item()['gp_distribution_params']
    if 'fit_direct_hgp_leaveout' in method_name_list:
        results_b['fit_direct_hgp_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['fit_direct_hgp_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()['gp_distribution_params']

    # read fit hpl hgp params
    if 'hpl_hgp_end_to_end' in method_name_list:
        results_b['hpl_hgp_end_to_end_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_leaveout' in method_name_list:
        results_b['hpl_hgp_end_to_end_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_end_to_end_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_from_scratch' in method_name_list:
        results_b['hpl_hgp_end_to_end_from_scratch_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b_from_scratch.npy'),
            allow_pickle=True).item()
    if 'hpl_hgp_end_to_end_leaveout_from_scratch' in method_name_list:
        results_b['hpl_hgp_end_to_end_leaveout_from_scratch_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_end_to_end_leaveout_from_scratch_params'][dataset_id] = np.load(
                os.path.join(dir_path,
                             'split_fit_hpl_hgp_end_to_end_setup_b_leaveout_{}_from_scratch.npy'.format(dataset_id)),
                allow_pickle=True).item()
    if 'hpl_hgp_two_step' in method_name_list:
        results_b['hpl_hgp_two_step_params'] = np.load(
            os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b.npy'), allow_pickle=True).item()
    if 'hpl_hgp_two_step_leaveout' in method_name_list:
        results_b['hpl_hgp_two_step_leaveout_params'] = {}
        for dataset_id in setup_b_id_list:
            results_b['hpl_hgp_two_step_leaveout_params'][dataset_id] = np.load(
                os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b_leaveout_{}.npy'.format(dataset_id)),
                allow_pickle=True).item()

    # run BO and compute NLL
    results_b['bo_results'] = {}
    results_b['bo_results_total'] = {}
    results_b['nll_results'] = {}
    results_b['nll_results_total'] = {}
    results_b['training_loss_eval'] = {}
    results_b['kl_eval'] = {}

    if IS_HPOB:
        results_b['bo_results_pd1'] = {}
        results_b['bo_results_pd1_total'] = {}
        results_b['nll_results_pd1'] = {}
        results_b['nll_results_pd1_total'] = {}
        for method_name in PD1_METHOD_NAME_LIST:
            results_b['bo_results_pd1'][method_name] = {}
            results_b['bo_results_pd1_total'][method_name] = {}
            results_b['nll_results_pd1'][method_name] = {'train': {}, 'test': {}}
            results_b['nll_results_pd1_total'][method_name] = {}

            for ac_func_type in ac_func_type_list:
                results_b['bo_results_pd1'][method_name][ac_func_type] = np.load(
                    os.path.join(dir_path,
                                 'split_test_bo_setup_b_id_pd1_{}_{}.npy'.format(method_name, ac_func_type)),
                    allow_pickle=True,
                ).item()

                regrets_all_list = []
                for i in range(n_bo_runs):
                    regrets_all_list.append(results_b['bo_results_pd1'][method_name][ac_func_type]['regrets_list'][i])

                regrets_all_list = jnp.array(regrets_all_list)
                regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
                regrets_std_total = jnp.std(regrets_all_list, axis=0)

                results_b['bo_results_pd1_total'][method_name][ac_func_type] = {
                    'regrets_all_list': regrets_all_list,
                    'regrets_mean': regrets_mean_total,
                    'regrets_std': regrets_std_total,
                }

                # NLL evaluations

                if method_name == 'random':
                    continue
                if method_name not in ['hpl_hgp_two_step', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch']:
                    continue

                if method_name in ['base', 'hyperbo', 'ablr', 'fsbo', 'gt_gp']:
                    nll_on_train_list = []
                    for k in range(eval_nll_n_batches):
                        nll_on_train_list.append([])
                    results_b['nll_results_pd1'][method_name]['train'] = np.load(os.path.join(
                        dir_path, 'split_eval_nll_setup_b_train_id_pd1_{}.npy'.format(method_name)),
                        allow_pickle=True).item()
                    for k in range(eval_nll_n_batches):
                        nll_on_train_list[k].append(
                            results_b['nll_results_pd1'][method_name]['train']['mean_per_batch'][k])

                    nll_on_test_list = []
                    for k in range(eval_nll_n_batches):
                        nll_on_test_list.append([])
                    results_b['nll_results_pd1'][method_name]['test'] = np.load(os.path.join(
                        dir_path, 'split_eval_nll_setup_b_test_id_pd1_{}.npy'.format(method_name)),
                        allow_pickle=True).item()
                    for k in range(eval_nll_n_batches):
                        nll_on_test_list[k].append(
                            results_b['nll_results_pd1'][method_name]['test']['mean_per_batch'][k])

                    # redundant code as there is only one train/test dataset which is pd1
                    nll_on_train_batches = []
                    nll_on_test_batches = []
                    for k in range(eval_nll_n_batches):
                        nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
                        nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

                    results_b['nll_results_pd1_total'][method_name] = {
                        'nll_on_train_mean': np.mean(nll_on_train_batches),
                        'nll_on_train_std': np.std(nll_on_train_batches),
                        'nll_on_test_mean': np.mean(nll_on_test_batches),
                        'nll_on_test_std': np.std(nll_on_test_batches),
                    }
                else:
                    gp_retrain_n_batches = configs['eval_nll_gp_retrain_n_batches']
                    for i in range(gp_retrain_n_batches):
                        results_b['nll_results_pd1'][method_name]['train'][i] = {}
                        results_b['nll_results_pd1'][method_name]['test'][i] = {}
                        nll_on_train_list = []
                        for k in range(eval_nll_n_batches):
                            nll_on_train_list.append([])
                        results_b['nll_results_pd1'][method_name]['train'][i] = np.load(os.path.join(
                            dir_path, 'split_eval_nll_setup_b_train_id_pd1_{}_{}.npy'.format(method_name, i)),
                            allow_pickle=True).item()
                        for k in range(eval_nll_n_batches):
                            nll_on_train_list[k].append(
                                results_b['nll_results_pd1'][method_name]['train'][i]['mean_per_batch'][k])

                        nll_on_test_list = []
                        for k in range(eval_nll_n_batches):
                            nll_on_test_list.append([])
                        results_b['nll_results_pd1'][method_name]['test'][i] = np.load(os.path.join(
                            dir_path, 'split_eval_nll_setup_b_test_id_pd1_{}_{}.npy'.format(method_name, i)),
                            allow_pickle=True).item()
                        for k in range(eval_nll_n_batches):
                            nll_on_test_list[k].append(
                                results_b['nll_results_pd1'][method_name]['test'][i]['mean_per_batch'][k])

                        # redundant code as there is only one train/test dataset which is pd1
                        nll_on_train_batches = []
                        nll_on_test_batches = []
                        for k in range(eval_nll_n_batches):
                            nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
                            nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

                        results_b['nll_results_pd1_total'][method_name][i] = {
                            'nll_on_train_mean': np.mean(nll_on_train_batches),
                            'nll_on_train_std': np.std(nll_on_train_batches),
                            'nll_on_test_mean': np.mean(nll_on_test_batches),
                            'nll_on_test_std': np.std(nll_on_test_batches),
                        }

    for method_name in method_name_list:
        # BO results
        results_b['bo_results'][method_name] = {}
        results_b['bo_results_total'][method_name] = {}

        for ac_func_type in ac_func_type_list:
            results_b['bo_results'][method_name][ac_func_type] = {}

            regrets_all_list = []

            for test_id in setup_b_id_list:
                results_b['bo_results'][method_name][ac_func_type][test_id] = np.load(
                    os.path.join(dir_path,
                                 'split_test_bo_setup_b_id_{}_{}_{}.npy'.format(test_id, method_name,
                                                                                ac_func_type)),
                    allow_pickle=True,
                ).item()

            for i in range(n_bo_runs):
                regrets_all_list_i = []

                for test_id in setup_b_id_list:
                    regrets_ij = results_b['bo_results'][method_name][ac_func_type][test_id]['regrets_list'][i]
                    regrets_all_list_i.append(regrets_ij)

                regrets_all_list.append(jnp.mean(jnp.array(regrets_all_list_i), axis=0))

            regrets_all_list = jnp.array(regrets_all_list)
            regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
            regrets_std_total = jnp.std(regrets_all_list, axis=0)

            results_b['bo_results_total'][method_name][ac_func_type] = {
                'regrets_all_list': regrets_all_list,
                'regrets_mean': regrets_mean_total,
                'regrets_std': regrets_std_total,
            }

        # Training loss evaluation
        if (method_name.startswith('hpl_hgp')) and ('leaveout' not in method_name):
            results_b['training_loss_eval'][method_name] = {}
            results_b['training_loss_eval'][method_name]['train'] = np.load(os.path.join(
                dir_path, 'split_eval_loss_setup_b_train_{}.npy'.format(method_name)), allow_pickle=True).item()
            results_b['training_loss_eval'][method_name]['test'] = np.load(os.path.join(
                dir_path, 'split_eval_loss_setup_b_test_{}.npy'.format(method_name)), allow_pickle=True).item()

        if not IS_HPOB:
            # KL divergence evaluation
            if method_name in EVAL_KL_METHOD_NAME_LIST:
                results_b['kl_eval'][method_name] = np.load(os.path.join(
                    dir_path, 'split_eval_kl_with_gt_{}.npy'.format(method_name)), allow_pickle=True).item()

        # NLL evaluations

        if method_name == 'random':
            continue
        if method_name not in ['hpl_hgp_two_step', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch']:
            continue

        results_b['nll_results'][method_name] = {
            'train': {},
            'test': {},
        }
        results_b['nll_results_total'][method_name] = {}

        if method_name in ['base', 'hyperbo', 'ablr', 'fsbo', 'gt_gp']:
            nll_on_train_list = []
            for k in range(eval_nll_n_batches):
                nll_on_train_list.append([])
            for train_id in setup_b_id_list:
                results_b['nll_results'][method_name]['train'][train_id] = np.load(os.path.join(
                    dir_path, 'split_eval_nll_setup_b_train_id_{}_{}.npy'.format(train_id, method_name)),
                    allow_pickle=True).item()
                for k in range(eval_nll_n_batches):
                    nll_on_train_list[k].append(
                        results_b['nll_results'][method_name]['train'][train_id]['mean_per_batch'][k])

            nll_on_test_list = []
            for k in range(eval_nll_n_batches):
                nll_on_test_list.append([])
            for test_id in setup_b_id_list:
                results_b['nll_results'][method_name]['test'][test_id] = np.load(os.path.join(
                    dir_path, 'split_eval_nll_setup_b_test_id_{}_{}.npy'.format(test_id, method_name)),
                    allow_pickle=True).item()
                for k in range(eval_nll_n_batches):
                    nll_on_test_list[k].append(
                        results_b['nll_results'][method_name]['test'][test_id]['mean_per_batch'][k])

            nll_on_train_batches = []
            nll_on_test_batches = []
            for k in range(eval_nll_n_batches):
                nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
                nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

            results_b['nll_results_total'][method_name] = {
                'nll_on_train_mean': np.mean(nll_on_train_batches),
                'nll_on_train_std': np.std(nll_on_train_batches),
                'nll_on_test_mean': np.mean(nll_on_test_batches),
                'nll_on_test_std': np.std(nll_on_test_batches),
            }
        else:
            gp_retrain_n_batches = configs['eval_nll_gp_retrain_n_batches']
            for i in range(gp_retrain_n_batches):
                results_b['nll_results'][method_name]['train'][i] = {}
                results_b['nll_results'][method_name]['test'][i] = {}
                nll_on_train_list = []
                for k in range(eval_nll_n_batches):
                    nll_on_train_list.append([])
                for train_id in setup_b_id_list:
                    results_b['nll_results'][method_name]['train'][i][train_id] = np.load(os.path.join(
                        dir_path, 'split_eval_nll_setup_b_train_id_{}_{}_{}.npy'.format(train_id, method_name, i)),
                        allow_pickle=True).item()
                    for k in range(eval_nll_n_batches):
                        nll_on_train_list[k].append(
                            results_b['nll_results'][method_name]['train'][i][train_id]['mean_per_batch'][k])

                nll_on_test_list = []
                for k in range(eval_nll_n_batches):
                    nll_on_test_list.append([])
                for test_id in setup_b_id_list:
                    results_b['nll_results'][method_name]['test'][i][test_id] = np.load(os.path.join(
                        dir_path, 'split_eval_nll_setup_b_test_id_{}_{}_{}.npy'.format(test_id, method_name, i)),
                        allow_pickle=True).item()
                    for k in range(eval_nll_n_batches):
                        nll_on_test_list[k].append(
                            results_b['nll_results'][method_name]['test'][i][test_id]['mean_per_batch'][k])

                nll_on_train_batches = []
                nll_on_test_batches = []
                for k in range(eval_nll_n_batches):
                    nll_on_train_batches.append(np.mean(nll_on_train_list[k]))
                    nll_on_test_batches.append(np.mean(nll_on_test_list[k]))

                results_b['nll_results_total'][method_name][i] = {
                    'nll_on_train_mean': np.mean(nll_on_train_batches),
                    'nll_on_train_std': np.std(nll_on_train_batches),
                    'nll_on_test_mean': np.mean(nll_on_test_batches),
                    'nll_on_test_std': np.std(nll_on_test_batches),
                }

    # save all results
    merge_path = os.path.join(dir_path, 'merge')
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    np.save(os.path.join(merge_path, 'results.npy'), results)

    # write part of results to text file
    with open(os.path.join(merge_path, 'results.txt'), 'w') as f:
        f.write(str(results))

    print('done.')
