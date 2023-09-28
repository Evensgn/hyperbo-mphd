import os

import jax.numpy as jnp
import numpy as np
from experiment_defs import *


def split_merge_vary_num_dataset(dir_path, group_id, configs):
    experiment_name = 'hpl_bo_split_group_id_{}_merge'.format(group_id)
    results = {}

    results['experiment_name'] = experiment_name

    # configs
    results['configs'] = configs
    results['configs']['hpl_lengthscale_mlp_features'] = HPL_LENGTHSCALE_MLP_FEATURES
    results['configs']['vary_num_dataset_method_name_list'] = VARY_NUM_DATASET_METHOD_NAME_LIST

    setup_b_id_list = configs['setup_b_id_list']
    method_name_list = configs['method_name_list']
    ac_func_type_list = configs['ac_func_type_list']
    n_bo_runs = configs['n_bo_runs']
    eval_nll_n_batches = configs['eval_nll_n_batches']

    results['setup_b'] = {}

    for num_dataset in VARY_NUM_DATASET_NUM_LIST:
        for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
            extra_suffix = '_n{}seed{}'.format(num_dataset, seed)
            # setup b
            results['setup_b'][extra_suffix] = {}
            results_b = results['setup_b'][extra_suffix]

            # fit gp parameters
            fit_single_gp_list = setup_b_id_list
            results_b['fit_base_gp_params'] = {}
            for dataset_id in fit_single_gp_list:
                results_b['fit_base_gp_params'][dataset_id] = np.load(
                    os.path.join(dir_path,
                                 'split_fit_gp_params_setup_b_id_{}.npy'.format(dataset_id)), allow_pickle=True).item()

            # read fit direct hgp params
            if 'fit_direct_hgp' in VARY_NUM_DATASET_METHOD_NAME_LIST:
                results_b['fit_direct_hgp_params'] = np.load(
                    os.path.join(dir_path, 'split_fit_direct_hgp_two_step_setup_b{}.npy'.format(extra_suffix)),
                    allow_pickle=True).item()['gp_distribution_params']

            # read fit hpl hgp params
            if 'hpl_hgp_end_to_end' in VARY_NUM_DATASET_METHOD_NAME_LIST:
                results_b['hpl_hgp_end_to_end_params'] = np.load(
                    os.path.join(dir_path, 'split_fit_hpl_hgp_end_to_end_setup_b{}.npy'.format(extra_suffix)),
                    allow_pickle=True).item()
            if 'hpl_hgp_two_step' in VARY_NUM_DATASET_METHOD_NAME_LIST:
                results_b['hpl_hgp_two_step_params'] = np.load(
                    os.path.join(dir_path, 'split_fit_hpl_hgp_two_step_setup_b{}.npy'.format(extra_suffix)),
                    allow_pickle=True).item()

            # run BO and compute NLL
            results_b['bo_results'] = {}
            results_b['bo_results_total'] = {}
            results_b['nll_results'] = {}
            results_b['nll_results_total'] = {}
            results_b['kl_eval'] = {}

            for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
                # BO results
                results_b['bo_results'][method_name] = {}
                results_b['bo_results_total'][method_name] = {}

                for ac_func_type in ac_func_type_list:
                    results_b['bo_results'][method_name][ac_func_type] = {}

                    regrets_all_list = []

                    for test_id in TEST_ID_LIST:
                        results_b['bo_results'][method_name][ac_func_type][test_id] = np.load(
                            os.path.join(dir_path,
                                         'split_test_bo_setup_b_id_{}_{}_{}{}.npy'.format(test_id, method_name,
                                                                                        ac_func_type,
                                                                                        extra_suffix)),
                            allow_pickle=True,
                        ).item()

                    for i in range(n_bo_runs):
                        regrets_all_list_i = []

                        for test_id in TEST_ID_LIST:
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

                # KL divergence evaluation
                if method_name in EVAL_KL_METHOD_NAME_LIST:
                    results_b['kl_eval'][method_name] = np.load(os.path.join(
                        dir_path, 'split_eval_kl_with_gt_{}{}.npy'.format(method_name, extra_suffix)),
                        allow_pickle=True).item()

    # save all results
    merge_path = os.path.join(dir_path, 'merge')
    if not os.path.exists(merge_path):
        os.mkdir(merge_path)
    np.save(os.path.join(merge_path, 'results.npy'), results)

    # write part of results to text file
    with open(os.path.join(merge_path, 'results.txt'), 'w') as f:
        f.write(str(results))

    print('done.')
