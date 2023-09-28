from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean

GROUP_ID = 'hpl_bo_experiment_group_id'

RANDOM_SEED = 0

IS_HPOB = True

IS_GCP = True

DISTRIBUTION_TYPE = 'gamma'

SYNTHETIC_VERSION = 'beta'

HPL_LENGTHSCALE_MLP_FEATURES = (16, 16, 2)

if IS_HPOB:
    BASELINE_MLP_FEATURES = (128, 128)
else:
    BASELINE_MLP_FEATURES = (32, 32)

HYPERBO_KERNEL_TYPE = kernel.matern52_mlp
HYPERBO_MEAN_TYPE = mean.linear_mlp
HYPERBO_MLP_FEATURES = BASELINE_MLP_FEATURES

ABLR_KERNEL_TYPE = kernel.dot_product_mlp
ABLR_MEAN_TYPE = mean.zero
ABLR_MLP_FEATURES = BASELINE_MLP_FEATURES

FSBO_KERNEL_TYPE = kernel.matern52_mlp
FSBO_MEAN_TYPE = mean.zero
FSBO_MLP_FEATURES = BASELINE_MLP_FEATURES

EVAL_NLL_GP_RETRAIN_N_BATCHES = 20

FIT_SINGLE_GP_METHOD_LIST = ['base', 'hyperbo', 'ablr', 'fsbo']

METHOD_NAME_LIST = ['random', 'base', 'hyperbo', 'ablr', 'fsbo', 'hand_hgp', 'uniform_hgp', 'log_uniform_hgp',
                    'fit_direct_hgp', 'fit_direct_hgp_leaveout', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_leaveout',
                    'hpl_hgp_two_step', 'hpl_hgp_two_step_leaveout'] + \
                   ['hpl_hgp_end_to_end_from_scratch', 'hpl_hgp_end_to_end_leaveout_from_scratch']

METHOD_NAME_LIST_LEAVEOUT = ['random', 'hand_hgp', 'uniform_hgp', 'log_uniform_hgp', 'fit_direct_hgp_leaveout',
                             'hpl_hgp_end_to_end_leaveout', 'hpl_hgp_two_step_leaveout'] + \
                            ['hpl_hgp_end_to_end_leaveout_from_scratch']

if not IS_HPOB:
    METHOD_NAME_LIST += ['gt_hgp', 'gt_gp']

PD1_METHOD_NAME_LIST = ['random', 'base', 'hyperbo', 'ablr', 'fsbo', 'hand_hgp', 'uniform_hgp', 'log_uniform_hgp',
                        'fit_direct_hgp', 'hpl_hgp_end_to_end', 'hpl_hgp_two_step'] + \
                       ['hpl_hgp_end_to_end_from_scratch']

PD1_METHOD_NAME_LIST_LEAVEOUT = ['random', 'hand_hgp', 'uniform_hgp', 'log_uniform_hgp', 'fit_direct_hgp',
                                 'hpl_hgp_end_to_end', 'hpl_hgp_two_step'] + ['hpl_hgp_end_to_end_from_scratch']

VARY_NUM_DATASET_METHOD_NAME_LIST = ['fit_direct_hgp', 'hpl_hgp_two_step']

EVAL_KL_METHOD_NAME_LIST = ['fit_direct_hgp', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch',
                            'hpl_hgp_two_step']

HPOB_TRAIN_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767']
HPOB_TEST_ID_LIST = ['6794', '7607', '7609', '5889']
HPOB_FULL_ID_LIST = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                     '6794', '7607', '7609', '5889']

SYNTHETIC_TRAIN_ID_LIST = [str(x) for x in range(16)]
SYNTHETIC_TEST_ID_LIST = [str(x) for x in range(16, 20)]
SYNTHETIC_FULL_ID_LIST = [str(x) for x in range(20)]

import numpy as np

VARY_NUM_DATASET_NUM_LIST = [1, 2, 3, 4]
VARY_NUM_DATASET_NUM_SEEDS = 5

SYNTHETIC_SMALLER_TRAIN_ID_LISTS = {}
for num_dataset in VARY_NUM_DATASET_NUM_LIST:
    for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
        np.random.seed(seed)
        datasets = np.random.choice(SYNTHETIC_TRAIN_ID_LIST, num_dataset, replace=False).tolist()
        SYNTHETIC_SMALLER_TRAIN_ID_LISTS['_n{}seed{}'.format(num_dataset, seed)] = datasets


AC_FUNC_TYPE_LIST = ['ucb', 'ei', 'pi']

if IS_HPOB:
    TRAIN_ID_LIST = HPOB_TRAIN_ID_LIST
    TEST_ID_LIST = HPOB_TEST_ID_LIST
    FULL_ID_LIST = HPOB_FULL_ID_LIST
else:
    TRAIN_ID_LIST = SYNTHETIC_TRAIN_ID_LIST
    TEST_ID_LIST = SYNTHETIC_TEST_ID_LIST
    FULL_ID_LIST = SYNTHETIC_FULL_ID_LIST

if IS_GCP:
    HPOB_DATA_PATH = '/gcs/gcs_bucket_name/hpob_data_dir'
    PD1_DATA_PATH = '/gcs/gcs_bucket_name/pd1_data_dir'
    SYNTHETIC_DATA_PATH = \
        '/gcs/gcs_bucket_name/synthetic_data_dir/dataset_{}_{}.npy'.format(SYNTHETIC_VERSION,
                                                                           DISTRIBUTION_TYPE)
    SYNTHETIC_DATA_CONFIG_PATH = \
        '/gcs/gcs_bucket_name/synthetic_data_dir/dataset_{}_{}_configs.npy'.format(SYNTHETIC_VERSION,
                                                                                   DISTRIBUTION_TYPE)
    RESULTS_DIR = '/gcs/gcs_bucket_name/exp_results_dir'
    HPOB_DATA_ANALYSIS_PATH = '/gcs/gcs_bucket_name/hpob_data_dir/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = '/gcs/gcs_bucket_name/synthetic_data_dir/data_analysis_{}_{}.npy'.format(
        SYNTHETIC_VERSION, DISTRIBUTION_TYPE)
    FITTING_NODE_CPU_COUNT = 4
    FITTING_E2E_NODE_CPU_COUNT = 4
    BO_NODE_CPU_COUNT = 4
    NLL_NODE_CPU_COUNT = 4
    BASIC_CPU_COUNT = 4
else:
    HPOB_DATA_PATH = 'local_hpob_data_dir'
    PD1_DATA_PATH = 'local_pd1_data_dir'
    SYNTHETIC_DATA_PATH = 'local_synthetic_data_dir/dataset_{}_{}.npy'.format(SYNTHETIC_VERSION, DISTRIBUTION_TYPE)
    SYNTHETIC_DATA_CONFIG_PATH = 'local_synthetic_data_dir/dataset_{}_{}_configs.npy'.format(SYNTHETIC_VERSION,
                                                                                             DISTRIBUTION_TYPE)
    RESULTS_DIR = 'local_exp_results_dir'
    HPOB_DATA_ANALYSIS_PATH = 'local_hpob_data_dir/data_analysis.npy'
    SYNTHETIC_DATA_ANALYSIS_PATH = 'local_synthetic_data_dir/data_analysis_{}_{}.npy'.format(SYNTHETIC_VERSION,
                                                                                             DISTRIBUTION_TYPE)
    FITTING_NODE_CPU_COUNT = 4
    FITTING_E2E_NODE_CPU_COUNT = 4
    BO_NODE_CPU_COUNT = 4
    NLL_NODE_CPU_COUNT = 4
