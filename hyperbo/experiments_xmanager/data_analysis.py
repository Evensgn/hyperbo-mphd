import functools
import logging
import time

from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.basics import data_utils
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
from functools import partial
import matplotlib.pyplot as plt
import gc
import subprocess
from experiment_defs import *


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


def create_dim_features(setup_b_id_list, dataset_func_split, dataset_func_combined, data_analysis_path):
    n_sub_datasets_total = 0
    max_sub_dataset_size = None

    dim_feature_set = {}
    # dim_feature: [is_discrete, is_continuous, n_discrete_dims, n_continuous_dims]
    for id in setup_b_id_list:
        dataset = dataset_func_split(id)['train']
        print('dataset id:', id)
        unique_set_per_dim = {}
        print('number of sub_datasets:', len(dataset))
        n_sub_datasets_total += len(dataset)
        for sub_dataset_key, sub_dataset in dataset.items():
            print('sub_dataset_key:', sub_dataset_key, ', sub_dataset_size:', sub_dataset.x.shape[0])
            print('min_y:', sub_dataset.y.min(), 'max_y:', sub_dataset.y.max())
            sub_dataset_size = sub_dataset.x.shape[0]
            if max_sub_dataset_size is None or sub_dataset_size > max_sub_dataset_size:
                max_sub_dataset_size = sub_dataset_size
            n_dim = sub_dataset.x.shape[1]
            for i in range(n_dim):
                unique_set_per_dim[i] = np.concatenate((unique_set_per_dim.get(i, np.array([])), sub_dataset.x[:, i]))

        n_discrete_dims = 0
        n_continuous_dims = 0
        for i in range(n_dim):
            unique_set_per_dim[i] = np.unique(unique_set_per_dim[i])
            if len(unique_set_per_dim[i]) == 2:
                n_discrete_dims += 1
            else:
                n_continuous_dims += 1
        dim_feature = []
        for i in range(n_dim):
            if len(unique_set_per_dim[i]) == 2:
                dim_feature.append([1., 0., n_discrete_dims, n_continuous_dims])
            else:
                dim_feature.append([0., 1., n_discrete_dims, n_continuous_dims])
        dim_feature_set[id] = np.array(dim_feature)

        print('dim_feature_set: ', dim_feature_set[id])

    print('n_sub_datasets_total: ', n_sub_datasets_total)
    print('max_sub_dataset_size: ', max_sub_dataset_size)
    np.save(data_analysis_path, dim_feature_set)


if __name__ == '__main__':
    group_id = GROUP_ID
    results_dir = RESULTS_DIR

    hpob_negative_y = False
    normalize_x = True
    normalize_y = True

    hpob_data_path = HPOB_DATA_PATH
    pd1_data_path = PD1_DATA_PATH
    dataset_func_combined = partial(data.hpob_dataset_v2, hpob_data_path=hpob_data_path, negative_y=hpob_negative_y,
                                    normalize_x=normalize_x, normalize_y=normalize_y)
    dataset_func_split = partial(data.hpob_dataset_v3, hpob_data_path=hpob_data_path, pd1_data_path=pd1_data_path,
                                 negative_y=hpob_negative_y, normalize_x=normalize_x, normalize_y=normalize_y)
    create_dim_features(HPOB_FULL_ID_LIST + ['pd1'], dataset_func_split, dataset_func_combined, HPOB_DATA_ANALYSIS_PATH)

    synthetic_data_path = SYNTHETIC_DATA_PATH
    dataset_func_combined = partial(data.hpl_bo_synthetic_dataset_combined, synthetic_data_path,
                                    normalize_x=False, normalize_y=False)
    dataset_func_split = partial(data.hpl_bo_synthetic_dataset_split, synthetic_data_path,
                                 normalize_x=False, normalize_y=False)

    create_dim_features(SYNTHETIC_FULL_ID_LIST, dataset_func_split, dataset_func_combined, SYNTHETIC_DATA_ANALYSIS_PATH)
