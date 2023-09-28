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


DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params
jit = jax.jit


hpob_full_id_list = ['4796', '5527', '5636', '5859', '5860', '5891', '5906', '5965', '5970', '5971', '6766', '6767',
                     '6794', '7607', '7609', '5889']


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    sub_sample_batch_size = 100

    dataset_combined = {}
    dataset_train_save = {}
    dataset_test_save = {}
    for id in hpob_full_id_list:
        print('id = {}'.format(id))
        dataset = data.hpob_dataset_v2(id)
        key, subkey = jax.random.split(key, 2)
        dataset_iter = data_utils.sub_sample_dataset_iterator(
            subkey, dataset, sub_sample_batch_size)
        dataset_combined[id] = next(dataset_iter)

        dataset_train, dataset_test = data.hpob_dataset_v3(id)
        key, subkey = jax.random.split(key, 2)
        dataset_train_iter = data_utils.sub_sample_dataset_iterator(
            subkey, dataset_train, sub_sample_batch_size)
        dataset_train_save[id] = next(dataset_train_iter)
        key, subkey = jax.random.split(key, 2)
        dataset_test_iter = data_utils.sub_sample_dataset_iterator(
            subkey, dataset_test, sub_sample_batch_size)
        dataset_test_save[id] = next(dataset_test_iter)

    dataset_converted = {
        'combined': dataset_combined,
        'train': dataset_train_save,
        'test': dataset_test_save
    }

    np.save('./hpob_converted_data/sub_sample_{}.npy'.format(sub_sample_batch_size), dataset_converted)

