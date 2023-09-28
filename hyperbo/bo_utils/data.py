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

"""Load or generate data for learning priors in BayesOpt."""

import functools
import itertools
import pickle

import os
import math
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from absl import logging
from tensorflow.io import gfile
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma, LogNormal

from hyperbo.basics import data_utils
from hyperbo.basics import definitions as defs
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import mean

partial = functools.partial

SubDataset = defs.SubDataset

PD1 = {
    ('phase0', 'matched'): '../pd1/pd1_matched_phase0_results.jsonl',
    ('phase1', 'matched'): '../pd1/pd1_matched_phase1_results.jsonl',
    ('phase0', 'unmatched'): '../pd1/pd1_unmatched_phase0_results.jsonl',
    ('phase1', 'unmatched'): '../pd1/pd1_unmatched_phase1_results.jsonl',
}


# HPOB_ROOT_DIR = './../hpob-data/'


def sample_dataframe(key, df, p_remove=0.):
    """Randomly sample dataframe by the removal percentage."""
    if p_remove < 0 or p_remove >= 1:
        raise ValueError(
            f'p_remove={p_remove} but p_remove must be <1 and >= 0.')
    if p_remove > 0:
        n_remain = (1 - p_remove) * len(df)
        n_remain = int(np.ceil(n_remain))
        df = df.sample(n=n_remain, replace=False, random_state=key[0])
    return df


def get_aligned_dataset(trials,
                        study_identifier,
                        labels,
                        key=None,
                        p_remove=0.,
                        verbose=True):
    """Get aligned dataset from processed trials from get_dataset.

    Args:
      trials: pandas.DataFrame that stores all the trials.
      study_identifier: a label that uniquely identifies each study group.
      labels: labels of parameters and an eval metric (the last one).
      key: Jax random state.
      p_remove: proportion of data to be removed.
      verbose: print info about data if True.

    Returns:
      aligned_dataset: Dict[str, SubDataset], mapping from aligned dataset names
      to an aligned SubDataset, with n x d input x, n x m evals y and non-empty
      aligned field with concatenated m study group names and aligned_suffix.
    """
    aligned_dataset = {}
    trials = trials[trials['aligned']]
    for aligned_suffix in trials['aligned_suffix'].unique():
        aligned_trials = trials[trials['aligned_suffix'] == aligned_suffix]
        aligned_groups = aligned_trials[study_identifier].unique()
        pivot_df = aligned_trials.pivot(
            index=labels[:-1], columns=study_identifier, values=labels[-1])
        nan_groups = [
            c for c in pivot_df.columns if pivot_df[c].isna().values.any()
        ]
        combnum = min(3, len(nan_groups) + 1, len(aligned_groups) - 1)
        for groups in itertools.chain(
                *[itertools.combinations(nan_groups, r) for r in range(combnum)]):
            remain_groups = [sg for sg in aligned_groups if sg not in groups]
            if groups:
                index = np.all([pivot_df[sg].isnull() for sg in groups], axis=0)
                sub_df = pivot_df.loc[index, remain_groups].dropna().reset_index()
            else:
                sub_df = pivot_df.dropna().reset_index()
            if sub_df.shape[0] > 0:
                if verbose:
                    print('removed groups: ', groups)
                    print('remaining groups: ', remain_groups)
                    print('sub_df: ', sub_df.shape)
                aligned_key = ';'.join(list(groups) + [aligned_suffix])

                key, subkey = jax.random.split(key, 2)
                sub_df = sample_dataframe(subkey, sub_df, p_remove=p_remove)

                xx = jnp.array(sub_df[labels[:-1]])
                yy = jnp.array(sub_df[remain_groups])
                aligned_dataset[aligned_key] = SubDataset(
                    x=xx, y=yy, aligned=';'.join(remain_groups + [aligned_suffix]))
    msg = f'aligned dataset: {jax.tree_map(jnp.shape, aligned_dataset)}'
    logging.info(msg=msg)
    if verbose:
        print(msg)
    return aligned_dataset


def get_dataset(trials, study_identifier, labels, verbose=True):
    """Get dataset from a dataframe.

    Args:
      trials: pandas.DataFrame that stores all the trials.
      study_identifier: a label that uniquely identifies each study group.
      labels: labels of parameters and an eval metric (the last one).
      verbose: print info about data if True.

    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    """
    study_groups = trials[study_identifier].unique()
    dataset = {}
    for sg in study_groups:
        study_trials = trials.loc[trials[study_identifier] == sg, labels]
        xx = jnp.array(study_trials[labels[:-1]])
        yy = jnp.array(study_trials[labels[-1:]])
        dataset[sg] = SubDataset(x=xx, y=yy)
    msg = f'dataset before align: {jax.tree_map(jnp.shape, dataset)}'
    logging.info(msg)
    if verbose:
        print(msg)
    return dataset


def sample_sub_dataset(key,
                       trials,
                       study_identifier,
                       labels,
                       p_observed=0.,
                       verbose=True,
                       sub_dataset_key=None):
    """Sample a sub-dataset from trials dataframe.

    Args:
      key: random state for jax.random.
      trials: pandas.DataFrame that stores all the trials.
      study_identifier: a label that uniquely identifies each study group.
      labels: labels of parameters and an eval metric (the last one).
      p_observed: percentage of data that is observed.
      verbose: print info about data if True.
      sub_dataset_key: sub_dataset name to be queried.

    Returns:
      trials: remaining trials after removing the sampled sub-dataset.
      sub_dataset_key: study group key for testing in dataset.
      queried_sub_dataset: SubDataset to be queried.
    """
    test_study_key, observed_key = jax.random.split(key, 2)
    if sub_dataset_key is None:
        study_groups = trials[study_identifier].unique()
        sub_dataset_id = jax.random.choice(test_study_key, len(study_groups))
        sub_dataset_key = study_groups[sub_dataset_id]
    else:
        study_groups = trials[study_identifier].unique()
        if sub_dataset_key not in study_groups:
            raise ValueError(f'{sub_dataset_key} must be in dataframe.')

    queried_trials = trials[trials[study_identifier] == sub_dataset_key].sample(
        frac=1. - p_observed, replace=False, random_state=observed_key[0])
    trials = trials.drop(queried_trials.index)

    xx = jnp.array(queried_trials[labels[:-1]])
    yy = jnp.array(queried_trials[labels[-1:]])
    queried_sub_dataset = SubDataset(x=xx, y=yy)

    msg = (f'removed study={sub_dataset_key}  '
           f'removed study shape: x-{queried_sub_dataset.x.shape}, '
           f'y-{queried_sub_dataset.y.shape}')
    logging.info(msg)
    if verbose:
        print(msg)

    return trials, sub_dataset_key, queried_sub_dataset


def process_dataframe(
        key,
        trials,
        study_identifier,
        labels,
        p_observed=0.,
        maximize_metric=True,
        warp_func=None,
        verbose=True,
        sub_dataset_key=None,
        num_remove=0,
        p_remove=0.,
):
    """Process a dataframe and return needed info for an experiment.

    Args:
      key: random state for jax.random or sub_dataset_key to be queried.
      trials: pandas.DataFrame that stores all the trials.
      study_identifier: a label that uniquely identifies each study group.
      labels: labels of parameters and an eval metric (the last one).
      p_observed: percentage of data that is observed.
      maximize_metric: a boolean indicating if higher values of the eval metric
        are better or not. If maximize_metric is False and there is no warping for
        the output label, we negate all outputs.
      warp_func: mapping from label names to warping functions.
      verbose: print info about data if True.
      sub_dataset_key: sub_dataset name to be queried.
      num_remove: number of sub-datasets to remove.
      p_remove: proportion of data to be removed.

    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
      sub_dataset_key: study group key for testing in dataset.
      queried_sub_dataset: SubDataset to be queried.
    """
    trials = trials[[study_identifier] + labels +
                    ['aligned', 'aligned_suffix']].copy(deep=True)
    trials = trials.dropna()

    if verbose:
        print('trials: ', trials.shape)
    if not warp_func:
        warp_func = {}
    logging.info(msg=f'warp_func = {warp_func}')
    if labels[-1] not in warp_func and not maximize_metric:
        warp_func[labels[-1]] = lambda x: -x
    for la, fun in warp_func.items():
        if la in labels:
            trials.loc[:, la] = fun(trials.loc[:, la])
    assert len(trials) == len(trials.dropna()), ('nan appeared after applying '
                                                 f'warp_func={warp_func}')
    key, subkey = jax.random.split(key)
    trials, sub_dataset_key, queried_sub_dataset = sample_sub_dataset(
        key=subkey,
        trials=trials,
        study_identifier=study_identifier,
        labels=labels,
        p_observed=p_observed,
        verbose=verbose,
        sub_dataset_key=sub_dataset_key)

    for _ in range(num_remove):
        key, subkey = jax.random.split(key)
        removed_sub_dataset_key = None
        sub_dataset_key_split = sub_dataset_key.split(',')
        if len(sub_dataset_key_split) > 1:
            task_dataset_name = sub_dataset_key_split[1]
            study_groups = trials[study_identifier].unique()
            for s in study_groups:
                if task_dataset_name in s:
                    removed_sub_dataset_key = s
        trials, _, _ = sample_sub_dataset(
            key=subkey,
            trials=trials,
            study_identifier=study_identifier,
            labels=labels,
            p_observed=p_observed,
            verbose=verbose,
            sub_dataset_key=removed_sub_dataset_key)
        if trials.empty:
            raise ValueError(
                f'All datapoints are removed. Is num_remove={num_remove} too large?')
    key, subkey = jax.random.split(key)
    aligned_dataset = get_aligned_dataset(
        trials=trials,
        study_identifier=study_identifier,
        labels=labels,
        key=subkey,
        p_remove=p_remove,
        verbose=verbose)
    key, subkey = jax.random.split(key)
    trials = sample_dataframe(subkey, trials, p_remove=p_remove)

    dataset = get_dataset(
        trials=trials,
        study_identifier=study_identifier,
        labels=labels,
        verbose=verbose)

    dataset.update(aligned_dataset)
    return dataset, sub_dataset_key, queried_sub_dataset


def pd1(key,
        p_observed,
        verbose=True,
        sub_dataset_key=None,
        input_warp=True,
        output_log_warp=True,
        num_remove=0,
        metric_name='best_valid/error_rate',
        p_remove=0.,
        data_files=None):
    """Load PD1(Nesterov) from init2winit and pick a random study as test function.

    This function loads the PD1 dataset and reorganize it for experiments.
    For matched dataframes, we set `aligned` to True in its trials and reflect it
    in its corresponding SubDataset.
    The `aligned` value and sub-dataset key has suffix aligned_suffix, which is
    its phase identifier.

    Args:
      key: random state for jax.random.
      p_observed: percentage of test sub-dataset that is observed and included in
        the training dataset.
      verbose: print info about data if True.
      sub_dataset_key: sub-dataset name to be queried; if it is None, a random
        sub-dataset is selected as the sub-dataset to be queried.
      input_warp: apply input warping to data if True.
      output_log_warp: use log warping for output if True.
      num_remove: number of sub-datasets to be removed in the training dataset.
        This parameter should be used for holding out training tasks.
      metric_name: name of metric; default is 'best_valid/error_rate'.
      p_remove: proportion of data to be removed in each training sub-dataset.
      data_files: a dict mapping data descriptions to files. See PD1 for an
        example.

    Returns:
      dataset: training dataset that can be used to pre-train a Gaussian process,
        in the format of Dict[str, SubDataset], mapping from study group to a
        SubDataset.
      sub_dataset_key: study group (sub-dataset) key for testing, which can be a
        key in dataset if data from this test sub-dataset is used for model
        training.
      queried_sub_dataset: SubDataset to be queried; also known as test
        sub-dataset.
    """
    if data_files is None:
        data_files = PD1.copy()
    all_trials = []
    for k, v in data_files.items():
        if 'pkl' in v:
            with gfile.GFile(v, 'rb') as f:
                trials = pickle.load(f)
                trials.loc[:, 'aligned'] = (k[1] == 'matched')
                trials.loc[:, 'aligned_suffix'] = k[0]
                all_trials.append(trials)
        else:
            with gfile.GFile(v, 'r') as f:
                trials = pd.read_json(
                    f, orient='records', lines=True, precise_float=True)
                trials.loc[:, 'aligned'] = (k[1] == 'matched')
                trials.loc[:, 'aligned_suffix'] = k[0]
                all_trials.append(trials)
    trials = pd.concat(all_trials)
    trials = trials.reset_index(drop=True)
    labels = [
        'hps.lr_hparams.decay_steps_factor', 'hps.lr_hparams.initial_value',
        'hps.lr_hparams.power', 'hps.opt_hparams.momentum', metric_name
    ]
    warp_func = {}
    if input_warp:
        warp_func = {
            'hps.opt_hparams.momentum': lambda x: np.log(1 - x),
            'hps.lr_hparams.initial_value': np.log,
        }
    if output_log_warp:
        warp_func['best_valid/error_rate'] = lambda x: -np.log(x + 1e-10)

    return process_dataframe(
        key=key,
        trials=trials,
        study_identifier='study_group',
        labels=labels,
        p_observed=p_observed,
        maximize_metric=False,
        warp_func=warp_func if input_warp else None,
        verbose=verbose,
        sub_dataset_key=sub_dataset_key,
        num_remove=num_remove,
        p_remove=p_remove)


def _deduplicate(x, y, dataset_name, verbose=True):
    """Deduplicates x values, keeping the ones with highest y."""
    # Sort by decreasing Y values: deduplication keeps points with best rewards.
    sorted_xy = list(zip(*sorted(zip(x, y), key=lambda xy: xy[1], reverse=True)))
    x = np.array(sorted_xy[0])
    y = np.array(sorted_xy[1])
    _, idx = np.unique(x, axis=0, return_index=True)
    if verbose:
        print(
            f'Removed {x.shape[0] - len(idx)} duplicated points from {dataset_name}'
        )
    return x[idx, :], y[idx, :]


def _normalize_maf_dataset(maf_dataset, num_hparams, neg_error_to_accuracy):
    """Project the hparam values to [0, 1], and optionally convert y values.

    Args:
      maf_dataset: a dictionary of the format `{subdataset_name: {"X": ...,
        "Y":...}}`.
      num_hparams: the number of different hyperparameters being optimized.
      neg_error_to_accuracy: whether to transform y values that correspond to
        negative error rates to accuracy values.

    Returns:
      `maf_dataset` with normalized "X", and maybe "Y", values.
    """
    min_vals = np.ones(num_hparams) * np.inf
    max_vals = -np.ones(num_hparams) * np.inf

    for k, subdataset in maf_dataset.items():
        min_vals = np.minimum(min_vals, np.min(subdataset['X'], axis=0))
        max_vals = np.maximum(max_vals, np.max(subdataset['X'], axis=0))

    for k in maf_dataset:
        maf_dataset[k]['X'] = (maf_dataset[k]['X'] - min_vals) / (
                max_vals - min_vals)

        if neg_error_to_accuracy:
            maf_dataset[k]['Y'] = 1 + maf_dataset[k]['Y']
    return maf_dataset


def process_pd1_for_maf(outfile_path,
                        min_num_points,
                        input_warp,
                        output_log_warp,
                        neg_error_to_accuracy,
                        enforce_same_size_subdatasets,
                        verbose=True):
    """Store the pd1 dataset on cns in a MAF baseline-friendly format.

    Args:
      outfile_path: cns path where the pickled output is stored.
      min_num_points: Minimum number of points that must be in a subdataset to
        keep it.
      input_warp: apply warping to data if True.
      output_log_warp: use log warping for output.
      neg_error_to_accuracy: whether to transform y values that correspond to
        negative error rates to accuracy values. Cannot be True if output_log_warp
        is True.
      enforce_same_size_subdatasets: whether to to only keep `n` points of each
        subdataset, where `n` is the size of the smallest remaining dataset.
      verbose: print deduplication info about if True.
    """
    if output_log_warp and neg_error_to_accuracy:
        raise ValueError('Cannot transform y-values when the pd1 outputs are '
                         'log-warped!')

    key = jax.random.PRNGKey(0)
    dataset, _, _ = pd1(
        key, p_observed=1, input_warp=input_warp, output_log_warp=output_log_warp)

    num_hparams = dataset[list(dataset.keys())[0]].x.shape[1]
    excluded_subdatasets = ['imagenet_resnet50,imagenet,resnet,resnet50,1024']

    # Load and deduplicate data.
    maf_dataset = {}
    for k, subdataset in dataset.items():
        if subdataset.aligned is None and k not in excluded_subdatasets:
            x, y = _deduplicate(
                np.array(subdataset.x),
                np.array(subdataset.y),
                dataset_name=k,
                verbose=verbose)
            if x.shape[0] > min_num_points:
                maf_dataset[k] = dict(X=x, Y=y)

    if enforce_same_size_subdatasets:
        min_subdataset_size = min(
            [maf_dataset[k]['X'].shape[0] for k in maf_dataset])
        for k, subdataset in maf_dataset.items():
            x, y = maf_dataset[k]['X'], maf_dataset[k]['Y']
            maf_dataset[k] = dict(
                X=x[:min_subdataset_size, :], Y=y[:min_subdataset_size, :])
    maf_dataset = _normalize_maf_dataset(
        maf_dataset,
        num_hparams=num_hparams,
        neg_error_to_accuracy=neg_error_to_accuracy)

    data_utils.log_dataset(maf_dataset)
    with gfile.Open(outfile_path, 'wb') as f:
        pickle.dump(maf_dataset, f, pickle.HIGHEST_PROTOCOL)


def get_output_warper(output_log_warp=True, return_warping=False):
    """Returns an output warper with the option to use -log on 1-y."""
    if output_log_warp:

        def output_warping(f):

            def warpped_f(x_array):
                y = f(x_array)
                if not np.all(y <= 1. + 1e-11):
                    raise ValueError(f'Use output_log_warp only if f({x_array})={y} '
                                     'is smaller than or equal to 1.')
                ret = -np.log(1. + 1e-6 - y)
                assert np.all(np.isfinite(ret)), (f'y={y} caused ret={ret}.')
                return ret

            return warpped_f
    else:
        output_warping = lambda f: f
    output_warper = output_warping(lambda x: x)
    if return_warping:
        return output_warper, output_warping
    return output_warper


def hpob_dataset(search_space_index,
                 test_dataset_id_index,
                 test_seed,
                 output_log_warp=True):
    """Load the original finite hpob dataset by search space and test dataset id.

    To use the hpob dataset, download data and import hpob_handler from
      https://github.com/releaunifreiburg/HPO-B.

    Args:
      search_space_index: int index or string of a search space.
      test_dataset_id_index: int index or string of test dataset.
      test_seed: Identifier of the seed for the evaluation. Options: test0, test1,
        test2, test3, test4.
      output_log_warp: log warp on output with max assumed to be 1.

    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
      sub_dataset_key: study group key for testing in dataset.
      queried_sub_dataset: SubDataset to be queried.
    """
    # pylint: disable=g-bad-import-order,g-import-not-at-top
    from hpob import hpob_handler
    # pylint: enable=g-bad-import-order,g-import-not-at-top
    handler = hpob_handler.HPOBHandler(root_dir=HPOB_ROOT_DIR, mode='v3')
    if isinstance(search_space_index, str):
        search_space = search_space_index
    elif isinstance(search_space_index, int):
        spaces = list(handler.meta_train_data.keys())
        spaces.sort()
        search_space = spaces[search_space_index]
    else:
        raise ValueError('search_space_index must be str or int.')
    if isinstance(test_dataset_id_index, str):
        test_dataset_id = test_dataset_id_index
    elif isinstance(test_dataset_id_index, int):
        test_dataset_ids = list(handler.meta_test_data[search_space].keys())
        test_dataset_ids.sort()
        test_dataset_id = test_dataset_ids[test_dataset_id_index]
    else:
        raise ValueError('test_dataset_id_index must be str or int.')
    dataset = {}
    output_warper = get_output_warper(output_log_warp)

    for dataset_id in handler.meta_train_data[search_space]:
        train_x = np.array(handler.meta_train_data[search_space][dataset_id]['X'])
        train_y = np.array(handler.meta_train_data[search_space][dataset_id]['y'])
        train_y = output_warper(train_y)
        dataset[dataset_id] = SubDataset(x=train_x, y=train_y)
    if test_seed in ['test0', 'test1', 'test2', 'test3', 'test4']:
        init_index = handler.bo_initializations[search_space][test_dataset_id][
            test_seed]
        test_x = np.array(
            handler.meta_test_data[search_space][test_dataset_id]['X'])
        test_y = np.array(
            handler.meta_test_data[search_space][test_dataset_id]['y'])
        if output_log_warp:
            # if output_log_warp is True, warp the clipped init y without surrogate.
            test_y = output_warper(test_y)
        dataset[test_dataset_id] = SubDataset(
            x=test_x[init_index], y=test_y[init_index])
    else:
        test_x = np.empty((0, train_x.shape[1]))
        test_y = np.empty((0, 1))
    return dataset, test_dataset_id, SubDataset(x=test_x, y=test_y)


NORMALIAZTION_EPS = 1e-12


def hpob_dataset_v2(search_space_index, hpob_data_path=None, negative_y=False, output_log_warp=False, normalize_x=True,
                    normalize_y=True):
    """Load the original finite hpob dataset by search space.
    Args:
      search_space_index: string of a search space. See https://arxiv.org/pdf/2106.06257.pdf Table 3 for reference.
      output_log_warp: log warp on output with max assumed to be 1.
    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    """

    # Make sure the hpob folder is also in your directory
    from hyperbo.bo_utils import hpob_handler

    handler = hpob_handler.HPOBHandler(root_dir=hpob_data_path, mode='v2')

    if isinstance(search_space_index, str):
        search_space = search_space_index
    else:
        raise ValueError('search_space_index must be str.')

    dataset_ids = list(handler.meta_test_data[search_space].keys())

    dataset = {}
    output_warper = get_output_warper(output_log_warp)

    for dataset_id in dataset_ids:
        x = np.array(handler.meta_test_data[search_space][dataset_id]['X'])
        y = np.array(handler.meta_test_data[search_space][dataset_id]['y'])

        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        if negative_y:
            y *= -1
        if output_log_warp:
            y = output_warper(y)

        dataset[dataset_id] = SubDataset(x=x, y=y)

    return dataset


def get_pd1_dataset(pd1_data_path, normalize_x=True, normalize_y=True):
    key = jax.random.PRNGKey(0)
    pd1_data_files = {
        ('phase0', 'matched'): os.path.join(pd1_data_path, 'pd1_matched_phase0_results.jsonl'),
        ('phase1', 'matched'): os.path.join(pd1_data_path, 'pd1_matched_phase1_results.jsonl'),
        ('phase0', 'unmatched'): os.path.join(pd1_data_path, 'pd1_unmatched_phase0_results.jsonl'),
        ('phase1', 'unmatched'): os.path.join(pd1_data_path, 'pd1_unmatched_phase1_results.jsonl'),
    }
    dataset, sub_dataset_key, queried_subdataset = pd1(
        key, p_observed=0, sub_dataset_key='wmt15_de_en_xfmr,translate_wmt,xformer_translate,xformer,64',
        data_files=pd1_data_files)
    dataset[sub_dataset_key] = queried_subdataset
    new_dataset = {}  # all sub-datasets in PD1
    for k, v in dataset.items():
        if v.aligned is None:
            x = v.x
            y = v.y
            if normalize_x:
                x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
                x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
            if normalize_y:
                y_max, y_min = np.max(y), np.min(y)
                y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
            new_dataset[k] = SubDataset(x=x, y=y)

    # split train and test
    all_sd_keys = list(new_dataset.keys())

    # use the very small sub-dataset for training
    train_sd_keys = ['imagenet_resnet50,imagenet,resnet,resnet50,1024']
    test_sd_keys = []
    all_sd_keys.remove('imagenet_resnet50,imagenet,resnet,resnet50,1024')

    test_sd_keys_idx = jax.random.choice(key, len(all_sd_keys), shape=(math.ceil(0.2 * len(all_sd_keys)),),
                                         replace=False)
    for i, key in enumerate(all_sd_keys):
        if i in test_sd_keys_idx:
            test_sd_keys.append(key)
        else:
            train_sd_keys.append(key)

    train_dataset = {}
    for key in train_sd_keys:
        train_dataset[key] = new_dataset[key]
    test_dataset = {}
    for key in test_sd_keys:
        test_dataset[key] = new_dataset[key]

    return {'train': train_dataset, 'test': test_dataset}


def hpob_dataset_v3(search_space_index, hpob_data_path=None, pd1_data_path=None, negative_y=False,
                    output_log_warp=False, normalize_x=True, normalize_y=True, read_init_index=False):
    """Load the original finite hpob dataset by search space.
    Args:
      search_space_index: string of a search space. See https://arxiv.org/pdf/2106.06257.pdf Table 3 for reference.
      output_log_warp: log warp on output with max assumed to be 1.
    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    """

    if search_space_index == 'pd1':
        return get_pd1_dataset(pd1_data_path, normalize_x=normalize_x, normalize_y=normalize_y)

    # Make sure the hpob folder is also in your directory
    from hyperbo.bo_utils import hpob_handler

    handler = hpob_handler.HPOBHandler(root_dir=hpob_data_path, mode='v3')

    if isinstance(search_space_index, str):
        search_space = search_space_index
    else:
        raise ValueError('search_space_index must be str.')

    train_dataset_ids = list(handler.meta_train_data[search_space].keys())
    validation_dataset_ids = list(handler.meta_validation_data[search_space].keys())
    test_dataset_ids = list(handler.meta_test_data[search_space].keys())

    output_warper = get_output_warper(output_log_warp)

    train_dataset = {}
    for dataset_id in train_dataset_ids:
        x = np.array(handler.meta_train_data[search_space][dataset_id]['X'])
        y = np.array(handler.meta_train_data[search_space][dataset_id]['y'])

        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        if negative_y:
            y *= -1
        if output_log_warp:
            y = output_warper(y)

        train_dataset[dataset_id] = SubDataset(x=x, y=y)

    validation_dataset = {}
    for dataset_id in validation_dataset_ids:
        x = np.array(handler.meta_validation_data[search_space][dataset_id]['X'])
        y = np.array(handler.meta_validation_data[search_space][dataset_id]['y'])

        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        if negative_y:
            y *= -1
        if output_log_warp:
            y = output_warper(y)

        validation_dataset[dataset_id] = SubDataset(x=x, y=y)

    test_dataset = {}
    for dataset_id in test_dataset_ids:
        x = np.array(handler.meta_test_data[search_space][dataset_id]['X'])
        y = np.array(handler.meta_test_data[search_space][dataset_id]['y'])

        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        if negative_y:
            y *= -1
        if output_log_warp:
            y = output_warper(y)

        test_dataset[dataset_id] = SubDataset(x=x, y=y)

    if read_init_index:
        init_index = handler.bo_initializations[search_space]
    else:
        init_index = None

    return {'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset, 'init_index': init_index}


def random(key,
           mean_func,
           cov_func,
           params,
           dim,
           n_observed,
           n_queries,
           n_func_historical=0,
           m_points_historical=0,
           warp_func=None):
    """Generate random historical data and observed data for current function.

    Args:
      key: random state for jax.random.
      mean_func: mean function handle that maps from (params, n x d input,
        warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
        more details).
      cov_func: covariance function handle that maps from (params, n1 x d input1,
        n2 x d input2, wrap_func) to a n1 x n2 covariance matrix (see matrix_map
        in kernel.py for more details).
      params: parameters for covariance, mean, and noise variance.
      dim: input dimension d.
      n_observed: number of observed data points for the queried function.
      n_queries: total number of data points that can be queried.
      n_func_historical: number of historical functions. These functions are
        observed before bayesopt on the new queried function.
      m_points_historical: number of points for each historical function.
      warp_func: optional dictionary that specifies the warping function for each
        parameter.

    Returns:
      dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
      sub_dataset_key: study group key for testing in dataset.
      queried_sub_dataset: SubDataset to be queried.
    """
    x_key, y_key, historical_key = jax.random.split(key, 3)

    # Generate historical data.
    hist_keys = jax.random.split(historical_key, n_func_historical)
    dataset = {}
    for i in range(n_func_historical):
        x_hist_key, y_hist_key = jax.random.split(hist_keys[i], 2)
        vx = jax.random.uniform(x_hist_key, (m_points_historical, dim))
        vy = gp.sample_from_gp(
            y_hist_key, mean_func, cov_func, params, vx, warp_func=warp_func)
        dataset[i] = SubDataset(x=vx, y=vy)

    # Generate observed data and possible queries for queried function.
    vx = jax.random.uniform(x_key, (n_observed + n_queries, dim))
    vy = gp.sample_from_gp(
        y_key, mean_func, cov_func, params, vx, warp_func=warp_func)
    x_queries, x_observed = vx[:n_queries], vx[n_queries:]
    y_queries, y_observed = vy[:n_queries], vy[n_queries:]
    dataset[n_func_historical] = SubDataset(x=x_observed, y=y_observed)
    queried_sub_dataset = SubDataset(x=x_queries, y=y_queries)
    return dataset, n_func_historical, queried_sub_dataset


# isotropic assumption
def generate_param_from_prior(key, num_samples, prior_params, prior_type='gamma'):
    if prior_type == 'gamma':
        gamma_alpha, gamma_beta = prior_params['alpha'], prior_params['beta']
        gamma = Gamma(gamma_alpha, gamma_beta)
        thetas = gamma.sample(num_samples, seed=key)
    elif prior_type == 'normal':
        normal_mu, normal_sigma = prior_params['mu'], prior_params['sigma']
        normal = Normal(normal_mu, normal_sigma)
        thetas = normal.sample(num_samples, seed=key)
    elif prior_type == 'lognormal':
        lognormal_mu, lognormal_sigma = prior_params['mu'], prior_params['sigma']
        lognormal = LogNormal(lognormal_mu, lognormal_sigma)
        thetas = lognormal.sample(num_samples, seed=key)
    else:
        raise ValueError('prior_type {} not supported.'.format(prior_type))
    return thetas


def generate_synthetic_data(key, n_search_space, cov_func, mean_func, constants, lengthscales, sig_vars, noise_vars,
                            n_funcs, n_func_dims, n_discrete_points):
    # discrete domain of test functions
    super_dataset = {}
    gp_params = {}

    for i in range(n_search_space):
        dataset_i = {}
        constant = constants[i]
        lengthscale = lengthscales[i]
        sig_var = sig_vars[i]
        noise_var = noise_vars[i]

        model_params_i = {
            'constant': constant,
            'lengthscale': lengthscale,
            'signal_variance': sig_var,
            'noise_variance': noise_var
        }
        params_i = defs.GPParams(model=model_params_i)

        for j in range(n_funcs[i]):
            new_key, key = jax.random.split(key)
            vx = jax.random.uniform(new_key, (n_discrete_points[i], n_func_dims[i]))

            new_key, key = jax.random.split(key)
            vy = gp.sample_from_gp(new_key, mean_func, cov_func, params_i, vx, num_samples=1)
            dataset_i[str(j)] = SubDataset(x=vx, y=vy)
        super_dataset[str(i)] = dataset_i
        gp_params[str(i)] = params_i

    return super_dataset, gp_params


# n_funcs: function # per search space
def gen_synthetic_super_dataset_bohss(
        key, n_search_space, n_funcs, n_func_dims, n_discrete_points, cov_func, mean_func, const_params,
        ls_linear_params, sig_var_params, noise_var_params, const_prior='normal', ls_prior='gamma',
        sig_var_prior='gamma', noise_var_prior='gamma'):
    new_key, key = jax.random.split(key)
    constants = generate_param_from_prior(new_key, n_search_space, const_params, const_prior)

    lengthscales = []
    for i in range(n_search_space):
        new_key, key = jax.random.split(key)
        if ls_prior == 'gamma':
            alpha = ls_linear_params['alpha_w'] * n_func_dims[i] + ls_linear_params['alpha_b']
            beta = ls_linear_params['beta_w'] * n_func_dims[i] + ls_linear_params['beta_b']
            lengthscale_params = {'alpha': alpha, 'beta': beta}
        elif ls_prior == 'lognormal':
            mu = ls_linear_params['mu_w'] * n_func_dims[i] + ls_linear_params['mu_b']
            sigma = ls_linear_params['sigma_w'] * n_func_dims[i] + ls_linear_params['sigma_b']
            lengthscale_params = {'mu': mu, 'sigma': sigma}
        else:
            raise ValueError('prior_type {} not supported.'.format(ls_prior))
        lengthscale = generate_param_from_prior(new_key, n_func_dims[i], lengthscale_params, ls_prior)
        lengthscales.append(lengthscale)

    new_key, key = jax.random.split(key)
    sig_vars = generate_param_from_prior(new_key, n_search_space, sig_var_params, sig_var_prior)

    new_key, key = jax.random.split(key)
    noise_vars = generate_param_from_prior(new_key, n_search_space, noise_var_params, noise_var_prior)

    new_key, key = jax.random.split(key)
    super_dataset, gp_params = generate_synthetic_data(key, n_search_space, cov_func, mean_func, constants,
                                                       lengthscales, sig_vars, noise_vars, n_funcs, n_func_dims,
                                                       n_discrete_points)
    return super_dataset, gp_params


def hpl_bo_synthetic_dataset_combined(synthetic_data_path, search_space_index, normalize_x=False,
                                            normalize_y=False):
    dataset_all = np.load(synthetic_data_path, allow_pickle=True).item()

    dataset = dataset_all[search_space_index]

    for sub_dataset_key in dataset.keys():
        x = dataset[sub_dataset_key].x
        y = dataset[sub_dataset_key].y
        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        dataset[sub_dataset_key] = SubDataset(x=x, y=y)

    return dataset


def hpl_bo_synthetic_dataset_split(synthetic_data_path, search_space_index, normalize_x=False, normalize_y=False):
    dataset_all = np.load(synthetic_data_path, allow_pickle=True).item()
    dataset_i = dataset_all[search_space_index]
    train_subdatasets = {}
    test_subdatasets = {}
    keys = list(dataset_i.keys())
    n_train = int(len(keys) * 0.8)
    for key in keys[:n_train]:
        train_subdatasets[key] = dataset_i[key]
    for key in keys[n_train:]:
        test_subdatasets[key] = dataset_i[key]

    for sub_dataset_key in train_subdatasets.keys():
        x = train_subdatasets[sub_dataset_key].x
        y = train_subdatasets[sub_dataset_key].y
        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        train_subdatasets[sub_dataset_key] = SubDataset(x=x, y=y)

    for sub_dataset_key in test_subdatasets.keys():
        x = test_subdatasets[sub_dataset_key].x
        y = test_subdatasets[sub_dataset_key].y
        if normalize_x:
            x_max, x_min = np.max(x, axis=0), np.min(x, axis=0)
            x = (x - x_min) / (x_max - x_min + NORMALIAZTION_EPS)
        if normalize_y:
            y_max, y_min = np.max(y), np.min(y)
            y = (y - y_min) / (y_max - y_min + NORMALIAZTION_EPS)
        test_subdatasets[sub_dataset_key] = SubDataset(x=x, y=y)

    return {'train': train_subdatasets, 'test': test_subdatasets}


# for old file loading only
def hyperbo_plus_synthetic_dataset_combined(synthetic_data_path, search_space_index, normalize_x=False,
                                            normalize_y=False):
    pass


# for old file loading only
def hyperbo_plus_synthetic_dataset_split(synthetic_data_path, search_space_index, normalize_x=False, normalize_y=False):
    pass
