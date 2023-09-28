import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tensorflow_probability.substrates.jax.distributions import Gamma
from experiment_defs import *
import jax.numpy as jnp
from hpl_bo_utils import get_gp_priors_from_direct_hgp, get_gp_priors_from_hpl_hgp, get_gp_priors_from_gt_hgp


def plot_performance_curve(ax, label, regrets_all, time_list, n_init_obs=0, log_plot=False, color=None, linestyle=None):
    regrets_all = regrets_all[:, n_init_obs:]

    regrets_middle = np.mean(regrets_all, axis=0)
    err_low = regrets_middle - np.std(regrets_all, axis=0)
    err_high = regrets_middle + np.std(regrets_all, axis=0)

    ax.set_xlim([time_list[0], time_list[-1]])

    if log_plot:
        if color:
            line = ax.semilogy(time_list, regrets_middle, label=label, color=color, linestyle=linestyle)[0]
        else:
            line = ax.semilogy(time_list, regrets_middle, label=label, linestyle=linestyle)[0]
    else:
        if color:
            line = ax.plot(time_list, regrets_middle, label=label, color=color, linestyle=linestyle)[0]
        else:
            line = ax.plot(time_list, regrets_middle, label=label, linestyle=linestyle)[0]
    ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


def plot_performance_curve_percentile(ax, label, regrets_all, time_list, n_init_obs=0, log_plot=False):
    regrets_all = regrets_all[:, n_init_obs:]

    regrets_middle = np.median(regrets_all, axis=0)
    err_low = np.percentile(regrets_all, 25, axis=0)
    err_high = np.percentile(regrets_all, 75, axis=0)

    if log_plot:
        line = ax.semilogy(time_list, regrets_middle, label=label)[0]
    else:
        line = ax.plot(time_list, regrets_middle, label=label)[0]
    ax.fill_between(time_list, err_low, err_high, alpha=0.2, color=line.get_color())


def plot_estimated_prior(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)
    for kernel in results['kernel_list']:
        kernel_results = results['kernel_results'][kernel]
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('BO iteration')
        ax.set_ylabel('average best sample simple regret')
        plot_performance_curve(ax, 'groudtruth', kernel_results['regrets_mean_groundtruth'],
                               kernel_results['regrets_std_groundtruth'], time_list)
        plot_performance_curve(ax, 'inferred', kernel_results['regrets_mean_inferred'],
                               kernel_results['regrets_std_inferred'], time_list)
        plot_performance_curve(ax, 'random', kernel_results['regrets_mean_random'],
                               kernel_results['regrets_std_random'], time_list)
        ax.legend()
        fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}.pdf'.format(kernel)))
        plt.close(fig)

        # visualize bo
        if results['visualize_bo']:
            visualize_bo_results = kernel_results['visualize_bo_results']
            n_visualize_grid_points = visualize_bo_results['n_visualize_grid_points']
            observations_groundtruth = visualize_bo_results['observations_groundtruth']
            observations_inferred = visualize_bo_results['observations_inferred']
            for i in range(results['budget']):
                # plot based on same observations
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_title('BO iteration = {} (same observations)'.format(i + 1))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.plot(visualize_bo_results['f_x'][:n_visualize_grid_points].squeeze(),
                        visualize_bo_results['f_y'][:n_visualize_grid_points].squeeze(), '--', label='f')
                ax.plot(visualize_bo_results['f_x'][n_visualize_grid_points:].squeeze(),
                        visualize_bo_results['f_y'][n_visualize_grid_points:].squeeze(), 'o', label='f_discrete')
                mean_groundtruth = visualize_bo_results['posterior_list'][i]['mean_groundtruth'].squeeze()
                std_groundtruth = visualize_bo_results['posterior_list'][i]['std_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_groundtruth[:n_visualize_grid_points], label='groundtruth')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] - std_groundtruth[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] + std_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_groundtruth[0][:i], observations_groundtruth[1][:i], 'o', color=line.get_color(),
                        label='obs_gt')
                mean_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i][
                    'mean_inferred_on_groundtruth'].squeeze()
                std_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i][
                    'std_inferred_on_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_inferred_on_groundtruth[:n_visualize_grid_points], label='inferred (on obs_gt)')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] - std_inferred_on_groundtruth[
                                                                                         :n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] + std_inferred_on_groundtruth[
                                                                                         :n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.legend()
                fig.savefig(
                    os.path.join(dir_path, 'regret_vs_iteration_{}_same_obs_iteration_{}.pdf'.format(kernel, i)))
                plt.close(fig)

                # plot based on different observations
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.set_title('BO iteration = {} (different observations)'.format(i + 1))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.plot(visualize_bo_results['f_x'][:n_visualize_grid_points].squeeze(),
                        visualize_bo_results['f_y'][:n_visualize_grid_points].squeeze(), '--', label='f')
                ax.plot(visualize_bo_results['f_x'][n_visualize_grid_points:].squeeze(),
                        visualize_bo_results['f_y'][n_visualize_grid_points:].squeeze(), 'o', label='f_discrete')
                mean_groundtruth = visualize_bo_results['posterior_list'][i]['mean_groundtruth'].squeeze()
                std_groundtruth = visualize_bo_results['posterior_list'][i]['std_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_groundtruth[:n_visualize_grid_points], label='groundtruth')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] - std_groundtruth[:n_visualize_grid_points],
                                mean_groundtruth[:n_visualize_grid_points] + std_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_groundtruth[0][:i], observations_groundtruth[1][:i], 'o', color=line.get_color(),
                        label='obs_gt')
                mean_inferred_on_inferred = visualize_bo_results['posterior_list'][i][
                    'mean_inferred_on_inferred'].squeeze()
                std_inferred_on_inferred = visualize_bo_results['posterior_list'][i][
                    'std_inferred_on_inferred'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_inferred_on_inferred[:n_visualize_grid_points],
                               label='inferred (on obs_inf)')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_inferred_on_inferred[:n_visualize_grid_points] - std_inferred_on_inferred[
                                                                                      :n_visualize_grid_points],
                                mean_inferred_on_inferred[:n_visualize_grid_points] + std_inferred_on_inferred[
                                                                                      :n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.plot(observations_inferred[0][:i], observations_inferred[1][:i], 'o', color=line.get_color(),
                        label='obs_inf')
                ax.legend()
                fig.savefig(
                    os.path.join(dir_path, 'regret_vs_iteration_{}_different_obs_iteration_{}.pdf'.format(kernel, i)))
                plt.close(fig)


method_name_list = ['random', 'base', 'hyperbo', 'ablr', 'fsbo', 'gt_hgp', 'hand_hgp', 'uniform_hgp', 'fit_direct_hgp',
                    'fit_direct_hgp_leaveout', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_leaveout', 'hpl_hgp_two_step',
                    'hpl_hgp_two_step_leaveout']

colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
    'black',
]

line_style_map_vary_num_dataset = {
    'fit_direct_hgp_4': ('tab:blue', '-'),
    'fit_direct_hgp_8': ('tab:orange', '-'),
    'fit_direct_hgp_12': ('tab:green', '-'),
    'fit_direct_hgp_16': ('tab:red', '-'),
    'hpl_hgp_two_step_4': ('tab:blue', '--'),
    'hpl_hgp_two_step_8': ('tab:orange', '--'),
    'hpl_hgp_two_step_12': ('tab:green', '--'),
    'hpl_hgp_two_step_16': ('tab:red', '--'),
    'hpl_hgp_end_to_end_4': ('tab:blue', ':'),
    'hpl_hgp_end_to_end_8': ('tab:orange', ':'),
    'hpl_hgp_end_to_end_12': ('tab:green', ':'),
    'hpl_hgp_end_to_end_16': ('tab:red', ':'),
}

line_style_map = {
    'random': ('tab:blue', '-'),
    'base': ('tab:orange', '-'),
    'hyperbo': ('tab:green', '-'),
    'fsbo': ('tab:red', '-'),
    'ablr': ('tab:purple', '-'),
    'gt_gp': ('black', '-'),
    'gt_hgp': ('tab:brown', '-'),
    'hand_hgp': ('tab:pink', '-'),
    'uniform_hgp': ('tab:gray', '-'),
    'log_uniform_hgp': ('tab:olive', '-'),
    'hpl_hgp_end_to_end': ('tab:blue', '--'),
    'hpl_hgp_end_to_end_leaveout': ('tab:orange', '--'),
    'hpl_hgp_two_step': ('tab:green', '--'),
    'hpl_hgp_two_step_leaveout': ('tab:red', '--'),
    'fit_direct_hgp': ('tab:purple', '--'),
    'fit_direct_hgp_leaveout': ('tab:brown', '--'),

    'hpl_hgp_end_to_end_from_scratch': ('tab:pink', '--'),
    'hpl_hgp_end_to_end_leaveout_from_scratch': ('tab:gray', '--'),
}


display_name_map = {
    'pi': 'PI',
    'ei': 'EI',
    'ucb': 'UCB',

    'random': 'Random',
    'base': 'Base GP',
    'hyperbo': 'HyperBO',
    'fsbo': 'FSBO',
    'ablr': 'ABLR',
    'gt_gp': 'Ground-truth GP',
    'gt_hgp': 'Ground-truth HGP',
    'hand_hgp': 'Hand-specified HGP',
    'uniform_hgp': 'Non-informative HGP',
    'log_uniform_hgp': 'Non-informative HGP (Log-uniform)',
    # 'hpl_hgp_end_to_end': 'MPHD Standard',
    # 'hpl_hgp_end_to_end_leaveout': 'MPHD Standard (NToT)',
    'hpl_hgp_two_step': 'MPHD Standard',
    'hpl_hgp_two_step_leaveout': 'MPHD Standard (NToT)',
    'fit_direct_hgp': 'MPHD Non-NN HGP',
    'fit_direct_hgp_leaveout': 'MPHD Non-NN HGP (NToT)',

    # 'hpl_hgp_end_to_end_from_scratch': 'MPHD End-to-end',
    # 'hpl_hgp_end_to_end_leaveout_from_scratch': 'MPHD End-to-end (NToT)',
}


def plot_lengthscale_distribution(results, dataset_name, dir_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax.set_xlabel('Length-scale')
    ax.set_ylabel('Frequency')
    ax2.set_ylabel('Probability Density')
    setup_b_id_list = results['setup_b_id_list']
    lengthscale_list = []
    for train_id in setup_b_id_list:
        lengthscale_list += [x for x in
                             np.array(results['setup_b']['fit_gp_params'][train_id]['gp_params']['lengthscale'])]

    bins = np.logspace(np.log10(min(lengthscale_list)), np.log10(max(lengthscale_list)), 50)
    counts, bins = np.histogram(lengthscale_list, bins=bins)
    # ax.hist(lengthscale_list, label='Estimated Length-scale Values')
    ax.stairs(counts, bins, label='Estimated Length-scale Values')

    learned_prior_a, learned_prior_b = results['setup_b']['gp_distribution_params']['lengthscale']
    x_range = (0, 15.0)
    x = np.linspace(x_range[0], x_range[1], 10000)
    dist = Gamma(learned_prior_a, learned_prior_b)
    ax2.plot(x, dist.prob(x), color='red', label='Learned Prior of Length-scale')
    fig.legend(bbox_to_anchor=(0.6, 0.95), loc='upper center', facecolor='white', framealpha=1)
    plt.tight_layout()
    fig.savefig(os.path.join(dir_path, 'plot_lengthscale_distribution_{}.pdf'.format(dataset_name)),
                bbox_inches='tight')
    plt.close(fig)


def get_display_name_vary_num_dataset(method_name, ac_func_type=None):
    if ac_func_type is None or method_name == 'random':
        return display_name_map[method_name]
    return '{} ({})'.format(display_name_map[method_name], display_name_map[ac_func_type]) + '_{}'


def plot_hpl_bo_vary_num_dataset(dir_path, dataset_name, log_plot, plot_ac_func_type, is_pd1=False,
                                 dataset_id=None, std_across_seed_dim=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    ax.set_xlabel('Number of BO Iterations')
    ax.set_ylabel('Average Normalized Simple Regret')

    results = np.load(os.path.join(dir_path, 'results.npy'), allow_pickle=True).item()
    for num_dataset in VARY_NUM_DATASET_NUM_LIST:
        configs = results['configs']
        time_list = range(1, configs['budget'] + 1)
        n_init_obs = configs['n_init_obs']
        n_bo_runs = configs['n_bo_runs']
        ac_func_type_list = configs['ac_func_type_list']

        if is_pd1:
            assert False
            method_name_list = VARY_NUM_DATASET_METHOD_NAME_LIST
            bo_results = results['setup_b']['bo_results_pd1_total']
        else:
            method_name_list = VARY_NUM_DATASET_METHOD_NAME_LIST
            if dataset_id is None:
                # bo_results = results['setup_b']['bo_results_total']
                bo_results = {}
                for method_name in method_name_list:
                    bo_results[method_name] = {}
                    for ac_func_type in ac_func_type_list:
                        regrets_all_list_multiple_seeds = []
                        for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                            extra_suffix = '_n{}seed{}'.format(num_dataset, seed)

                            regrets_all_list_seed = results['setup_b'][extra_suffix]['bo_results_total'][method_name][ac_func_type]['regrets_all_list']
                            if std_across_seed_dim:
                                regrets_all_list_multiple_seeds.append(jnp.mean(regrets_all_list_seed, axis=0))
                            else:
                                regrets_all_list_multiple_seeds.append(regrets_all_list_seed)

                        if std_across_seed_dim:
                            regrets_all_list = jnp.array(regrets_all_list_multiple_seeds)
                        else:
                            regrets_all_list = jnp.concatenate(regrets_all_list_multiple_seeds, axis=0)
                        regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
                        regrets_std_total = jnp.std(regrets_all_list, axis=0)

                        bo_results[method_name][ac_func_type] = {
                            'regrets_all_list': regrets_all_list,
                            'regrets_mean': regrets_mean_total,
                            'regrets_std': regrets_std_total,
                        }
            else:
                bo_results = {}
                for method_name in method_name_list:
                    bo_results[method_name] = {}
                    for ac_func_type in ac_func_type_list:
                        regrets_all_list_multiple_seeds = []
                        for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                            extra_suffix = '_n{}seed{}'.format(num_dataset, seed)

                            regrets_all_list_seed = []
                            for i in range(n_bo_runs):
                                regrets_ij = \
                                    results['setup_b'][extra_suffix]['bo_results'][method_name][ac_func_type][dataset_id]['regrets_list'][i]
                                regrets_all_list_seed.append(regrets_ij)
                            if std_across_seed_dim:
                                regrets_all_list_multiple_seeds.append(jnp.mean(jnp.array(regrets_all_list_seed), axis=0))
                            else:
                                regrets_all_list_multiple_seeds.append(jnp.array(regrets_all_list_seed))

                        if std_across_seed_dim:
                            regrets_all_list = jnp.array(regrets_all_list_multiple_seeds)
                        else:
                            regrets_all_list = jnp.concatenate(regrets_all_list_multiple_seeds, axis=0)
                        regrets_mean_total = jnp.mean(regrets_all_list, axis=0)
                        regrets_std_total = jnp.std(regrets_all_list, axis=0)

                        bo_results[method_name][ac_func_type] = {
                            'regrets_all_list': regrets_all_list,
                            'regrets_mean': regrets_mean_total,
                            'regrets_std': regrets_std_total,
                        }

        for method_name in method_name_list:
            if plot_ac_func_type == 'best':
                best_ac_func_type = None
                best_regret = float('inf')
                for ac_func_type in ac_func_type_list:
                    regrets_all = bo_results[method_name][ac_func_type]['regrets_all_list']
                    regrets_all = regrets_all[:, n_init_obs:]
                    regrets_middle = np.mean(regrets_all, axis=0)
                    if regrets_middle[-1] < best_regret:
                        best_ac_func_type = ac_func_type
                        best_regret = regrets_middle[-1]
                ac_func_type_to_use = best_ac_func_type
            else:
                ac_func_type_to_use = plot_ac_func_type
            plot_performance_curve(
                ax,
                get_display_name_vary_num_dataset(method_name, ac_func_type_to_use).format(num_dataset),
                bo_results[method_name][ac_func_type_to_use]['regrets_all_list'],
                time_list, n_init_obs,
                linestyle=line_style_map_vary_num_dataset[method_name + '_{}'.format(num_dataset)][1],
                color=line_style_map_vary_num_dataset[method_name + '_{}'.format(num_dataset)][0],
                log_plot=log_plot,
            )

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.2), facecolor='white',
               framealpha=1)
    fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}_log_{}_{}{}_{}.pdf'.format(
        dataset_name, log_plot, plot_ac_func_type,
        '_id_{}'.format(dataset_id) if dataset_id is not None else '',
        'std_across_seed_dim' if std_across_seed_dim else 'std_across_seed_and_subdataset',)),
                bbox_inches='tight'),
    plt.close(fig)


def plot_kl_vary_num_dataset(dir_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    ax.set_xlabel('Number of Training Datasets')
    ax.set_ylabel('Average KL Divergence')

    results = np.load(os.path.join(dir_path, 'results.npy'), allow_pickle=True).item()

    for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
        kl_mean_list = []
        kl_std_list = []
        kl_data_list = []
        for num_dataset in VARY_NUM_DATASET_NUM_LIST:
            kl_list_i = []
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                extra_suffix = '_n{}seed{}'.format(num_dataset, seed)
                kl_list_i.append(results['setup_b'][extra_suffix]['kl_eval'][method_name]['lengthscale']['mean'])
            kl_list_i = np.array(kl_list_i)
            kl_data_list.append(kl_list_i)
            kl_mean_list.append(np.mean(kl_list_i))
            kl_std_list.append(np.std(kl_list_i))

        kl_mean_list = np.array(kl_mean_list)
        kl_std_list = np.array(kl_std_list)

        label = display_name_map[method_name]
        if label == 'MPHD Non-NN HGP':
            label = 'Non-NN Prior'
        elif label == 'MPHD Standard':
            label = 'NN-based Prior'
        else:
            assert False
        color = line_style_map[method_name][0]
        linestyle = line_style_map[method_name][1]
        # line = ax.plot(VARY_NUM_DATASET_NUM_LIST, kl_mean_list, label=label, color=color, linestyle=linestyle)[0]
        # err_low = kl_mean_list - kl_std_list
        # err_high = kl_mean_list + kl_std_list
        # ax.fill_between(VARY_NUM_DATASET_NUM_LIST, err_low, err_high, alpha=0.2, color=line.get_color())

        parts = ax.violinplot(kl_data_list, positions=VARY_NUM_DATASET_NUM_LIST, showmeans=True, widths=0.5)
        print(parts.keys())
        color = parts['cbars'].get_color()
        '''
        for pc in parts['bodies']:
            # pc.set_facecolor(color)
            # pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        '''

        ax.plot(VARY_NUM_DATASET_NUM_LIST, kl_mean_list, label=label, color=color)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(dir_path, 'kl_vs_n_dataset_vary_num_datasets.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 20}
    axes = {}
    matplotlib.rc('font', **font)
    matplotlib.rc('axes', **axes)

    for distribution_type in ['gamma']:
        results_dir = 'fill in results dir for varying number of training datasets on the synthetic super-dataset'

        print(results_dir)

        dir_path = os.path.join(results_dir, 'merge')

        dataset_name = 'synthetic'
        plot_kl_vary_num_dataset(dir_path)
