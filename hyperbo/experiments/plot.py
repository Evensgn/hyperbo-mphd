from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import matplotlib
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma
import jax.numpy as jnp


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
                ax.plot(observations_groundtruth[0][:i], observations_groundtruth[1][:i], 'o', color=line.get_color(), label='obs_gt')
                mean_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i]['mean_inferred_on_groundtruth'].squeeze()
                std_inferred_on_groundtruth = visualize_bo_results['posterior_list'][i]['std_inferred_on_groundtruth'].squeeze()
                line = ax.plot(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                               mean_inferred_on_groundtruth[:n_visualize_grid_points], label='inferred (on obs_gt)')[0]
                ax.fill_between(visualize_bo_results['f_x'].squeeze()[:n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] - std_inferred_on_groundtruth[:n_visualize_grid_points],
                                mean_inferred_on_groundtruth[:n_visualize_grid_points] + std_inferred_on_groundtruth[:n_visualize_grid_points],
                                alpha=0.2, color=line.get_color())
                ax.legend()
                fig.savefig(os.path.join(dir_path, 'regret_vs_iteration_{}_same_obs_iteration_{}.pdf'.format(kernel, i)))
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


def plot_hyperbo_plus(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)
    ac_func_type_list = results['ac_func_type_list']
    n_init_obs = results['n_init_obs']

    for ac_func_type in ac_func_type_list:
        # setup a
        results_a = results['setup_a']
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Number of BO Iterations')
        ax.set_ylabel('Average Normalized Simple Regret')
        plot_performance_curve(ax, 'Hand-specified', results_a['bo_results_total'][ac_func_type]['fixed_regrets_all_list'], time_list, n_init_obs)
        plot_performance_curve(ax, 'Random', results_a['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs)
        plot_performance_curve(ax, 'Hyperbo+', results_a['bo_results_total'][ac_func_type]['gamma_regrets_all_list'], time_list, n_init_obs)
        if results['gt_gp_distribution_params']:
            plot_performance_curve(ax, 'Ground-truth', results_a['bo_results_total'][ac_func_type]['gt_regrets_all_list'], time_list, n_init_obs)
        ax.legend()
        fig.savefig(os.path.join(dir_path, '{}_setup_a_regret_vs_iteration.pdf'.format(ac_func_type)))
        plt.close(fig)

        # setup b
        results_b = results['setup_b']
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Number of BO Iterations')
        ax.set_ylabel('Average Normalized Simple Regret')
        plot_performance_curve(ax, 'Hand-specified', results_b['bo_results_total'][ac_func_type]['fixed_regrets_all_list'],
                                          time_list, n_init_obs)
        plot_performance_curve(ax, 'Random', results_b['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs)
        plot_performance_curve(ax, 'Hyperbo+', results_b['bo_results_total'][ac_func_type]['gamma_regrets_all_list'],
                                          time_list, n_init_obs)
        plot_performance_curve(ax, 'Hyperbo', results_b['bo_results_total'][ac_func_type]['hyperbo_regrets_all_list'],
                                          time_list, n_init_obs)
        if results['gt_gp_distribution_params']:
            plot_performance_curve(ax, 'Ground-truth', results_b['bo_results_total'][ac_func_type]['gt_regrets_all_list'],
                                              time_list, n_init_obs)
        ax.legend()
        fig.savefig(os.path.join(dir_path, '{}_setup_b_regret_vs_iteration.pdf'.format(ac_func_type)))
        plt.close(fig)


color_map = {
    'Hand-specified': 'tab:blue',
    'Random': 'tab:orange',
    'Hyperbo+': 'tab:green',
    'Hyperbo': 'tab:red',
    'Ground-truth': 'tab:purple',
    'Uniform-prior': 'tab:brown',
    'Hyperbo+ leaveout': 'tab:gray',
    'Discrete': 'tab:pink',
}

line_style_map = {
    'pi': '-',
    'ei': '--',
    'ucb': '-.',
}


def plot_hyperbo_plus_combined_split(results, ac_func_type, title_a, title_b, ax_a, ax_b):
    time_list = range(1, results['budget'] + 1)
    n_init_obs = results['n_init_obs']

    # setup a
    results_a = results['setup_a']
    ax_a.set_xlabel('Number of BO Iterations')
    ax_a.set_ylabel('Average Normalized Simple Regret')
    plot_performance_curve(ax_a, 'Hand-specified', results_a['bo_results_total'][ac_func_type]['fixed_regrets_all_list'], time_list, n_init_obs, color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'Random', results_a['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs, color=color_map['Random'])
    plot_performance_curve(ax_a, 'HyperBO+', results_a['bo_results_total'][ac_func_type]['gamma_regrets_all_list'], time_list, n_init_obs, color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth', results_a['bo_results_total'][ac_func_type]['gt_regrets_all_list'], time_list, n_init_obs, color=color_map['Ground-truth'])
    ax_a.set_title(title_a)

    # setup b
    results_b = results['setup_b']
    ax_b.set_xlabel('Number of BO Iterations')
    ax_b.set_ylabel('Average Normalized Simple Regret')
    plot_performance_curve(ax_b, 'Hand-specified', results_b['bo_results_total'][ac_func_type]['fixed_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'Random', results_b['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs, color=color_map['Random'])
    plot_performance_curve(ax_b, 'HyperBO+', results_b['bo_results_total'][ac_func_type]['gamma_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO', results_b['bo_results_total'][ac_func_type]['hyperbo_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth', results_b['bo_results_total'][ac_func_type]['gt_regrets_all_list'],
                                          time_list, n_init_obs, color=color_map['Ground-truth'])
    ax_b.set_title(title_b)


def plot_hyperbo_plus_combined(dir_path, results_synthetic, results_hpob):
    ac_func_type_list = results_synthetic['ac_func_type_list']
    matplotlib.rc('font', size=15)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for ac_func_type in ac_func_type_list:
        fig, ((ax_synthetic_a, ax_synthetic_b), (ax_hpob_a, ax_hpob_b)) = plt.subplots(nrows=2, ncols=2,
                                                                                       figsize=(20, 12.5))
        plot_hyperbo_plus_combined_split(results_synthetic, ac_func_type, '(a) Synthetic Super-dataset, Setup A',
                                         '(b) Synthetic Super-dataset, Setup B', ax_synthetic_a, ax_synthetic_b)
        plot_hyperbo_plus_combined_split(results_hpob, ac_func_type, '(c) HPO-B Super-dataset, Setup A',
                                         '(d) HPO-B Super-dataset, Setup B', ax_hpob_a, ax_hpob_b)
        handles, labels = ax_synthetic_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0))
        fig.savefig(os.path.join(dir_path, '{}_regret_vs_iteration_combined.pdf'.format(ac_func_type)))
        plt.close(fig)


def plot_hyperbo_plus_combined_split_all_acfuncs(results, title_a, title_b, ax_a, ax_b):
    time_list = range(1, results['budget'] + 1)
    n_init_obs = results['n_init_obs']

    # setup a
    ax_a.set_xlabel('Number of BO Iterations')
    ax_a.set_ylabel('Average Normalized Simple Regret')

    results_a = results['setup_a']
    plot_performance_curve(ax_a, 'Hand-specified (PI)', results_a['bo_results_total']['pi']['fixed_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'Random', results_a['bo_results_total']['pi']['random_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'])
    plot_performance_curve(ax_a, 'HyperBO+ (PI)', results_a['bo_results_total']['pi']['gamma_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (PI)', results_a['bo_results_total']['pi']['gt_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_a, 'Hand-specified (EI)', results_a['bo_results_total']['ei']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'HyperBO+ (EI)', results_a['bo_results_total']['ei']['gamma_regrets_all_list'], time_list,
                           n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (EI)', results_a['bo_results_total']['ei']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_a, 'Hand-specified (UCB)', results_a['bo_results_total']['ucb']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'HyperBO+ (UCB)', results_a['bo_results_total']['ucb']['gamma_regrets_all_list'],
                           time_list,
                           n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (UCB)', results_a['bo_results_total']['ucb']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Ground-truth'])

    ax_a.set_title(title_a)

    # setup b
    ax_b.set_xlabel('Number of BO Iterations')
    ax_b.set_ylabel('Average Normalized Simple Regret')

    results_b = results['setup_b']
    plot_performance_curve(ax_b, 'Hand-specified (PI)', results_b['bo_results_total']['pi']['fixed_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'Random', results_b['bo_results_total']['pi']['random_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'])
    plot_performance_curve(ax_b, 'HyperBO+ (PI)', results_b['bo_results_total']['pi']['gamma_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (PI)', results_b['bo_results_total']['pi']['hyperbo_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (PI)', results_b['bo_results_total']['pi']['gt_regrets_all_list'],
                                          time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_b, 'Hand-specified (EI)', results_b['bo_results_total']['ei']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'HyperBO+ (EI)', results_b['bo_results_total']['ei']['gamma_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (EI)', results_b['bo_results_total']['ei']['hyperbo_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (EI)', results_b['bo_results_total']['ei']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_b, 'Hand-specified (UCB)', results_b['bo_results_total']['ucb']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'HyperBO+ (UCB)', results_b['bo_results_total']['ucb']['gamma_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (UCB)', results_b['bo_results_total']['ucb']['hyperbo_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (UCB)', results_b['bo_results_total']['ucb']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Ground-truth'])

    ax_b.set_title(title_b)


def plot_hyperbo_plus_combined_all_acfuncs(dir_path, results_synthetic, results_hpob):
    matplotlib.rc('font', size=15)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig, ((ax_synthetic_a, ax_synthetic_b), (ax_hpob_a, ax_hpob_b)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 13))
    plot_hyperbo_plus_combined_split_all_acfuncs(results_synthetic, '(a) Synthetic Super-dataset, Setup A', '(b) Synthetic Super-dataset, Setup B', ax_synthetic_a, ax_synthetic_b)
    plot_hyperbo_plus_combined_split_all_acfuncs(results_hpob, '(c) HPO-B Super-dataset, Setup A', '(d) HPO-B Super-dataset, Setup B', ax_hpob_a, ax_hpob_b)
    handles, labels = ax_synthetic_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.savefig(os.path.join(dir_path, 'all_acfuncs_regret_vs_iteration_combined.pdf'))
    plt.close(fig)


def plot_hyperbo_plus_uniform_prior_baseline(results):
    experiment_name = results['experiment_name']
    dir_path = os.path.join('results', experiment_name)
    time_list = range(1, results['budget'] + 1)
    ac_func_type_list = results['ac_func_type_list']
    n_init_obs = results['n_init_obs']

    for ac_func_type in ac_func_type_list:
        # setup a
        results_a = results['setup_a']
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Number of BO Iterations')
        ax.set_ylabel('Average Normalized Simple Regret')
        plot_performance_curve(ax, 'Uniform-prior', results_a['bo_results_total'][ac_func_type]['uniform_regrets_all_list'], time_list, n_init_obs)
        ax.legend()
        fig.savefig(os.path.join(dir_path, '{}_setup_a_regret_vs_iteration_uniform_prior.pdf'.format(ac_func_type)))
        plt.close(fig)

        # setup b
        results_b = results['setup_b']
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.set_xlabel('Number of BO Iterations')
        ax.set_ylabel('Average Normalized Simple Regret')
        plot_performance_curve(ax, 'Uniform-prior', results_b['bo_results_total'][ac_func_type]['uniform_regrets_all_list'],
                                          time_list, n_init_obs)
        ax.legend()
        fig.savefig(os.path.join(dir_path, '{}_setup_b_regret_vs_iteration_uniform_prior.pdf'.format(ac_func_type)))
        plt.close(fig)


def plot_hyperbo_plus_combined_split_add_baselines(results, results_2, ac_func_type, title_a, title_b, ax_a, ax_b):
    time_list = range(1, results['budget'] + 1)
    n_init_obs = results['n_init_obs']

    # setup a
    results_a = results['setup_a']
    ax_a.set_xlabel('Number of BO Iterations')
    ax_a.set_ylabel('Average Normalized Simple Regret')
    plot_performance_curve(ax_a, 'Hand-specified', results_a['bo_results_total'][ac_func_type]['fixed_regrets_all_list'], time_list, n_init_obs, color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'Random', results_a['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs, color=color_map['Random'])
    plot_performance_curve(ax_a, 'HyperBO+', results_a['bo_results_total'][ac_func_type]['gamma_regrets_all_list'], time_list, n_init_obs, color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth', results_a['bo_results_total'][ac_func_type]['gt_regrets_all_list'], time_list, n_init_obs, color=color_map['Ground-truth'])
    plot_performance_curve(ax_a, 'Uniform-prior',
                           results_2['setup_a']['bo_results_total'][ac_func_type]['uniform_regrets_all_list'], time_list, n_init_obs,
                           color=color_map['Uniform-prior'])
    ax_a.set_title(title_a)

    # setup b
    results_b = results['setup_b']
    ax_b.set_xlabel('Number of BO Iterations')
    ax_b.set_ylabel('Average Normalized Simple Regret')
    plot_performance_curve(ax_b, 'Hand-specified', results_b['bo_results_total'][ac_func_type]['fixed_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'Random', results_b['bo_results_total'][ac_func_type]['random_regrets_all_list'], time_list, n_init_obs, color=color_map['Random'])
    plot_performance_curve(ax_b, 'HyperBO+', results_b['bo_results_total'][ac_func_type]['gamma_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO', results_b['bo_results_total'][ac_func_type]['hyperbo_regrets_all_list'],
                                      time_list, n_init_obs, color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth', results_b['bo_results_total'][ac_func_type]['gt_regrets_all_list'],
                                          time_list, n_init_obs, color=color_map['Ground-truth'])
    plot_performance_curve(ax_b, 'Uniform-prior',
                           results_2['setup_b']['bo_results_total'][ac_func_type]['uniform_regrets_all_list'],
                           time_list, n_init_obs,
                           color=color_map['Uniform-prior'])
    ax_b.set_title(title_b)


def plot_hyperbo_plus_combined_add_baselines(dir_path, results_synthetic, results_hpob, results_synthetic_2, results_hpob_2):
    ac_func_type_list = results_synthetic['ac_func_type_list']
    matplotlib.rc('font', size=15)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for ac_func_type in ac_func_type_list:
        fig, ((ax_synthetic_a, ax_synthetic_b), (ax_hpob_a, ax_hpob_b)) = plt.subplots(nrows=2, ncols=2,
                                                                                       figsize=(20, 12.5))
        plot_hyperbo_plus_combined_split_add_baselines(results_synthetic, results_synthetic_2, ac_func_type, '(a) Synthetic Super-dataset, Setup A',
                                         '(b) Synthetic Super-dataset, Setup B', ax_synthetic_a, ax_synthetic_b)
        plot_hyperbo_plus_combined_split_add_baselines(results_hpob, results_hpob_2, ac_func_type, '(c) HPO-B Super-dataset, Setup A',
                                         '(d) HPO-B Super-dataset, Setup B', ax_hpob_a, ax_hpob_b)
        handles, labels = ax_synthetic_b.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.0))
        fig.savefig(os.path.join(dir_path, '{}_regret_vs_iteration_combined.pdf'.format(ac_func_type)))
        plt.close(fig)


def plot_hyperbo_plus_combined_split_all_acfuncs_added_baselines(results, results_2, title_a, title_b, ax_a, ax_b):
    time_list = range(1, results['budget'] + 1)
    n_init_obs = results['n_init_obs']

    # setup a
    ax_a.set_xlabel('Number of BO Iterations')
    ax_a.set_ylabel('Average Normalized Simple Regret')

    results_a = results['setup_a']
    plot_performance_curve(ax_a, 'Hand-specified (PI)', results_a['bo_results_total']['pi']['fixed_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'Random', results_a['bo_results_total']['pi']['random_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'])
    plot_performance_curve(ax_a, 'HyperBO+ (PI)', results_a['bo_results_total']['pi']['gamma_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (PI)', results_a['bo_results_total']['pi']['gt_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_a, 'Hand-specified (EI)', results_a['bo_results_total']['ei']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'HyperBO+ (EI)', results_a['bo_results_total']['ei']['gamma_regrets_all_list'], time_list,
                           n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (EI)', results_a['bo_results_total']['ei']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_a, 'Hand-specified (UCB)', results_a['bo_results_total']['ucb']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_a, 'HyperBO+ (UCB)', results_a['bo_results_total']['ucb']['gamma_regrets_all_list'],
                           time_list,
                           n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_a, 'Ground-truth (UCB)', results_a['bo_results_total']['ucb']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_a, 'Uniform-prior (PI)', results_2['setup_a']['bo_results_total']['pi']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Uniform-prior'])
    plot_performance_curve(ax_a, 'Uniform-prior (EI)',
                           results_2['setup_a']['bo_results_total']['ei']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Uniform-prior'])
    plot_performance_curve(ax_a, 'Uniform-prior (UCB)',
                           results_2['setup_a']['bo_results_total']['ucb']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Uniform-prior'])

    ax_a.set_title(title_a)

    # setup b
    ax_b.set_xlabel('Number of BO Iterations')
    ax_b.set_ylabel('Average Normalized Simple Regret')

    results_b = results['setup_b']
    plot_performance_curve(ax_b, 'Hand-specified (PI)', results_b['bo_results_total']['pi']['fixed_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'Random', results_b['bo_results_total']['pi']['random_regrets_all_list'], time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'])
    plot_performance_curve(ax_b, 'HyperBO+ (PI)', results_b['bo_results_total']['pi']['gamma_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (PI)', results_b['bo_results_total']['pi']['hyperbo_regrets_all_list'],
                                      time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (PI)', results_b['bo_results_total']['pi']['gt_regrets_all_list'],
                                          time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_b, 'Hand-specified (EI)', results_b['bo_results_total']['ei']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'HyperBO+ (EI)', results_b['bo_results_total']['ei']['gamma_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (EI)', results_b['bo_results_total']['ei']['hyperbo_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (EI)', results_b['bo_results_total']['ei']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_b, 'Hand-specified (UCB)', results_b['bo_results_total']['ucb']['fixed_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hand-specified'])
    plot_performance_curve(ax_b, 'HyperBO+ (UCB)', results_b['bo_results_total']['ucb']['gamma_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'])
    plot_performance_curve(ax_b, 'HyperBO (UCB)', results_b['bo_results_total']['ucb']['hyperbo_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo'])
    if results['gt_gp_distribution_params']:
        plot_performance_curve(ax_b, 'Ground-truth (UCB)', results_b['bo_results_total']['ucb']['gt_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Ground-truth'])

    plot_performance_curve(ax_b, 'Uniform-prior (PI)',
                           results_2['setup_b']['bo_results_total']['pi']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Uniform-prior'])
    plot_performance_curve(ax_b, 'Uniform-prior (EI)',
                           results_2['setup_b']['bo_results_total']['ei']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Uniform-prior'])
    plot_performance_curve(ax_b, 'Uniform-prior (UCB)',
                           results_2['setup_b']['bo_results_total']['ucb']['uniform_regrets_all_list'],
                           time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Uniform-prior'])

    ax_b.set_title(title_b)


def plot_hyperbo_plus_combined_all_acfuncs_added_baselines(dir_path, results_synthetic, results_hpob, results_synthetic_2, results_hpob_2):
    matplotlib.rc('font', size=15)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig, ((ax_synthetic_a, ax_synthetic_b), (ax_hpob_a, ax_hpob_b)) = plt.subplots(nrows=2, ncols=2, figsize=(20, 13))
    plot_hyperbo_plus_combined_split_all_acfuncs_added_baselines(results_synthetic, results_synthetic_2, '(a) Synthetic Super-dataset, Setup A', '(b) Synthetic Super-dataset, Setup B', ax_synthetic_a, ax_synthetic_b)
    plot_hyperbo_plus_combined_split_all_acfuncs_added_baselines(results_hpob, results_hpob_2, '(c) HPO-B Super-dataset, Setup A', '(d) HPO-B Super-dataset, Setup B', ax_hpob_a, ax_hpob_b)
    handles, labels = ax_synthetic_b.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.99))
    fig.savefig(os.path.join(dir_path, 'all_acfuncs_regret_vs_iteration_combined.pdf'))
    plt.close(fig)


def plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_single_ax(ax, results, results_2, dataset_name, setup, log_plot=False, results_3=None, results_4=None, ucb=False):
    time_list = range(1, results['budget'] + 1)
    n_init_obs = results['n_init_obs']

    ax.set_xlabel('Number of BO Iterations')
    ax.set_ylabel('Average Normalized Simple Regret')

    if setup == 'a':
        # setup a
        results_a = results['setup_a']
        plot_performance_curve(ax, 'Non-informative (PI)',
                               results_2['setup_a']['bo_results_total']['pi']['uniform_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Uniform-prior'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'Non-informative (EI)',
                               results_2['setup_a']['bo_results_total']['ei']['uniform_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Uniform-prior'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'Non-informative (UCB)',
                                   results_2['setup_a']['bo_results_total']['ucb']['uniform_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                   color=color_map['Uniform-prior'], log_plot=log_plot)

        plot_performance_curve(ax, 'Hand-specified (PI)', results_a['bo_results_total']['pi']['fixed_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'],
                               log_plot=log_plot)

        plot_performance_curve(ax, 'Hand-specified (EI)', results_a['bo_results_total']['ei']['fixed_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'Hand-specified (UCB)',
                                   results_a['bo_results_total']['ucb']['fixed_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                   color=color_map['Hand-specified'], log_plot=log_plot)

        plot_performance_curve(ax, 'Random', results_a['bo_results_total']['pi']['random_regrets_all_list'], time_list,
                               n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'], log_plot=log_plot)

        plot_performance_curve(ax, 'HyperBO+ (PI)', results_a['bo_results_total']['pi']['gamma_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'HyperBO+ (EI)', results_a['bo_results_total']['ei']['gamma_regrets_all_list'],
                               time_list,
                               n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'HyperBO+ (UCB)', results_a['bo_results_total']['ucb']['gamma_regrets_all_list'],
                                   time_list,
                                   n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'],
                                   log_plot=log_plot)

        if results['gt_gp_distribution_params']:
            plot_performance_curve(ax, 'Ground-truth (PI)', results_a['bo_results_total']['pi']['gt_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['pi'],
                                   color=color_map['Ground-truth'], log_plot=log_plot)
            plot_performance_curve(ax, 'Ground-truth (EI)', results_a['bo_results_total']['ei']['gt_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ei'],
                                   color=color_map['Ground-truth'], log_plot=log_plot)
            if ucb:
                plot_performance_curve(ax, 'Ground-truth (UCB)',
                                       results_a['bo_results_total']['ucb']['gt_regrets_all_list'],
                                       time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                       color=color_map['Ground-truth'], log_plot=log_plot)
        if results_4 is not None:
            plot_performance_curve(ax, '$^x$HyperBO+ (PI)',
                                   results_4['setup_a']['bo_results_total']['pi']['discrete_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Discrete'],
                                   log_plot=log_plot)
            plot_performance_curve(ax, '$^x$HyperBO+ (EI)',
                                   results_4['setup_a']['bo_results_total']['ei']['discrete_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Discrete'],
                                   log_plot=log_plot)
            if ucb:
                plot_performance_curve(ax, '$^x$HyperBO+ (UCB)',
                                       results_4['setup_a']['bo_results_total']['ucb']['discrete_regrets_all_list'],
                                       time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Discrete'],
                                       log_plot=log_plot)
    elif setup == 'b':
        # setup b
        results_b = results['setup_b']
        plot_performance_curve(ax, 'Non-informative (PI)',
                               results_2['setup_b']['bo_results_total']['pi']['uniform_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Uniform-prior'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'Non-informative (EI)',
                               results_2['setup_b']['bo_results_total']['ei']['uniform_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Uniform-prior'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'Non-informative (UCB)',
                                   results_2['setup_b']['bo_results_total']['ucb']['uniform_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                   color=color_map['Uniform-prior'], log_plot=log_plot)

        plot_performance_curve(ax, 'Hand-specified (PI)', results_b['bo_results_total']['pi']['fixed_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hand-specified'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'Hand-specified (EI)', results_b['bo_results_total']['ei']['fixed_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hand-specified'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'Hand-specified (UCB)',
                                   results_b['bo_results_total']['ucb']['fixed_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                   color=color_map['Hand-specified'], log_plot=log_plot)

        plot_performance_curve(ax, 'Random', results_b['bo_results_total']['pi']['random_regrets_all_list'], time_list,
                               n_init_obs, linestyle=line_style_map['pi'], color=color_map['Random'], log_plot=log_plot)

        plot_performance_curve(ax, 'HyperBO+ (PI)', results_b['bo_results_total']['pi']['gamma_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo+'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'HyperBO+ (EI)', results_b['bo_results_total']['ei']['gamma_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo+'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'HyperBO+ (UCB)', results_b['bo_results_total']['ucb']['gamma_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo+'],
                                   log_plot=log_plot)

        plot_performance_curve(ax, 'HyperBO (PI)', results_b['bo_results_total']['pi']['hyperbo_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Hyperbo'],
                               log_plot=log_plot)
        plot_performance_curve(ax, 'HyperBO (EI)', results_b['bo_results_total']['ei']['hyperbo_regrets_all_list'],
                               time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Hyperbo'],
                               log_plot=log_plot)
        if ucb:
            plot_performance_curve(ax, 'HyperBO (UCB)',
                                   results_b['bo_results_total']['ucb']['hyperbo_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Hyperbo'],
                                   log_plot=log_plot)

        if results['gt_gp_distribution_params']:
            plot_performance_curve(ax, 'Ground-truth (PI)', results_b['bo_results_total']['pi']['gt_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['pi'],
                                   color=color_map['Ground-truth'], log_plot=log_plot)
            plot_performance_curve(ax, 'Ground-truth (EI)', results_b['bo_results_total']['ei']['gt_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ei'],
                                   color=color_map['Ground-truth'], log_plot=log_plot)
            if ucb:
                plot_performance_curve(ax, 'Ground-truth (UCB)',
                                       results_b['bo_results_total']['ucb']['gt_regrets_all_list'],
                                       time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                       color=color_map['Ground-truth'], log_plot=log_plot)

        if results_3 is not None:
            plot_performance_curve(ax, '$^z$HyperBO+ (PI)',
                                   results_3['setup_b']['bo_results_total']['pi']['leaveout_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['pi'],
                                   color=color_map['Hyperbo+ leaveout'],
                                   log_plot=log_plot)
            plot_performance_curve(ax, '$^z$HyperBO+ (EI)',
                                   results_3['setup_b']['bo_results_total']['ei']['leaveout_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ei'],
                                   color=color_map['Hyperbo+ leaveout'],
                                   log_plot=log_plot)
            if ucb:
                plot_performance_curve(ax, '$^z$HyperBO+ (UCB)',
                                       results_3['setup_b']['bo_results_total']['ucb']['leaveout_regrets_all_list'],
                                       time_list, n_init_obs, linestyle=line_style_map['ucb'],
                                       color=color_map['Hyperbo+ leaveout'],
                                       log_plot=log_plot)
        else:
            print('No leaveout results')

        if results_4 is not None:
            plot_performance_curve(ax, '$^x$HyperBO+ (PI)',
                                   results_4['setup_b']['bo_results_total']['pi']['discrete_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['pi'], color=color_map['Discrete'],
                                   log_plot=log_plot)
            plot_performance_curve(ax, '$^x$HyperBO+ (EI)',
                                   results_4['setup_b']['bo_results_total']['ei']['discrete_regrets_all_list'],
                                   time_list, n_init_obs, linestyle=line_style_map['ei'], color=color_map['Discrete'],
                                   log_plot=log_plot)
            if ucb:
                plot_performance_curve(ax, '$^x$HyperBO+ (UCB)',
                                       results_4['setup_b']['bo_results_total']['ucb']['discrete_regrets_all_list'],
                                       time_list, n_init_obs, linestyle=line_style_map['ucb'], color=color_map['Discrete'],
                                       log_plot=log_plot)
    else:
        raise ValueError('Invalid setup, must be a or b')


def plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_single(results, results_2, dataset_name, setup, log_plot=False, results_3=None, results_4=None, ucb=False):
    matplotlib.rc('font', size=20)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_single_ax(ax, results, results_2, dataset_name, setup,
                                                                           log_plot=log_plot, results_3=results_3, results_4=results_4, ucb=ucb)

    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.59, 0.96), fontsize=15, facecolor='white', framealpha=1)
    fig.savefig(os.path.join(dir_path, 'ei_pi_regret_vs_iteration_{}_setup_{}_log_{}_ucb_{}.pdf'.format(dataset_name, setup, log_plot, ucb)), bbox_inches='tight')
    plt.close(fig)


def plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_double(results, results_2, dataset_name, log_plot=False, results_3=None, results_4=None, ucb=False):
    matplotlib.rc('font', size=20)

    fig, (ax_a, ax_b) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

    plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_single_ax(ax_a, results, results_2, dataset_name, 'a',
                                                                           log_plot=log_plot, results_3=results_3, results_4=results_4, ucb=ucb)
    ax_a.set_title('(a) Setup A')
    plot_hyperbo_plus_split_combined_all_acfuncs_added_baselines_single_ax(ax_b, results, results_2, dataset_name,
                                                                           'b',
                                                                           log_plot=log_plot, results_3=results_3, results_4=results_4,
                                                                           ucb=ucb)
    ax_b.set_title('(b) Setup B')

    handles, labels = ax_b.get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=15, facecolor='white', framealpha=1)
    fig.savefig(os.path.join(dir_path, 'ei_pi_regret_vs_iteration_{}_log_{}_ucb_{}.pdf'.format(dataset_name, log_plot, ucb)), bbox_inches='tight')
    plt.close(fig)


def plot_lengthscale_distribution(results, dataset_name, dir_path):
    matplotlib.rc('font', size=15)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax.set_xscale('log')
    ax2 = ax.twinx()
    ax.set_xlabel('Length-scale')
    ax.set_ylabel('Frequency')
    ax2.set_ylabel('Probability Density')
    setup_b_id_list = results['setup_b_id_list']
    lengthscale_list = []
    for train_id in setup_b_id_list:
        lengthscale_list += [x for x in np.array(results['setup_b']['fit_gp_params'][train_id]['gp_params']['lengthscale'])]

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


if __name__ == '__main__':
    dir_path = os.path.join('../results', 'hyperbo_plus_plots')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
