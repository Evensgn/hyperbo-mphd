from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import utils
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import Normal, Gamma, LogNormal, Uniform
from experiment_defs import *


def build_lengthscale_prior(per_dim_log_prob_list):
    n_dim = len(per_dim_log_prob_list)
    return lambda ls: jnp.sum(jnp.array([per_dim_log_prob_list[i](ls[i]) for i in range(n_dim)]))


def get_gp_priors_from_gt_hgp(gt_hgp_params, distribution_type, n_dim):
    constant_mu, constant_sigma = gt_hgp_params['constant']['mu'], gt_hgp_params['constant']['sigma']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        lengthscale_a = gt_hgp_params['lengthscale']['alpha_w'] * n_dim + gt_hgp_params['lengthscale']['alpha_b']
        lengthscale_b = gt_hgp_params['lengthscale']['beta_w'] * n_dim + gt_hgp_params['lengthscale']['beta_b']
        lengthscale_dist = Gamma(lengthscale_a, lengthscale_b)

        signal_variance_a, signal_variance_b = gt_hgp_params['signal_variance']['alpha'], \
                                               gt_hgp_params['signal_variance']['beta']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = gt_hgp_params['noise_variance']['alpha'], \
                                             gt_hgp_params['noise_variance']['beta']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    elif distribution_type == 'lognormal':
        lengthscale_mu = gt_hgp_params['lengthscale']['mu_w'] * n_dim + gt_hgp_params['lengthscale']['mu_b']
        lengthscale_sigma = gt_hgp_params['lengthscale']['sigma_w'] * n_dim + gt_hgp_params['lengthscale']['sigma_b']
        lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)

        signal_variance_mu, signal_variance_sigma = gt_hgp_params['signal_variance']['mu'], \
                                                    gt_hgp_params['signal_variance']['sigma']
        signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
        noise_variance_mu, noise_variance_sigma = gt_hgp_params['noise_variance']['mu'], \
                                                  gt_hgp_params['noise_variance']['sigma']
        noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    priors = {
        'constant': constant_dist.log_prob,
        'lengthscale': build_lengthscale_prior([lengthscale_dist.log_prob] * n_dim),
        'signal_variance': signal_variance_dist.log_prob,
        'noise_variance': noise_variance_dist.log_prob,
    }
    return priors


class LogUniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def log_prob(self, x):
        return jax.lax.cond(
            jnp.logical_or(x < jnp.exp(self.a), x > jnp.exp(self.b)),
            lambda _: -jnp.inf,
            lambda _: jnp.log(1 / (x * (self.b - self.a))),
            operand=None,
        )


def get_gp_priors_from_direct_hgp(direct_hgp_params, distribution_type, n_dim):
    if distribution_type == 'all_uniform':
        constant_a, constant_b = direct_hgp_params['constant']
        constant_dist = Uniform(constant_a, constant_b)
        lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
        lengthscale_dist = Uniform(lengthscale_a, lengthscale_b)
        signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
        signal_variance_dist = Uniform(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
        noise_variance_dist = Uniform(noise_variance_a, noise_variance_b)
    elif distribution_type == 'log_uniform':
        constant_a, constant_b = direct_hgp_params['constant']
        constant_dist = Uniform(constant_a, constant_b)
        lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
        lengthscale_dist = LogUniform(lengthscale_a, lengthscale_b)
        signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
        signal_variance_dist = LogUniform(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
        noise_variance_dist = LogUniform(noise_variance_a, noise_variance_b)
    else:
        constant_mu, constant_sigma = direct_hgp_params['constant']
        constant_dist = Normal(constant_mu, constant_sigma)

        if distribution_type == 'gamma':
            lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
            lengthscale_dist = Gamma(lengthscale_a, lengthscale_b)
            signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
            signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
            noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
            noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
        elif distribution_type == 'lognormal':
            lengthscale_mu, lengthscale_sigma = direct_hgp_params['lengthscale']
            lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
            signal_variance_mu, signal_variance_sigma = direct_hgp_params['signal_variance']
            signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
            noise_variance_mu, noise_variance_sigma = direct_hgp_params['noise_variance']
            noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
        else:
            raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    priors = {
        'constant': constant_dist.log_prob,
        'lengthscale': build_lengthscale_prior([lengthscale_dist.log_prob] * n_dim),
        'signal_variance': signal_variance_dist.log_prob,
        'noise_variance': noise_variance_dist.log_prob,
    }
    return priors


def get_gp_priors_from_hpl_hgp(hpl_hgp_params, distribution_type, n_dim, dim_feature_value):
    constant_mu, constant_sigma = hpl_hgp_params['constant_normal_params']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = hpl_hgp_params['signal_variance_gamma_params']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = hpl_hgp_params['noise_variance_gamma_params']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    elif distribution_type == 'lognormal':
        signal_variance_mu, signal_variance_sigma = hpl_hgp_params['signal_variance_lognormal_params']
        signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
        noise_variance_mu, noise_variance_sigma = hpl_hgp_params['noise_variance_lognormal_params']
        noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    lengthscale_dist_mlp_features = HPL_LENGTHSCALE_MLP_FEATURES
    if distribution_type == 'gamma':
        lengthscale_dist_mlp_params = hpl_hgp_params['lengthscale_gamma_mlp_params']
    elif distribution_type == 'lognormal':
        lengthscale_dist_mlp_params = hpl_hgp_params['lengthscale_lognormal_mlp_params']
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))
    lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
    lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params},
                                                      dim_feature_value)
    lengthscales_log_prob_list = []
    for dim in range(n_dim):
        if distribution_type == 'gamma':
            lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
            lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
            lengthscale_dist = Gamma(lengthscale_a, rate=lengthscale_b)
        elif distribution_type == 'lognormal':
            lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
            lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
            lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
        else:
            raise ValueError('Unknown distribution type: {}'.format(distribution_type))
        lengthscales_log_prob_list.append(lengthscale_dist.log_prob)

    priors = {
        'constant': constant_dist.log_prob,
        'lengthscale': build_lengthscale_prior(lengthscales_log_prob_list),
        'signal_variance': signal_variance_dist.log_prob,
        'noise_variance': noise_variance_dist.log_prob,
    }
    return priors


def get_gp_params_samples_from_direct_hgp(key, direct_hgp_params, distribution_type, n_gp_params_samples, n_dim):
    if distribution_type == 'all_uniform':
        constant_a, constant_b = direct_hgp_params['constant']
        constant_dist = Uniform(constant_a, constant_b)
        lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
        lengthscale_dist = Uniform(lengthscale_a, lengthscale_b)
        signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
        signal_variance_dist = Uniform(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
        noise_variance_dist = Uniform(noise_variance_a, noise_variance_b)
    else:
        constant_mu, constant_sigma = direct_hgp_params['constant']
        constant_dist = Normal(constant_mu, constant_sigma)

        if distribution_type == 'gamma':
            lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
            lengthscale_dist = Gamma(lengthscale_a, lengthscale_b)
            signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
            signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
            noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
            noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
        elif distribution_type == 'lognormal':
            lengthscale_mu, lengthscale_sigma = direct_hgp_params['lengthscale']
            lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
            signal_variance_mu, signal_variance_sigma = direct_hgp_params['signal_variance']
            signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
            noise_variance_mu, noise_variance_sigma = direct_hgp_params['noise_variance']
            noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
        else:
            raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    new_key, key = jax.random.split(key)
    constants = constant_dist.sample(n_gp_params_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    lengthscales = lengthscale_dist.sample((n_gp_params_samples, n_dim), seed=new_key)
    new_key, key = jax.random.split(key)
    signal_variances = signal_variance_dist.sample(n_gp_params_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    noise_variances = noise_variance_dist.sample(n_gp_params_samples, seed=new_key)
    gp_params_samples = (constants, lengthscales, signal_variances, noise_variances)
    return gp_params_samples


def get_gp_params_samples_from_hpl_hgp(key, hpl_hgp_params, distribution_type, n_gp_params_samples, n_dim,
                                       dim_feature_value):
    constant_mu, constant_sigma = hpl_hgp_params['constant_normal_params']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = hpl_hgp_params['signal_variance_gamma_params']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = hpl_hgp_params['noise_variance_gamma_params']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    elif distribution_type == 'lognormal':
        signal_variance_mu, signal_variance_sigma = hpl_hgp_params['signal_variance_lognormal_params']
        signal_variance_dist = LogNormal(signal_variance_mu, signal_variance_sigma)
        noise_variance_mu, noise_variance_sigma = hpl_hgp_params['noise_variance_lognormal_params']
        noise_variance_dist = LogNormal(noise_variance_mu, noise_variance_sigma)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    new_key, key = jax.random.split(key)
    constants = constant_dist.sample(n_gp_params_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    signal_variances = signal_variance_dist.sample(n_gp_params_samples, seed=new_key)
    new_key, key = jax.random.split(key)
    noise_variances = noise_variance_dist.sample(n_gp_params_samples, seed=new_key)

    # sample lengthscales
    lengthscale_dist_mlp_features = HPL_LENGTHSCALE_MLP_FEATURES
    if distribution_type == 'gamma':
        lengthscale_dist_mlp_params = hpl_hgp_params['lengthscale_gamma_mlp_params']
    elif distribution_type == 'lognormal':
        lengthscale_dist_mlp_params = hpl_hgp_params['lengthscale_lognormal_mlp_params']
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))
    lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
    lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params},
                                                      dim_feature_value)
    lengthscales_dim_list = []
    for dim in range(n_dim):
        if distribution_type == 'gamma':
            lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[dim])
            lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
            lengthscale_dist = Gamma(lengthscale_a, rate=lengthscale_b)
        elif distribution_type == 'lognormal':
            lengthscale_dist_params_dim = utils.lognormal_params_warp(lengthscale_dist_params[dim])
            lengthscale_mu, lengthscale_sigma = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
            lengthscale_dist = LogNormal(lengthscale_mu, lengthscale_sigma)
        else:
            raise ValueError('Unknown distribution type: {}'.format(distribution_type))
        new_key, key = jax.random.split(key)
        lengthscales_dim = lengthscale_dist.sample(n_gp_params_samples, seed=new_key)
        lengthscales_dim_list.append(lengthscales_dim)
    lengthscales = jnp.stack(lengthscales_dim_list, axis=1)
    gp_params_samples = (constants, lengthscales, signal_variances, noise_variances)
    return gp_params_samples


def get_gp_prior_dist_from_gt_hgp(gt_hgp_params, distribution_type, n_dim):
    constant_mu, constant_sigma = gt_hgp_params['constant']['mu'], gt_hgp_params['constant']['sigma']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        lengthscale_a = gt_hgp_params['lengthscale']['alpha_w'] * n_dim + gt_hgp_params['lengthscale']['alpha_b']
        lengthscale_b = gt_hgp_params['lengthscale']['beta_w'] * n_dim + gt_hgp_params['lengthscale']['beta_b']
        lengthscale_dist = Gamma(lengthscale_a, lengthscale_b)

        signal_variance_a, signal_variance_b = gt_hgp_params['signal_variance']['alpha'], \
                                               gt_hgp_params['signal_variance']['beta']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = gt_hgp_params['noise_variance']['alpha'], \
                                             gt_hgp_params['noise_variance']['beta']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    priors = {
        'constant': constant_dist,
        'lengthscale': lengthscale_dist,
        'signal_variance': signal_variance_dist,
        'noise_variance': noise_variance_dist,
    }
    return priors


def get_gp_prior_dist_from_direct_hgp(direct_hgp_params, distribution_type):
    constant_mu, constant_sigma = direct_hgp_params['constant']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        lengthscale_a, lengthscale_b = direct_hgp_params['lengthscale']
        lengthscale_dist = Gamma(lengthscale_a, lengthscale_b)
        signal_variance_a, signal_variance_b = direct_hgp_params['signal_variance']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = direct_hgp_params['noise_variance']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    priors = {
        'constant': constant_dist,
        'lengthscale': lengthscale_dist,
        'signal_variance': signal_variance_dist,
        'noise_variance': noise_variance_dist,
    }
    return priors


def get_gp_prior_dist_for_synthetic_from_hpl_hgp(hpl_hgp_params, distribution_type, n_dim):
    constant_mu, constant_sigma = hpl_hgp_params['constant_normal_params']
    constant_dist = Normal(constant_mu, constant_sigma)

    if distribution_type == 'gamma':
        signal_variance_a, signal_variance_b = hpl_hgp_params['signal_variance_gamma_params']
        signal_variance_dist = Gamma(signal_variance_a, signal_variance_b)
        noise_variance_a, noise_variance_b = hpl_hgp_params['noise_variance_gamma_params']
        noise_variance_dist = Gamma(noise_variance_a, noise_variance_b)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    lengthscale_dist_mlp_features = HPL_LENGTHSCALE_MLP_FEATURES
    if distribution_type == 'gamma':
        lengthscale_dist_mlp_params = hpl_hgp_params['lengthscale_gamma_mlp_params']
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))
    lengthscale_model = bf.MLP(lengthscale_dist_mlp_features)
    dim_feature_value = jnp.array([[0., 1., 0., n_dim]])
    lengthscale_dist_params = lengthscale_model.apply({'params': lengthscale_dist_mlp_params},
                                                      dim_feature_value)
    if distribution_type == 'gamma':
        lengthscale_dist_params_dim = utils.gamma_params_warp(lengthscale_dist_params[0])
        lengthscale_a, lengthscale_b = lengthscale_dist_params_dim[0], lengthscale_dist_params_dim[1]
        lengthscale_dist = Gamma(lengthscale_a, rate=lengthscale_b)
    else:
        raise ValueError('Unknown distribution type: {}'.format(distribution_type))

    priors = {
        'constant': constant_dist,
        'lengthscale': lengthscale_dist,
        'signal_variance': signal_variance_dist,
        'noise_variance': noise_variance_dist,
    }
    return priors
