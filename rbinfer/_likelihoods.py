import numpy as np
from scipy.stats import poisson, nbinom
from scipy.special import logsumexp

def _log_lik(t, lambda_val, b, n_sample):
    """Log-likelihood of observing t given lambda, b, and n_sample."""
    mu = n_sample * (lambda_val + b)
    if mu < 0:
        return -np.inf
    return poisson(mu=mu).logpmf(t)

def _log_prior_pred(t, alpha, scale, b, n_sample):
    """
    Exact log marginal probability m(t) using the convolution of 
    a Negative Binomial and a Poisson distribution.
    """
    # Z ~ NB(alpha, 1/(1 + n_sample * scale))
    p = 1.0 / (1.0 + n_sample * scale)
    k_vals = np.arange(0, t + 1)
    
    # log PMFs for Z and Y where X = Z + Y
    log_z = nbinom.logpmf(k_vals, alpha, p)
    log_y = poisson.logpmf(t - k_vals, mu=n_sample * b)
    
    # logsumexp computes the log of the sum of the probabilities exactly
    return logsumexp(log_z + log_y)

def _log_lik_vb(t, lambda_val, alpha_b, scale_b, n_sample):
    """Exact log m(t | λ_0) using Poisson ⊛ Negative Binomial convolution."""
    if lambda_val < 0:          # ← add this guard
        return -np.inf
    p_b = 1.0 / (1.0 + n_sample * scale_b)
    k_vals = np.arange(0, t + 1)
    
    log_z_lam = poisson.logpmf(k_vals, mu=n_sample * lambda_val)
    log_z_b   = nbinom.logpmf(t - k_vals, alpha_b, p_b)
    
    return logsumexp(log_z_lam + log_z_b)

def _log_prior_pred_vb(t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample):
    """Exact full log m(t) using NB ⊛ NB convolution."""
    p_lam = 1.0 / (1.0 + n_sample * scale_lam)
    p_b   = 1.0 / (1.0 + n_sample * scale_b)
    k_vals = np.arange(0, t + 1)
    
    log_z_lam = nbinom.logpmf(k_vals, alpha_lam, p_lam)
    log_z_b   = nbinom.logpmf(t - k_vals, alpha_b, p_b)
    
    return logsumexp(log_z_lam + log_z_b)