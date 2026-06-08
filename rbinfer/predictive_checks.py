import numpy as np
from scipy.stats import poisson, nbinom
from scipy.special import logsumexp

from ._likelihoods import _log_prior_pred, _log_prior_pred_vb

def _log_cdf_pred(t_obs, alpha, scale, b, n_sample):
    """
    Exact log CDF log(P(T <= t_obs)) using logsumexp.
    T = Z + Y, so P(T <= t) = \sum P(Z = k) * P(Y <= t - k)
    """
    p = 1.0 / (1.0 + n_sample * scale)
    mu_b = n_sample * b
    
    k_vals = np.arange(0, t_obs + 1)
    
    # log P(Z = k)
    log_z = nbinom.logpmf(k_vals, alpha, p)
    # log P(Y <= t_obs - k)
    log_cdf_y = poisson.logcdf(t_obs - k_vals, mu=mu_b)
    
    # log(\sum \exp(log_z + log_cdf_y))
    return logsumexp(log_z + log_cdf_y)

def bisect_log_quantile(target_log_p, alpha, scale, b, n_sample, max_t=1_000_000):
    """
    Find the exact quantile (smallest t where log_cdf(t) >= target_log_p) using binary search.
    """
    low, high = 0, max_t
    
    while low < high:
        mid = (low + high) // 2
        log_cdf_mid = _log_cdf_pred(mid, alpha, scale, b, n_sample)
        
        if log_cdf_mid < target_log_p:
            low = mid + 1
        else:
            high = mid
            
    return low

def check_prior_data_conflict_nuisance_exact(t_obs, alpha, scale, b, n_sample):
    """Prior Predictive Check using exact Log-Space Bisection."""
    
    # 1. Exact log p-value: log(P(T <= t_obs))
    log_pvalue = _log_cdf_pred(t_obs, alpha, scale, b, n_sample)
    pvalue = np.exp(log_pvalue)
    
    # 2. Exact quantiles via binary search
    log_q_lower_target = np.log(0.025)
    log_q_upper_target = np.log(0.975)
    
    q_lower = bisect_log_quantile(log_q_lower_target, alpha, scale, b, n_sample)
    q_upper = bisect_log_quantile(log_q_upper_target, alpha, scale, b, n_sample)
    
    in_central = bool(q_lower <= t_obs <= q_upper)
    
    return pvalue, in_central, q_lower, q_upper

def _log_cdf_pred_vb(t_obs, alpha_lam, scale_lam, alpha_b, scale_b, n_sample):
    """
    Exact log CDF log(P(T <= t_obs)) using NB ⊛ NB convolution.
    T = Z_lam + Z_b, so P(T <= t) = \sum_{k=0}^t P(Z_lam = k) * P(Z_b <= t - k)
    """
    p_lam = 1.0 / (1.0 + n_sample * scale_lam)
    p_b   = 1.0 / (1.0 + n_sample * scale_b)
    
    k_vals = np.arange(0, t_obs + 1)
    
    # log P(Z_lam = k)
    log_z_lam = nbinom.logpmf(k_vals, alpha_lam, p_lam)
    
    # log P(Z_b <= t_obs - k)
    # nbinom.logcdf handles the right-side CDF sum perfectly in log-space
    log_cdf_z_b = nbinom.logcdf(t_obs - k_vals, alpha_b, p_b)
    
    # log(\sum \exp(log_z_lam + log_cdf_z_b))
    return logsumexp(log_z_lam + log_cdf_z_b)

def bisect_log_quantile_vb(target_log_p, alpha_lam, scale_lam, alpha_b, scale_b, n_sample, max_t=1_000_000):
    """
    Find the exact quantile (smallest t where log_cdf(t) >= target_log_p) using binary search.
    """
    low, high = 0, max_t
    
    while low < high:
        mid = (low + high) // 2
        log_cdf_mid = _log_cdf_pred_vb(mid, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)
        
        if log_cdf_mid < target_log_p:
            low = mid + 1
        else:
            high = mid
            
    return low

def check_prior_data_conflict_nuisance_vb_exact(t_obs, alpha_lam, scale_lam, alpha_b, scale_b, n_sample):
    """Prior Predictive Check using exact Log-Space Bisection (NB ⊛ NB)."""
    
    # 1. Exact log p-value: log(P(T <= t_obs))
    log_pvalue = _log_cdf_pred_vb(t_obs, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)
    pvalue = np.exp(log_pvalue)
    
    # 2. Exact quantiles via binary search
    log_q_lower_target = np.log(0.025)
    log_q_upper_target = np.log(0.975)
    
    q_lower = bisect_log_quantile_vb(log_q_lower_target, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)
    q_upper = bisect_log_quantile_vb(log_q_upper_target, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)
    
    in_central = bool(q_lower <= t_obs <= q_upper)
    
    return float(pvalue), in_central, float(q_lower), float(q_upper)