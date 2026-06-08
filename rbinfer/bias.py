import numpy as np
from scipy.stats import poisson, nbinom

from ._likelihoods import (
    _log_lik,
    _log_prior_pred,
    _log_lik_vb,
    _log_prior_pred_vb
)

def compute_bif(
    lambda_0, alpha, scale, b, n_sample, delta, t_max=None
):
    """
    Computes the bias in favor of lambda_0 evaluated under true lambda_0 +/- delta.
    """
        # ── Validate and unpack b ────────────────────────────────────────────────
    if isinstance(b, (list, tuple, np.ndarray)):
        if len(b) != 2:
            raise ValueError(
                f"`b` must be a single number or a 2-element list [alpha_b, scale_b], "
                f"got a sequence of length {len(b)}."
            )
        alpha_b, scale_b = b
        b_mean = alpha_b * scale_b
        b_random = True
    elif isinstance(b, (int, float, np.floating, np.integer)):
        b_random = False
        b_mean = b
    else:
        raise TypeError(
            f"`b` must be a number or a 2-element list [alpha_b, scale_b], "
            f"got {type(b).__name__}."
        )
    
    if t_max is None:
        # if b_random:
        #     # Var(T|λ) = n·λ + Var(NB) where Var(NB) = α_b*(1-p_b)/p_b²
        #     p_b = 1.0 / (1.0 + n_sample * scale_b)
        #     # Fully correct — use variance at λ₀+δ, not λ₀
        #     var_t = n_sample * (lambda_0 + delta) + alpha_b * (1 - p_b) / p_b**2


        #     std_t = np.sqrt(var_t)
        #     mean_t_plus = n_sample * (lambda_0 + delta + b_mean)
        #     t_max = int(np.ceil(mean_t_plus + 6 * std_t))
        if b_random:
            p_b = 1.0 / (1.0 + n_sample * scale_b)
            
            # Calculate exact 99.999% quantiles for both components
            mu_plus = n_sample * (lambda_0 + delta)
            t_max_lambda = poisson(mu=mu_plus).ppf(0.99999) if np.isfinite(mu_plus) else 0.0
            t_max_b = nbinom(n=alpha_b, p=p_b).ppf(0.99999)
            
            # The sum of quantiles safely bounds the total convolution
            t_max = int(np.ceil(t_max_lambda + t_max_b))
        else:    
            # Ensure the center of the null and alternatives are fully covered
            mu_0     = max(0.0, n_sample * (lambda_0 + b_mean))
            mu_plus  = max(0.0, n_sample * (lambda_0 + delta + b_mean))
            mu_minus = max(0.0, n_sample * (lambda_0 - delta + b_mean))
            
            # 0.99999 cutoff guarantees virtually zero truncation error
            t_0     = poisson(mu=mu_0).ppf(0.99999) if np.isfinite(mu_0) else 0.0
            t_plus  = poisson(mu=mu_plus).ppf(0.99999) if np.isfinite(mu_plus) else 0.0
            t_minus = poisson(mu=mu_minus).ppf(0.99999) if np.isfinite(mu_minus) else 0.0
            t_max   = int(np.ceil(max(t_0, t_plus, t_minus)))

    bif_plus = 0.0
    bif_minus = 0.0

    if b_random:
        for t in range(t_max + 1):    
            log_m_t     = _log_prior_pred_vb(t, alpha, scale, alpha_b, scale_b, n_sample)
            log_f_0     = _log_lik_vb(t, lambda_0,         alpha_b, scale_b, n_sample)
            log_f_plus  = _log_lik_vb(t, lambda_0 + delta, alpha_b, scale_b, n_sample)
            log_f_minus = _log_lik_vb(t, lambda_0 - delta, alpha_b, scale_b, n_sample)
            # Evidence in favor of lambda_0 is when RB(lambda_0 | t) >= 1
        # i.e., f(t | lambda_0) >= m(t)
            if log_f_0 - log_m_t >= 0.0:
                bif_plus  += np.exp(log_f_plus)
                bif_minus += np.exp(log_f_minus)
    else:
        for t in range(t_max + 1):
            
            log_m_t = _log_prior_pred(t, alpha, scale, b, n_sample)
            
            # The Relative Belief ratio is evaluated at the null hypothesis lambda_0
            log_f_0 = _log_lik(t, lambda_0, b, n_sample)
            
            # We need the likelihoods under the alternatives to accumulate their probability
            log_f_plus  = _log_lik(t, lambda_0 + delta, b, n_sample)
            log_f_minus = _log_lik(t, lambda_0 - delta, b, n_sample)

            # Evidence in favor of lambda_0 is when RB(lambda_0 | t) >= 1
            # i.e., f(t | lambda_0) >= m(t)
            if log_f_0 - log_m_t >= 0.0:
                bif_plus  += np.exp(log_f_plus)
                bif_minus += np.exp(log_f_minus)
    if lambda_0 - delta <= 0:
        bif_minus = bif_plus
    return bif_plus, bif_minus

def compute_baga(
    lambda_0, alpha, scale, b, n_sample, t_max=None
):
    """
    Computes the bias against of lambda_0 evaluated under true lambda_0.
    """
        # ── Validate and unpack b ────────────────────────────────────────────────
    if isinstance(b, (list, tuple, np.ndarray)):
        if len(b) != 2:
            raise ValueError(
                f"`b` must be a single number or a 2-element list [alpha_b, scale_b], "
                f"got a sequence of length {len(b)}."
            )
        alpha_b, scale_b = b
        b_mean = alpha_b*scale_b
        b_random = True
    elif isinstance(b, (int, float, np.floating, np.integer)):
        b_random = False
        b_mean = b
    else:
        raise TypeError(
            f"`b` must be a number or a 2-element list [alpha_b, scale_b], "
            f"got {type(b).__name__}."
        )
    
    if t_max is None:
        # if b_random:
        #     # Var(T|λ) = n·λ + Var(NB) where Var(NB) = α_b*(1-p_b)/p_b²
        #     p_b = 1.0 / (1.0 + n_sample * scale_b)
        #     var_t = n_sample * lambda_0 + alpha_b * (1 - p_b) / p_b**2

        #     std_t = np.sqrt(var_t)
        #     mean_t = n_sample * (lambda_0 + b_mean)
        #     # Use mean + 6σ as a safe upper bound
        #     t_max = int(np.ceil(mean_t + 6 * std_t))
        if b_random:
            p_b = 1.0 / (1.0 + n_sample * scale_b)
            
            mu_0 = n_sample * lambda_0
            t_max_lambda = poisson(mu=mu_0).ppf(0.99999) if np.isfinite(mu_0) else 0.0
            t_max_b = nbinom(n=alpha_b, p=p_b).ppf(0.99999)
            
            t_max = int(np.ceil(t_max_lambda + t_max_b))
        else:        
            # Ensure the center of the null and alternatives are fully covered
            mu_0     = max(0.0, n_sample * (lambda_0 + b_mean))
            
            # 0.99999 cutoff guarantees virtually zero truncation error
            t_max     = poisson(mu=mu_0).ppf(0.99999) if np.isfinite(mu_0) else 0.0
        
    baga = 0

    if b_random:
        for t in range(int(np.ceil(t_max)) + 1):    
            # b_random branch of compute_baga — swap the assignments:
            log_m_t_exact = _log_prior_pred_vb(t, alpha, scale, alpha_b, scale_b, n_sample)      # m(t)
            log_f_0_exact = _log_lik_vb(t, lambda_0, alpha_b, scale_b, n_sample)  # m(t|λ_0)

            if log_f_0_exact - log_m_t_exact <= 0.0:   # m(t|λ_0) < m(t) → RB < 1 → against ✓
                baga += np.exp(log_f_0_exact)  # accumulate m(t|λ_0) ✓
    
    else:
        for t in range(int(np.ceil(t_max)) + 1):
            log_f_0_exact = _log_lik(t, lambda_0, b, n_sample)
            log_m_t_exact = _log_prior_pred(t, alpha, scale, b, n_sample)
            
            if log_f_0_exact - log_m_t_exact <= 0.0:
                baga += np.exp(log_f_0_exact)

    return baga

def _worker_bif(args):
    lambda_0, alpha, scale, b, n_sample, delta, t_max = args
    bif_plus, bif_minus = compute_bif(
        lambda_0, alpha, scale, b, n_sample, delta, t_max=t_max
    )
    bif_max = max(bif_plus, bif_minus)
    return lambda_0, bif_plus, bif_minus, bif_max

def _worker_baga(args):
    lambda_0, alpha, scale, b, n_sample, t_max = args
    baga = compute_baga(
        lambda_0, alpha, scale, b, n_sample, t_max=t_max
    )
    return lambda_0, baga
