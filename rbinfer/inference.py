import numpy as np
from scipy.stats import gamma
from scipy.special import logsumexp

from ._likelihoods import (
    _log_lik,
    _log_prior_pred,
    _log_lik_vb,
    _log_prior_pred_vb
)

def auto_lambda_max(alpha: float, scale: float,
                     t: int, n_sample: float,
                     factor: float = 8.0) -> float:
    """
    Heuristic upper bound for the λ grid.
    Posterior mean ≈ (α + t) / (1/scale + n_sample); take `factor` × that.
    """
    posterior_mean_approx = (alpha + t) / (1.0 / scale + n_sample)
    return max(posterior_mean_approx * factor, 1e-6)


def _posterior_log_probs_on_grid(lambda_grid: np.ndarray,
                                  log_integrated_lik: np.ndarray,
                                  alpha: float, scale: float) -> np.ndarray:
    """
    Normalised log-posterior on a grid:  π(λ | t) ∝ m(t | λ) · π(λ)

    Parameters
    ----------
    lambda_grid        : 1-D array of λ values
    log_integrated_lik : log m(t | λ) at each grid point
    alpha, scale       : Gamma(α, scale) prior hyper-parameters for λ

    Returns
    -------
    log_post : 1-D array of normalised log-posterior values
    """
    log_prior       = gamma.logpdf(lambda_grid, a=alpha, scale=scale)
    log_post_unnorm = log_integrated_lik + log_prior
    log_post_unnorm -= logsumexp(log_post_unnorm)   # normalise in log-space
    return log_post_unnorm

def _rb_credible_region(lambda_grid: np.ndarray,
                         rb: np.ndarray,
                         post_probs: np.ndarray,
                         gamma: float) -> tuple[float, float]:
    """
    γ-credible relative belief region  { λ : RB(λ | t) ≥ c_γ }.
    Grid points are sorted from highest to lowest RB; posterior mass is
    accumulated until ≥ γ is reached to find the threshold c_gamma.
    
    The bounding interval is returned by interpolating the exact points
    where RB(λ) == c_gamma, ensuring equal RB ratios at both endpoints.

    Parameters
    ----------
    lambda_grid : grid of λ values
    rb          : relative belief ratios on the same grid
    post_probs  : normalised posterior probabilities on the same grid
    gamma       : desired posterior coverage in (0, 1)

    Returns
    -------
    (lower, upper) of the γ-credible relative belief interval
    """
    # 1. Find the RB threshold (c_gamma) that meets the posterior coverage
    sorted_idx = np.argsort(rb)[::-1]          # descending RB
    cumulative = np.cumsum(post_probs[sorted_idx])
    cut        = np.searchsorted(cumulative, gamma)
    
    # Safety catch for numerical edges
    cut = min(cut, len(rb) - 1)
    c_gamma    = rb[sorted_idx[cut]]

    # 2. Find the exact boundaries via interpolation
    mode_idx = np.argmax(rb)
    
    # -- Left Boundary --
    if rb[0] >= c_gamma:
        # Truncated at zero
        lam_L = float(lambda_grid[0])
    else:
        # np.interp needs the x-coordinates (RB) to be monotonically increasing
        # On the left side of the mode, RB is increasing
        rb_left = rb[:mode_idx + 1]
        lam_left = lambda_grid[:mode_idx + 1]
        lam_L = float(np.interp(c_gamma, rb_left, lam_left))
        
    # -- Right Boundary --
    if rb[-1] >= c_gamma:
        # Hits the upper limit of the defined grid
        lam_U = float(lambda_grid[-1])
    else:
        # On the right side of the mode, RB is decreasing.
        # We must reverse the arrays so RB becomes monotonically increasing for np.interp
        rb_right = rb[mode_idx:][::-1]
        lam_right = lambda_grid[mode_idx:][::-1]
        lam_U = float(np.interp(c_gamma, rb_right, lam_right))
        
    return lam_L, lam_U
# ── Case 1: Fixed (known) background ─────────────────────────────────────────


def rb_ratio_fixed_b(
    lambda_grid, t: int,
    alpha: float, scale: float,
    b: float, n_sample: float,
) -> tuple[np.ndarray, float]:
    """
    Relative belief ratio RB(λ | t) on a grid — fixed-background model.

    Model
    -----
    T | λ  ~  Poisson( n_sample · (λ + b) )
    λ      ~  Gamma(α, scale=1/rate)

    Identity
    --------
    log RB(λ | t) = log f(t | λ) − log m(t)
      log f(t | λ) = _log_lik(t, λ, b, n_sample)
      log m(t)     = _log_prior_pred(t, α, scale, b, n_sample)

    Parameters
    ----------
    lambda_grid : array-like  grid of λ values (≥ 0)
    t           : int         observed count
    alpha       : float       Gamma prior shape
    scale       : float       Gamma prior scale (= 1/rate)
    b           : float       known background rate
    n_sample    : float       exposure / sample size

    Returns
    -------
    rb      : np.ndarray  RB(λ | t) at each grid point
    log_m_t : float       log prior predictive log m(t)
    """
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    log_m_t     = _log_prior_pred(t, alpha, scale, b, n_sample)
    log_rb      = np.array(
        [_log_lik(t, lam, b, n_sample) - log_m_t for lam in lambda_grid]
    )
    return np.exp(log_rb), float(log_m_t)


def rb_plausible_interval_fixed_b(
    t: int,
    alpha: float, scale: float,
    b: float, n_sample: float,
    *,
    gamma_credible: float | None = None,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 5_0000,
    rb_cutoff: float = 1.0,
) -> dict:
    """
    Relative belief plausible interval for λ — fixed-background model.

    Plausible region  Pl(t) = { λ : RB(λ | t) > 1 }
                             = { λ : f(t | λ)  > m(t) }

    Optionally also returns a γ-credible RB region, i.e. the smallest set
    { λ : RB(λ | t) ≥ c } whose posterior probability is ≥ γ.

    Parameters
    ----------
    t              : int           observed count
    alpha          : float         Gamma prior shape for λ
    scale          : float         Gamma prior scale for λ
    b              : float         known background rate
    n_sample       : float         exposure
    gamma_credible : float | None  if given (e.g. 0.95), compute γ-credible RB
                                   region in addition to the plausible interval
    lambda_min     : float         grid lower bound  (default 0)
    lambda_max     : float         grid upper bound  (auto if None)
    n_grid         : int           number of grid points (default 5 000)

    Returns
    -------
    dict with keys
      "interval"        : (lower, upper)  plausible interval
      "rb_estimate"     : float           λ at argmax RB  (= MLE)
      "strength"        : float           Π( Pl(t) | t )
      "rb_ratio"        : np.ndarray      RB values on the grid
      "lambda_grid"     : np.ndarray      λ grid
      "log_m_t"         : float           log m(t)
      "credible_region" : (lower, upper) | None
    """
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha, scale, t, n_sample)

    lambda_grid = np.linspace(lambda_min, lambda_max, n_grid)
    rb, log_m_t = rb_ratio_fixed_b(lambda_grid, t, alpha, scale, b, n_sample)

    # plausible region
    plausible_mask = rb > rb_cutoff
    if np.any(plausible_mask):
        pl_lower = lambda_grid[np.argmax(plausible_mask)]
        pl_upper = lambda_grid[len(plausible_mask) - 1
                                - np.argmax(plausible_mask[::-1])]
    else:
        pl_lower = pl_upper = float("nan")

    # RB estimate
    rb_estimate = float(lambda_grid[np.argmax(rb)])

    # strength  S(t) = Π( Pl(t) | t )
    log_integrated_lik = np.array(
        [_log_lik(t, lam, b, n_sample) for lam in lambda_grid]
    )
    log_post  = _posterior_log_probs_on_grid(lambda_grid, log_integrated_lik,
                                              alpha, scale)
    post_probs = np.exp(log_post)
    strength   = float(post_probs[plausible_mask].sum())

    # optional γ-credible RB region
    credible_region = None
    if gamma_credible is not None:
        if not 0 < gamma_credible < 1:
            raise ValueError("gamma_credible must be in (0, 1).")
        credible_region = _rb_credible_region(
            lambda_grid, rb, post_probs, gamma_credible
        )

    return {
        "interval":        (float(pl_lower), float(pl_upper)),
        "rb_estimate":     rb_estimate,
        "strength":        strength,
        "rb_ratio":        rb,
        "lambda_grid":     lambda_grid,
        "log_m_t":         log_m_t,
        "credible_region": credible_region,
    }

# ── Case 2: Variable (unknown) background ────────────────────────────────────

def rb_ratio_variable_b(
    lambda_grid, t: int,
    alpha_lam: float, scale_lam: float,
    alpha_b: float, scale_b: float,
    n_sample: float,
) -> tuple[np.ndarray, float]:
    """
    Relative belief ratio RB(λ | t) on a grid — variable-background model.

    Model
    -----
    T | λ, b  ~  Poisson( n_sample · (λ + b) )
    λ         ~  Gamma(α_λ, scale_λ)
    b         ~  Gamma(α_b, scale_b)

    RB is obtained by marginalising over b:
        log RB(λ | t) = log m(t | λ) − log m(t)

      log m(t | λ) = _log_lik_vb(t, λ, α_b, scale_b, n_sample)
      log m(t)     = _log_prior_pred_vb(t, α_λ, scale_λ, α_b, scale_b, n)

    Parameters
    ----------
    lambda_grid : array-like  grid of λ values (≥ 0)
    t           : int         observed count
    alpha_lam   : float       Gamma prior shape for λ
    scale_lam   : float       Gamma prior scale for λ
    alpha_b     : float       Gamma prior shape for b
    scale_b     : float       Gamma prior scale for b
    n_sample    : float       exposure / sample size

    Returns
    -------
    rb      : np.ndarray  RB(λ | t) at each grid point
    log_m_t : float       log m(t)
    """
    lambda_grid = np.asarray(lambda_grid, dtype=float)
    log_m_t = _log_prior_pred_vb(t, alpha_lam, scale_lam,
                                   alpha_b, scale_b, n_sample)
    log_rb = np.array(
        [_log_lik_vb(t, lam, alpha_b, scale_b, n_sample) - log_m_t
         for lam in lambda_grid]
    )
    return np.exp(log_rb), float(log_m_t)


def rb_plausible_interval_variable_b(
    t: int,
    alpha_lam: float, scale_lam: float,
    alpha_b: float, scale_b: float,
    n_sample: float,
    *,
    gamma_credible: float | None = None,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 5_000,
) -> dict:
    """
    Relative belief plausible interval for λ — variable-background model.

    Plausible region  Pl(t) = { λ : RB(λ | t) > 1 }
                             = { λ : m(t | λ)  > m(t) }

    m(t | λ) = ∫ f(t | λ, b) π(b) db  is the b-marginalised likelihood.

    Parameters
    ----------
    t              : int           observed count
    alpha_lam      : float         Gamma prior shape for λ
    scale_lam      : float         Gamma prior scale for λ
    alpha_b        : float         Gamma prior shape for b
    scale_b        : float         Gamma prior scale for b
    n_sample       : float         exposure
    gamma_credible : float | None  optional γ-credible RB region
    lambda_min     : float         grid lower bound  (default 0)
    lambda_max     : float         grid upper bound  (auto if None)
    n_grid         : int           number of grid points (default 5 000)

    Returns
    -------
    dict  (same keys as rb_plausible_interval_fixed_b)
    """
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha_lam, scale_lam, t, n_sample)

    lambda_grid = np.linspace(lambda_min, lambda_max, n_grid)
    rb, log_m_t = rb_ratio_variable_b(
        lambda_grid, t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample
    )

    # plausible region
    plausible_mask = rb > 1.0
    if np.any(plausible_mask):
        pl_lower = lambda_grid[np.argmax(plausible_mask)]
        pl_upper = lambda_grid[len(plausible_mask) - 1
                                - np.argmax(plausible_mask[::-1])]
    else:
        pl_lower = pl_upper = float("nan")

    # RB estimate
    rb_estimate = float(lambda_grid[np.argmax(rb)])

    # strength
    log_integrated_lik = np.array(
        [_log_lik_vb(t, lam, alpha_b, scale_b, n_sample)
         for lam in lambda_grid]
    )
    log_post   = _posterior_log_probs_on_grid(lambda_grid, log_integrated_lik,
                                               alpha_lam, scale_lam)
    post_probs = np.exp(log_post)
    strength   = float(post_probs[plausible_mask].sum())

    # optional γ-credible RB region
    credible_region = None
    if gamma_credible is not None:
        if not 0 < gamma_credible < 1:
            raise ValueError("gamma_credible must be in (0, 1).")
        credible_region = _rb_credible_region(
            lambda_grid, rb, post_probs, gamma_credible
        )

    return {
        "interval":        (float(pl_lower), float(pl_upper)),
        "rb_estimate":     rb_estimate,
        "strength":        strength,
        "rb_ratio":        rb,
        "lambda_grid":     lambda_grid,
        "log_m_t":         log_m_t,
        "credible_region": credible_region,
    }

# Helper Functions for Sim
def rb_plausible_interval_fixed_b_fast_accurate(
    t: int,
    alpha: float, scale: float,
    b: float, n_sample: float,
    *,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 50_000,
    n_grid_coarse: int = 500,       # new: coarse grid for strength
    refine_width: float = 0.05,     # new: fraction of grid range to refine around boundaries
    rb_cutoff: float = 1.0,
) -> dict:
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha, scale, t, n_sample)

    # ── Phase 1: COARSE grid — find interval endpoints ──────────────────────
    grid_coarse = np.linspace(lambda_min, lambda_max, n_grid_coarse)
    rb_coarse, _ = rb_ratio_fixed_b(grid_coarse, t, alpha, scale, b, n_sample)

    plausible_mask_c = rb_coarse > rb_cutoff
    idx_c = np.where(plausible_mask_c)[0]

    if idx_c.size:
        # Bracket the true boundaries: one step outside on each side
        lo_idx = max(idx_c[0] - 1, 0)
        hi_idx = min(idx_c[-1] + 1, n_grid_coarse - 1)
        coarse_step = (lambda_max - lambda_min) / (n_grid_coarse - 1)
        refine_half = max(refine_width * (lambda_max - lambda_min), coarse_step * 2)

        lo_bracket = (
            max(lambda_min, grid_coarse[lo_idx] - refine_half),
            min(lambda_max, grid_coarse[lo_idx] + refine_half),
        )
        hi_bracket = (
            max(lambda_min, grid_coarse[hi_idx] - refine_half),
            min(lambda_max, grid_coarse[hi_idx] + refine_half),
        )

        # ── Phase 2: FINE grid — refine each boundary independently ─────────
        n_refine = n_grid // 10          # e.g. 5 000 pts per boundary
        lo_fine  = np.linspace(*lo_bracket, n_refine)
        hi_fine  = np.linspace(*hi_bracket, n_refine)

        rb_lo, _ = rb_ratio_fixed_b(lo_fine, t, alpha, scale, b, n_sample)
        rb_hi, _ = rb_ratio_fixed_b(hi_fine, t, alpha, scale, b, n_sample)

        mask_lo = rb_lo > rb_cutoff
        mask_hi = rb_hi > rb_cutoff

        idx_lo = np.where(mask_lo)[0]
        idx_hi = np.where(mask_hi)[0]

        pl_lower = float(lo_fine[idx_lo[0]])  if idx_lo.size else float("nan")
        pl_upper = float(hi_fine[idx_hi[-1]]) if idx_hi.size else float("nan")

    else:
        pl_lower = pl_upper = float("nan")

    return {
        "interval": (pl_lower, pl_upper),
    }

def rb_plausible_interval_variable_b_fast_accurate(
    t: int,
    alpha_lam: float, scale_lam: float,
    alpha_b: float, scale_b: float,
    n_sample: float,
    *,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 5_000,
    n_grid_coarse: int = 300,
    refine_width: float = 0.05,
) -> dict:
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha_lam, scale_lam, t, n_sample)

    # ── Phase 1: COARSE grid — find boundary brackets ───────────────────────
    grid_coarse = np.linspace(lambda_min, lambda_max, n_grid_coarse)
    rb_coarse, _ = rb_ratio_variable_b(
        grid_coarse, t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample
    )

    plausible_mask_c = rb_coarse > 1.0
    idx_c = np.where(plausible_mask_c)[0]

    if idx_c.size:
        coarse_step  = (lambda_max - lambda_min) / (n_grid_coarse - 1)
        refine_half  = max(refine_width * (lambda_max - lambda_min), coarse_step * 2)

        lo_idx = max(idx_c[0]  - 1, 0)
        hi_idx = min(idx_c[-1] + 1, n_grid_coarse - 1)

        lo_bracket = (
            max(lambda_min, grid_coarse[lo_idx] - refine_half),
            min(lambda_max, grid_coarse[lo_idx] + refine_half),
        )
        hi_bracket = (
            max(lambda_min, grid_coarse[hi_idx] - refine_half),
            min(lambda_max, grid_coarse[hi_idx] + refine_half),
        )

        # ── Phase 2: FINE grid — refine each boundary ───────────────────────
        n_refine = max(n_grid // 10, 200)
        lo_fine  = np.linspace(*lo_bracket, n_refine)
        hi_fine  = np.linspace(*hi_bracket, n_refine)

        rb_lo, _ = rb_ratio_variable_b(lo_fine, t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)
        rb_hi, _ = rb_ratio_variable_b(hi_fine, t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample)

        idx_lo = np.where(rb_lo > 1.0)[0]
        idx_hi = np.where(rb_hi > 1.0)[0]

        pl_lower = float(lo_fine[idx_lo[0]])  if idx_lo.size else float("nan")
        pl_upper = float(hi_fine[idx_hi[-1]]) if idx_hi.size else float("nan")

    else:
        pl_lower = pl_upper = float("nan")

    return {
        "interval": (pl_lower, pl_upper),
    }

def strength_at_a_point_fixed_b(
    t: int,
    alpha_lam: float, scale_lam: float,
    b: float, n_sample: float,
    lambda_null: float | None = None,
    *,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 5_000
) -> dict:
    """
    Relative belief ratio and evidence for/against a \lambda_0 
    for λ — fixed-background model.

    Strength of evidence  Pl(t) = { λ : RB(λ | t) < RB(λ_0 | t) }
                                
    Parameters
    ----------
    t              : int           observed count
    alpha_lam      : float         Gamma prior shape for λ
    scale_lam      : float         Gamma prior scale for λ
    b              : float         known background rate
    n_sample       : float         exposure
    lambda_min     : float         grid lower bound  (default 0)
    lambda_max     : float         grid upper bound  (auto if None)
    n_grid         : int           number of grid points (default 5 000)
    lambda_null    : float         lambda value for H_0

    Returns
    -------
    dict  
    """
    
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha_lam, scale_lam, t, n_sample)

    lambda_grid = np.linspace(lambda_min, lambda_max, n_grid)
    cutoff = rb_ratio_fixed_b(
    [lambda_null],t,
    alpha_lam, scale_lam,
    b, n_sample)[0][0]
    print(cutoff)
    rb, log_m_t = rb_ratio_fixed_b(
        lambda_grid, t, alpha_lam, scale_lam, b, n_sample
    )
    log_integrated_lik = np.array(
        [_log_lik(t, lam, b, n_sample)
         for lam in lambda_grid]
    )
    log_post   = _posterior_log_probs_on_grid(lambda_grid, log_integrated_lik,
                                               alpha_lam, scale_lam)
    post_probs = np.exp(log_post)

    strength = None
    e_string = "Neither In Favor or Against"
    eif_pl_lower = None
    eif_pl_upper = None
    if float(cutoff) == 1.0:
        return {
        "evidence":        e_string,
        "eif_interval":    (eif_pl_lower, eif_pl_upper),
        "strength": strength,
        }

    # plausible region
    eif_plausible_mask = rb >= cutoff
    if np.any(eif_plausible_mask):                          # ← fixed mask
        eif_pl_lower = lambda_grid[np.argmax(eif_plausible_mask)]
        eif_pl_upper = lambda_grid[len(eif_plausible_mask) - 1
                                - np.argmax(eif_plausible_mask[::-1])]
    else:
        eif_pl_lower = eif_pl_upper = float("nan")
    
    strength = 1-float(post_probs[eif_plausible_mask].sum())
    e_string = "Against. Small value of strength indicates strong evidence against."
    if cutoff > 1:    
        e_string = "In Favor. Large value of strength indicates strong evidence in favor."
    return {
        "rb_ratio": cutoff,
        "evidence": e_string,
        "strength": strength,
    }

def strength_at_a_point_variable_b(
    t: int,
    alpha_lam: float, scale_lam: float,
    alpha_b: float, scale_b: float,
    n_sample: float,
    lambda_null: float | None = None,
    *,
    lambda_min: float = 0.0,
    lambda_max: float | None = None,
    n_grid: int = 5_000
) -> dict:
    """
    Relative belief ratio and evidence for/against a \lambda_0 
    for λ — variable-background model.

    Strength of evidence  Pl(t) = { λ : RB(λ | t) < RB(λ_0 | t) }


    m(t | λ) = ∫ f(t | λ, b) π(b) db  is the b-marginalised likelihood.

    Parameters
    ----------
    t              : int           observed count
    alpha_lam      : float         Gamma prior shape for λ
    scale_lam      : float         Gamma prior scale for λ
    alpha_b        : float         Gamma prior shape for b
    scale_b        : float         Gamma prior scale for b
    n_sample       : float         exposure
    lambda_min     : float         grid lower bound  (default 0)
    lambda_max     : float         grid upper bound  (auto if None)
    n_grid         : int           number of grid points (default 5 000)
    lambda_null    : float         lambda value for H_0

    Returns
    -------
    dict  
    """
    
    if lambda_max is None:
        lambda_max = auto_lambda_max(alpha_lam, scale_lam, t, n_sample)

    lambda_grid = np.linspace(lambda_min, lambda_max, n_grid)
    cutoff = rb_ratio_variable_b(
    [lambda_null],t,
    alpha_lam, scale_lam,
    alpha_b, scale_b,
    n_sample)[0][0]
    print(cutoff)
    rb, log_m_t = rb_ratio_variable_b(
        lambda_grid, t, alpha_lam, scale_lam, alpha_b, scale_b, n_sample
    )
    log_integrated_lik = np.array(
        [_log_lik_vb(t, lam, alpha_b, scale_b, n_sample)
         for lam in lambda_grid]
    )
    log_post   = _posterior_log_probs_on_grid(lambda_grid, log_integrated_lik,
                                               alpha_lam, scale_lam)
    post_probs = np.exp(log_post)

    strength = None
    e_string = "Neither In Favor or Against"
    eif_pl_lower = None
    eif_pl_upper = None
    if float(cutoff) == 1.0:
        return {
        "evidence":        e_string,
        "eif_interval":    (eif_pl_lower, eif_pl_upper),
        "strength": strength,
        }

    # plausible region
    eif_plausible_mask = rb >= cutoff
    if np.any(eif_plausible_mask):                          # ← fixed mask
        eif_pl_lower = lambda_grid[np.argmax(eif_plausible_mask)]
        eif_pl_upper = lambda_grid[len(eif_plausible_mask) - 1
                                - np.argmax(eif_plausible_mask[::-1])]
    else:
        eif_pl_lower = eif_pl_upper = float("nan")
    
    strength = 1-float(post_probs[eif_plausible_mask].sum())
    e_string = "Against. Small value of strength indicates strong evidence against."
    if cutoff > 1:    
        e_string = "In Favor. Large value of strength indicates strong evidence in favor."
    return {
        "rb_ratio": cutoff,
        "evidence": e_string,
        "strength": strength,
    }
