import math
import warnings
from scipy.optimize import brentq
from scipy.stats import gamma, norm

def elicit_exp_rate(x_lower, x_upper, target_prob, which="left"):
    """
    Find the rate β of Exp(β) such that P(x_lower < X < x_upper) = target_prob.

    P(β) = e^(−x_lower·β) − e^(−x_upper·β) is unimodal in β with maximum
    p_max at β* = log(x_upper/x_lower) / (x_upper − x_lower).
    For target_prob < p_max, two solutions exist:
      - "left"  (β < β*): lighter concentration, heavier tail — default
      - "right" (β > β*): higher concentration near 0

    Note: when x_lower ≈ 0, the right root is degenerate (β → ∞) and "left"
    is the only meaningful choice.

    Parameters
    ----------
    x_lower     : float ≥ 0 — lower bound of the credible interval
    x_upper     : float > x_lower — upper bound
    target_prob : float in (0, 1) — desired probability content
    which       : {"left", "right"} — which root to return (default "left")

    Returns
    -------
    float : rate β
    """
    if x_lower < 0 or x_upper <= x_lower:
        raise ValueError(f"Need 0 ≤ x_lower < x_upper, got ({x_lower}, {x_upper}).")
    if not 0 < target_prob < 1:
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob}.")

    # ── Degenerate case: x_lower = 0 → single root ───────────────────────
    if x_lower == 0:
        return -math.log(1.0 - target_prob) / x_upper

    def prob_content(beta):
        return math.exp(-x_lower * beta) - math.exp(-x_upper * beta)

    beta_star = math.log(x_upper / x_lower) / (x_upper - x_lower)
    p_max     = prob_content(beta_star)

    if target_prob >= p_max - 1e-12:
        if abs(target_prob - p_max) < 1e-10:
            return beta_star
        raise ValueError(
            f"target_prob={target_prob:.6f} is infeasible; "
            f"maximum achievable is p_max={p_max:.6f} at β*={beta_star:.6f}."
        )

    def f(beta):
        return prob_content(beta) - target_prob

    # ── Dynamic upper bound: double from β* until P drops below target ───
    beta_upper = 2.0 * beta_star
    for _ in range(200):                        # hard cap — avoids infinite loop
        if prob_content(beta_upper) <= target_prob:
            break
        beta_upper *= 2.0

    roots = {}
    try:
        roots["left"]  = brentq(f, 1e-14, beta_star - 1e-12, xtol=1e-14)
    except ValueError:
        pass
    try:
        roots["right"] = brentq(f, beta_star + 1e-12, beta_upper, xtol=1e-14)
    except ValueError:
        pass

    if not roots:
        return beta_star

    if which in roots:
        return roots[which]

    available = next(iter(roots))
    warnings.warn(
        f"Root '{which}' not found; returning '{available}' root. "
        f"When x_lower ≈ 0 the right root is very large and numerically "
        f"indistinct — use which='left' for this scenario."
    )
    return roots[available]

def elicit_gamma_prior(mode_target, x_lower, x_upper,
                       target_prob=0.99, verbose=False):
    """
    Find Gamma(α, rate) parameters such that mode = mode_target and
    P(x_lower < X < x_upper) = target_prob.

    Uses brentq on P(α) = target_prob directly (more robust than minimising
    the squared residual, which fails when P plateaus numerically at 1.0).

    Parameters
    ----------
    mode_target : float > 0 — desired mode of the Gamma prior
    x_lower     : float ≥ 0 — lower bound of the credible interval
    x_upper     : float > x_lower — upper bound
    target_prob : float in (0, 1) — desired probability content (default 0.99)
    verbose     : bool — print solved (α, scale, rate) if True (default False)

    Returns
    -------
    (alpha, beta) : tuple[float, float]
        Shape α and rate (= 1/scale) of the Gamma prior.

    Raises
    ------
    ValueError if inputs are invalid or no solution exists in the search range.
    """
    if mode_target <= 0:
        raise ValueError(f"mode_target must be > 0, got {mode_target}.")
    if x_lower < 0 or x_upper <= x_lower:
        raise ValueError(f"Need 0 ≤ x_lower < x_upper, got ({x_lower}, {x_upper}).")
    if not 0 < target_prob < 1:
        raise ValueError(f"target_prob must be in (0, 1), got {target_prob}.")

    def prob_content(alpha):
        scale = mode_target / (alpha - 1)
        return (gamma.cdf(x_upper, a=alpha, scale=scale)
              - gamma.cdf(x_lower, a=alpha, scale=scale))

    # ── Adaptive upper bound via normal approximation (×20 safety) ────────
    z = norm.ppf(target_prob)
    spread = max(x_upper - mode_target, x_upper * 0.05)   # avoid /0
    alpha_upper = max(200.0, (1 + (z * mode_target / spread) ** 2) * 20)

    # Verify feasibility (P must cross target_prob within search range)
    p_lo = prob_content(1.01)
    p_hi = prob_content(alpha_upper)
    if p_lo >= target_prob:
        raise ValueError(
            f"target_prob={target_prob:.4f} is already exceeded at α=1.01 "
            f"(P={p_lo:.4f}). Lower target_prob or widen the interval."
        )
    if p_hi < target_prob:
        raise ValueError(
            f"target_prob={target_prob:.4f} is not reachable up to α={alpha_upper:.0f} "
            f"(max P={p_hi:.6f}). Check that mode_target is inside [x_lower, x_upper]."
        )

    alpha_sol = brentq(
        lambda a: prob_content(a) - target_prob,
        1.01, alpha_upper,
        xtol=1e-12,
    )
    scale_sol = mode_target / (alpha_sol - 1)

    if verbose:
        print(f"Gamma prior: α={alpha_sol:.4f}, scale={scale_sol:.4f}, "
              f"rate={1/scale_sol:.4f}, mode={mode_target:.4f}")

    return alpha_sol, 1/scale_sol

def poisson_prefactor(t):
    """c_t = sup_u P(Poisson(u)=t-1)."""
    if t < 1 or int(t) != t:
        raise ValueError("t must be a positive integer")
    if t == 1:
        return 1.0
    k = t - 1
    # use logs for numerical stability
    log_c = -k + k * math.log(k) - math.lgamma(k + 1)
    return math.exp(log_c)

def gamma_mad(alpha, beta):
    """Exact E|b - E[b]| for b ~ Gamma(alpha, beta) with rate beta."""
    log_mad = math.log(2.0) + alpha * math.log(alpha) - alpha - math.log(beta) - math.lgamma(alpha)
    return math.exp(log_mad)

def bound(t, alpha, beta):
    return poisson_prefactor(t) * gamma_mad(alpha, beta)

