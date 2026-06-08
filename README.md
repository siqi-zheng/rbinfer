# rbinfer

A Python library for **Relative Belief (RB) inference** in Poisson signal-plus-background counting experiments, implementing the Evans (2015) framework.

The library covers the full workflow: prior elicitation → experimental design (sample size calculation) → prior-data conflict checking → RB ratio computation → plausible intervals → bias-in-favor / bias-against.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/siqi-zheng/rbinfer.git
cd rbinfer
pip install -e .
```

To include development/test dependencies:

```bash
pip install -e ".[dev]"
```

**Requires Python >= 3.10.**

---

## Quickstart

```python
from rbinfer import *

# 1. Elicit a Gamma prior: mode = 5, 99% mass in [3, 10]
alpha, rate = elicit_gamma_prior(mode_target=5.0, x_lower=3.0, x_upper=10.0, verbose=True)
scale = 1.0 / rate

# 2. Compute the RB plausible interval for observed count t = 12
result = rb_plausible_interval_fixed_b(
    t=12, alpha=alpha, scale=scale,
    b=2.0, n_sample=1.0,
    gamma_credible=0.95,
)

print(result["interval"])        # plausible interval  { λ : RB(λ|t) > 1 }
print(result["rb_estimate"])     # MLE  =  argmax RB(λ|t)
print(result["strength"])        # S(t) = Π( Pl(t) | t )
print(result["credible_region"]) # 95 % γ-credible RB interval

# 3. Prior-data conflict check
pvalue, in_central, q_lo, q_hi = check_prior_data_conflict_nuisance_exact(
    t_obs=12, alpha=alpha, scale=scale, b=2.0, n_sample=1.0
)
print(f"p-value = {pvalue:.4f}, 95% predictive band = [{q_lo}, {q_hi}]")
```

---

## Module Overview

| Module | Public symbols |
|---|---|
| `priors` | `elicit_exp_rate`, `elicit_gamma_prior` |
| `inference` | `rb_ratio_fixed_b`, `rb_plausible_interval_fixed_b`, `rb_ratio_variable_b`, `rb_plausible_interval_variable_b`, `rb_plausible_interval_fixed_b_fast_accurate`, `rb_plausible_interval_variable_b_fast_accurate` |
| `predictive_checks` | `check_prior_data_conflict_nuisance_exact`, `check_prior_data_conflict_nuisance_vb_exact` |
| `bias` | `compute_bif`, `compute_baga` |
| `simulation` | `run_bif_sweep`, `run_baga_sweep`, `sweep_and_plot_bif`, `sweep_and_plot_baga` |
| `_likelihoods` *(private)* | `_log_lik`, `_log_prior_pred`, `_log_lik_vb`, `_log_prior_pred_vb` |
| `_compat` *(private)* | `find_rscript` (Windows helper) |

---

## Statistical Background

### Model

Two observation models are supported:

**Fixed background** — background rate `b` is known:

$$T \mid \lambda \sim \text{Poisson}\bigl(n \cdot (\lambda + b)\bigr), \qquad \lambda \sim \text{Gamma}(\alpha, \text{scale})$$

**Variable background** — background rate `b` is unknown with its own prior:

$$T \mid \lambda, b \sim \text{Poisson}\bigl(n \cdot (\lambda + b)\bigr), \quad \lambda \sim \text{Gamma}(\alpha_\lambda, s_\lambda), \quad b \sim \text{Gamma}(\alpha_b, s_b)$$

### Relative Belief Ratio

The core identity (Evans 2015):

$$\text{RB}(\lambda \mid t) = \frac{\pi(\lambda \mid t)}{\pi(\lambda)} = \frac{m(t \mid \lambda)}{m(t)}$$

where $m(t \mid \lambda)$ is the marginal likelihood and $m(t)$ is the prior predictive.

### Plausible Region and Strength

$$\text{Pl}(t) = \{\lambda : \text{RB}(\lambda \mid t) > 1\}$$

$$S(t) = \Pi\bigl(\text{Pl}(t) \mid t\bigr) \quad \text{(posterior probability of the plausible region)}$$

### Hypothesis Testing and Strength

Evidence in favor of $H_0: \lambda_0 = 0$ if $\text{RB}(\lambda_0 \mid t) > 1$
Evidence against $H_0: \lambda_0 = 0$ if $\text{RB}(\lambda_0 \mid t) < 1$

$$S(t) = \Pi\bigl( \text{RB}(\lambda \mid t) \leq \text{RB}(\lambda_0 \mid t) \bigr)$$

That is, the posterior probability of the true value of $\lambda$ has a relative belief ratio no larger than that obtained for $\lambda_0$.

When there is evidence against the value $\lambda_0$, then a __small__ value of strength indicates a large belief that the true value of $\lambda$ is in the set $\{\lambda : \text{RB}(\lambda \mid t) \leq \text{RB}(\lambda_0 \mid t)\}$ and so there is __strong__ evidence against $\lambda_0$. 

When there is evidence in favor of the value $\lambda_0$, then a __small__ value of strength indicates a large belief that the true value of $\lambda$ is in the set $\{\lambda : \text{RB}(\lambda \mid t) \leq \text{RB}(\lambda_0 \mid t)\}$ and so there is __weak__ evidence in favor of $\lambda_0$.
 
### Bias-in-Favor and Bias-Against

For hypothesis $H_0 : \lambda = \lambda_0$ evaluated at a perturbation $\pm\delta$:

$$\text{BiF}_\pm(\lambda_0) = \sum_{t : \text{RB}(\lambda_0 \mid t) \geq 1} f(t \mid \lambda_0 \pm \delta)$$

$$\text{BaGA}(\lambda_0) = \sum_{t : \text{RB}(\lambda_0 \mid t) < 1} f(t \mid \lambda_0)$$

### Prior Marginal — Exact Convolution

For the fixed-background model, the marginal $m(t)$ is computed exactly as a **Negative Binomial ⊛ Poisson** convolution in log-space via `logsumexp`. For the variable-background model, a **Negative Binomial ⊛ Negative Binomial** convolution is used.

---

## Prior Elicitation

```python
from rbinfer import elicit_gamma_prior, elicit_exp_rate

# Gamma prior: mode = 8, 95% mass in [6.4, 9.6]
alpha, rate = elicit_gamma_prior(mode_target=8.0, x_lower=6.4, x_upper=9.6,
                                  target_prob=0.95, verbose=True)

# Exponential prior: 95% mass in (0, 10]
rate_exp = elicit_exp_rate(x_lower=0.0, x_upper=10.0, target_prob=0.95)
```

---

## Bias Sweeps

```python
import numpy as np
from rbinfer import elicit_exp_rate
from rbinfer.simulation import sweep_and_plot_bif, sweep_and_plot_baga

alpha, beta = 1.0, elicit_exp_rate(1e-6, 10.0, 0.95)
scale = 1.0 / beta
grid = np.arange(1.0, 10.0, 1.0)

sweep_and_plot_bif(
    n_sample_list=[10, 20, 50],
    lambda_grid=grid,
    alpha=alpha, scale=scale,
    b=8.0, delta=1.0,
    base_output_dir="output/bif",
)
```

Results are saved as `bias_in_favor.csv` and `bias_in_favor_summary.csv` per run, with an overlay plot saved as a PNG.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy >= 1.24` | Array operations, grid computations |
| `scipy >= 1.10` | `poisson`, `nbinom`, `gamma`, `norm`, `brentq`, `logsumexp` |
| `pandas >= 2.0` | Sweep results, CSV I/O |
| `matplotlib >= 3.7` | Bias curve plots |
| `tqdm >= 4.60` | Parallel sweep progress bars |

> **Note:** `winreg` (used in `_compat.py` for `find_rscript`) to run the r script `run_fcci.R` is a Windows standard library module and is not listed as a pip dependency.

---

## Reference

Evans, M. (2015). *Measuring Statistical Evidence Using Relative Belief*. CRC Press / Chapman & Hall. ISBN 978-1-4822-1826-2.
