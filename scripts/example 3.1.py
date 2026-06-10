import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import poisson, gamma
from fcci_v2 import FC_poisson
from rbinfer import *
# ==========================================
# 1. SCENARIO CONFIGURATION
# ==========================================
# Physics Context: L = 100 fb^-1
# Signal (lambda): 1 fb -> 100 events
# Background (b): 2 fb -> 200 events
np.random.seed(42)

N_SAMPLE = 10.0        # Integrated Luminosity (fb^-1)
MU_TRUE = 5           # True Signal Cross-section (fb)
B_BG_NOMINAL = 1     # Nominal Background Cross-section (fb)
DELTA = 2
# Prior Beliefs
B_BOUND_FOR_PRIOR = 10 # Signal is likely < 10 fb
ALPHA_PRIOR = 1.0        # Exponential signal prior

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================


def check_prior_data_conflict_nuisance(t_obs, s_alpha, s_scale, B_BG_NOMINAL, n_sample, n_mc_samples=100000):
    """Prior Predictive Check including Background Uncertainty."""
    lam_samples = np.random.gamma(shape=s_alpha, scale=s_scale, size=n_mc_samples)
    
    mu_samples = n_sample * (lam_samples + B_BG_NOMINAL)
    t_samples = np.random.poisson(mu_samples)
    
    pvalue = (t_samples <= t_obs).mean()
    q_lower = np.percentile(t_samples, 2.5)
    q_upper = np.percentile(t_samples, 97.5)
    in_central = (q_lower <= t_obs <= q_upper)
    
    return pvalue, in_central, q_lower, q_upper, t_samples


def plot_prior_predictive(t_obs, t_samples, bins=50):
    """
    Plots the prior predictive distribution with a stepped outline and 
    fills the tail area based on the observed value.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate weights to normalize the histogram (probability density/mass)
    weights = np.ones_like(t_samples) / len(t_samples)
    
    # 3. Plot the outline (The "No filled connected histogram")
    # histtype='step' draws the outline
    counts, bin_edges, _ = ax.hist(t_samples, bins=bins, weights=weights, 
                                   histtype='step', color='black', linewidth=1.5, 
                                   label='Prior Predictive Distribution')

    # 4. Determine which tail to fill "based on the obs"
    # If obs is below median, shade left. If above, shade right.
    median_t = np.median(t_samples)
    
    if t_obs < median_t:
        # Left Tail Case
        tail_mask = t_samples <= t_obs
        tail_label = rf'Tail ($T \leq {t_obs}$)'
        tail_color = 'tab:blue'
        p_val_display = np.mean(tail_mask) # The p-value from your snippet
    else:
        # Right Tail Case
        tail_mask = t_samples >= t_obs
        tail_label = rf'Tail ($T \geq {t_obs}$)'
        tail_color = 'tab:red'
        p_val_display = np.mean(tail_mask) # The complementary p-value

    # 5. Plot the filled tail
    # We plot a second histogram of ONLY the tail samples, utilizing the same bins
    # histtype='stepfilled' fills the area
    
    # We must ensure the weights align with the original sample size to keep height correct
    tail_weights = np.ones_like(t_samples[tail_mask]) / len(t_samples)
    
    ax.hist(t_samples[tail_mask], bins=bin_edges, weights=tail_weights,
            histtype='stepfilled', color=tail_color, alpha=0.4,
            label=tail_label)

    # 6. Add Visuals (Observed Line, Intervals)
    ax.axvline(t_obs, color='k', linestyle='--', linewidth=2, 
               label=rf'Observed $t_{{obs}}={t_obs}$')

    # Optional: Plot 95% Central Region interval lines for context
    q_lower = np.percentile(t_samples, 2.5)
    q_upper = np.percentile(t_samples, 97.5)
    ax.axvline(q_lower, color='gray', linestyle=':', alpha=0.6, label='95% Central Region interval')
    ax.axvline(q_upper, color='gray', linestyle=':', alpha=0.6)
    
    # 7. Final Formatting
    ax.set_title(f'Prior Predictive Check\nTail Prob: {p_val_display:.4f}', fontsize=14)
    ax.set_xlabel('Count')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.2)
    
    return fig, ax
# ==========================================
# 4. EXECUTION FLOW
# ==========================================
if __name__ == '__main__':

    print("--- 1. Elicitation ---")
    # Signal Prior
    beta_elicited = elicit_exp_rate(0, B_BOUND_FOR_PRIOR, 0.95)
    s_scale = 1.0 / beta_elicited
    print(f"Signal Prior: Gamma(1, {beta_elicited:.2f})")

    # Grid and Iteration Values
    # lambda_grid = np.arange(0, 10, 0.1)
    epsilon = 1e-10
    low = max(epsilon, gamma.ppf(0.0001, a=ALPHA_PRIOR, scale=s_scale))
    high = gamma.ppf(0.9999, a=ALPHA_PRIOR, scale=s_scale)
    
    # 2. Create the lambda_0 grid
    lambda_grid = np.linspace(low, high, int((high-low)/.1))
    
    # n_samples_list = [25]
    # deltas_list = [1,2,3]

    # print("Generating LaTeX Tables...\n")

    # # ==========================================
    # # Generate Sweeps and LaTeX Tables
    # # ==========================================
    # bif_rows = list()
    # baga_rows = list()
    # for n_sample in n_samples_list:
    #     out_dir_baga = f"output/baga_N{n_sample}"
        
    #     # 1. Run Parallel Sweeps
    #     df_baga = run_baga_sweep(
    #         lambda_grid=lambda_grid, alpha=ALPHA_PRIOR, scale=s_scale, 
    #         b=B_BG_NOMINAL, n_sample=n_sample, output_dir=out_dir_baga
    #     )
    #                 # BagA is typically the maximum probability of false evidence against the null
    #     summary_baga = summarise_baga(df_baga, out_dir_baga, ALPHA_PRIOR, s_scale)
    #     baga_val = summary_baga['max_bias']
    #     baga_val_avg = summary_baga['average_bias']
    #     baga_val0 = summary_baga['bias_at_zero']
    #     # Coverage metrics based on the prior calculations
    #     freq_coverage = 1.0 - baga_val
    #     bayes_coverage = 1.0 - baga_val_avg
    #     baga_rows.append(f"{n_sample} & {baga_val:.3f} & {freq_coverage:.3f} & {baga_val_avg:.3f} & {bayes_coverage:.3f} & {baga_val0:.3f} &  \\\\")    
    #     print(f"\\begin{{table}}[h!]")
    # print(f"\\centering")
    # print(f"\\caption{{Bias Against and Coverage}}")
    # print(f"\\begin{{tabular}}{{lccccc}}")
    # print(f"\\hline")
    # print(f"Sample Size ($N$) & Max Baga & Freq. Coverage & Avg. Baga & Bayes Coverage & Baga at 0  \\\\")
    # print(f"\\hline")
    
    # for row in baga_rows:
    #     print(row)    
    # # End LaTeX Table formatting
    # print(f"\\hline")
    # print(f"\\end{{tabular}}")
    # print(f"\\label{{tab:bias_against}}")
    # print(f"\\end{{table}}\n")

    # for delta in deltas_list:
    #     for n_sample in n_samples_list:
    #         out_dir_bif = f"output/bif_N{n_sample}_d{delta}"
    #         epsilon = 1e-10
    #         low = max(epsilon, gamma.ppf(0.0001, a=ALPHA_PRIOR, scale=s_scale))
    #         high = gamma.ppf(0.9999, a=ALPHA_PRIOR, scale=s_scale)
            
    #         # 2. Create the lambda_0 grid
    #         lambda_grid_bif = np.linspace(low, high, int((high-low)/delta))
    #         df_bif = run_bif_sweep(
    #             lambda_grid=lambda_grid_bif, alpha=ALPHA_PRIOR, scale=s_scale, 
    #             b=B_BG_NOMINAL, n_sample=n_sample, delta=delta, output_dir=out_dir_bif
    #         )
            
    #         # 2. Extract Summaries
    #         # Assuming summarise_bif returns 'min_bias' / 'bias_at_zero' / 'average_bias' 
    #         # and summarise_baga returns 'max_bias' / 'average_bias'
    #         summary_bif = summarise_bif(df_bif, out_dir_bif, ALPHA_PRIOR, s_scale)
            
    #         # 3. Calculate Required Table Columns
    #         # BiF is the maximum probability of false evidence in favor of the null at distance delta
    #         # Since df_bif contains the column 'bias_in_favor_fix_b_max_d', we extract its max
    #         bif_val = summary_bif['average_bias']
    #         bif_val0 = summary_bif['bias_at_zero']
            
            
    #         # Print table row
    #         bif_rows.append(f"{n_sample} & {delta} & {bif_val0:.3f} & {bif_val:.3f} \\\\")
            
    # # Begin LaTeX Table formatting
    # print(f"\\begin{{table}}[h!]")
    # print(f"\\centering")
    # print(f"\\caption{{Bias In Favor}}")
    # print(f"\\begin{{tabular}}{{lccc}}")
    # print(f"\\hline")
    # print(f"Sample Size ($N$) & Delta & Bif at 0 & Avg. Bif  \\\\")
    # print(f"\\hline")
    
    # for row in bif_rows:
    #     print(row)    
    # # End LaTeX Table formatting
    # print(f"\\hline")
    # print(f"\\end{{tabular}}")
    # print(f"\\label{{tab:bias_in_favor}}")
    # print(f"\\end{{table}}\n")





    print("\n--- 2. Bias Calculation (Pre-Data) ---")
    bias_against = compute_baga(0.0, ALPHA_PRIOR, s_scale, 
                                    B_BG_NOMINAL, N_SAMPLE)
    # Alternative hypothesis: Lambda=5 (significant signal)
    bias_favor = np.max(compute_bif(DELTA, ALPHA_PRIOR, s_scale, 
                                        B_BG_NOMINAL, N_SAMPLE, DELTA))
    print(f"Bias Against H0 (lambda=0): {bias_against:.4f}")
    print(f"Bias In Favor of H0 (vs lambda=5): {bias_favor:.4f}")

    print("\n--- 3. Data Generation ---")
    # Generate Obs from mixture (True Signal + Nominal BG)
    # Note: In a real run, Nature picks one specific b. 
    # But for "expected" behavior we usually simulate from the nominal rates.
    t_obs = np.random.poisson(N_SAMPLE * (MU_TRUE + B_BG_NOMINAL))
    print(f"Observation: {t_obs}")
    #     plt.show()
    print("\n--- 4. Analysis ---")
    # Prior Predictive Check
    pval, in_central, qL, qU, t_samples = check_prior_data_conflict_nuisance(
        t_obs, ALPHA_PRIOR, s_scale, B_BG_NOMINAL, N_SAMPLE
        )
    pval = np.min([pval,1-pval])
    print(f"Prior Predictive Check: {pval}")
    # pval, in_central, qL, qU = check_prior_data_conflict_nuisance_exact(
    #     t_obs, ALPHA_PRIOR, s_scale, B_BG_NOMINAL, N_SAMPLE)[0]
    # print(f"Prior Predictive Check (Non-MC): {pval}")
    plot_prior_predictive(t_obs, t_samples)

    # Interval and RB Curve
    # lam_plot_grid = np.linspace(0, 15, 1000) # Plot range relevant to signal
    
    rb_info = rb_plausible_interval_fixed_b(
        t_obs,ALPHA_PRIOR, s_scale, B_BG_NOMINAL, N_SAMPLE
    )

    lam_L, lam_U = rb_info['interval']
    rb_vec = rb_info['rb_ratio']
    lam_plot_grid = rb_info['lambda_grid']
    rb_at_0 = np.interp(0.0, lam_plot_grid, rb_vec)
    strength = rb_info['strength']

    out_dir_baga = "output/true_data"
    df_baga = run_baga_sweep(
        lambda_grid=lambda_grid, alpha=ALPHA_PRIOR, scale=s_scale, 
        b=B_BG_NOMINAL, n_sample=N_SAMPLE, output_dir=out_dir_baga
    )
    # BagA is typically the maximum probability of false evidence against the null
    summary_baga = summarise_baga(df_baga, out_dir_baga, ALPHA_PRIOR, s_scale)
    baga_val = summary_baga['max_bias']
    baga_val_avg = summary_baga['average_bias']
    baga_val0 = summary_baga['bias_at_zero']

        # Posterior Content
    # Unnormalized Post ~ RB * Prior
    lambda_max = auto_lambda_max(ALPHA_PRIOR, s_scale, t_obs, N_SAMPLE)
    n_grid = 100000
    lambda_grid = np.linspace(0, lambda_max, n_grid)
    prior_dens = gamma.pdf(lam_plot_grid, a=ALPHA_PRIOR, scale=s_scale)
    post_unnorm = rb_vec * prior_dens
    norm_const = np.trapezoid(post_unnorm, lam_plot_grid)
    post_dens = post_unnorm / norm_const

    mask_int = (lam_plot_grid >= lam_L) & (lam_plot_grid <= lam_U)
    post_content = np.trapezoid(post_dens[mask_int], lam_plot_grid[mask_int])
    
    print(f"RB(0): {rb_at_0:.3f}")
    print(f"Interval: [{lam_L}, {lam_U}]")
    print(f"Content: {post_content:.3f}")
    rb_info2 = rb_plausible_interval_fixed_b_fast_accurate(
        t_obs,ALPHA_PRIOR, s_scale, B_BG_NOMINAL, N_SAMPLE
    )
    print(f"Interval_fa_method: {rb_info2['interval']}")
    mask_int = (lam_plot_grid >= rb_info2['interval'][0]) & (lam_plot_grid <= rb_info2['interval'][1])
    post_content = np.trapezoid(post_dens[mask_int], lam_plot_grid[mask_int])
    print(f"Content_fa_method: {post_content:.3f}")
    print(f"Strength: {strength:.3f}")
    FCCI_LL, FCCI_UL = np.array(FC_poisson(t_obs, B_BG_NOMINAL, N_SAMPLE, conf=1-baga_val))/N_SAMPLE
    print(FCCI_LL, FCCI_UL)

    # ==========================================
    # 5. PLOTTING
    # ==========================================
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
        "axes.linewidth": 1.2, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True #, "legend.frameon": False
    })

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1

    # --- Plot 1: Densities ---

    # --- Plot 1: Densities ---
    # 1. Signal Prior: Changed from Gray to Green for better visibility
    ax[0].plot(lam_plot_grid, prior_dens, linestyle='--', color='#2ca02c', lw=2,
               label=r'Signal Prior $\pi(\lambda)$')
               
    # 2. Background: Red Dash-Dot
    ax[0].axvline(B_BG_NOMINAL, color='#cc0000', linestyle=':', alpha=0.6, lw=2,
                  label=r'Background')
                  
    # 3. Posterior: Blue Solid
    ax[0].plot(lam_plot_grid, post_dens, linestyle='-', color='#0033cc', lw=2.5,
               label=r'Posterior $\pi(\lambda|\bar{{x}})$')
               
    # # 4. Fill Plausible Region
    # ax[0].fill_between(lam_plot_grid, 0, post_dens, where=mask_int, color='#0033cc', alpha=0.15,
    #                    label='Plausible Region')
                       
    # # 5. Fill FCCI Region: Changed from Gray to Orange
    # ax[0].fill_between(lam_plot_grid, 0, post_dens, where=FCCI_mask_int, color='#ff7f0e', alpha=0.15,
    #                    label='FCCI Region', hatch="//")
                       
    # 6. NEW: Add dashed vertical lines for both intervals in Plot 1
    ax[0].axvline(lam_L, color='#0033cc', linestyle='--', alpha=0.6, label='Plausible Limits')
    ax[0].axvline(lam_U, color='#0033cc', linestyle='--', alpha=0.6)
    ax[0].axvline(FCCI_LL, color='#ff7f0e', linestyle='-.', alpha=0.6, label='FCCI Limits')
    ax[0].axvline(FCCI_UL, color='#ff7f0e', linestyle='-.', alpha=0.6)

    # Decoration
    ax[0].set_xlabel(r'Parameters ($\lambda$ and $b$)') 
    ax[0].set_ylabel('Probability Density')
    
    # Text Box
    ax[0].text(0.95, 0.95,
               f'N = {N_SAMPLE:.0f}\n'
               f'$\\bar{{x}}={t_obs/N_SAMPLE}$\n'
               f'$b={B_BG_NOMINAL}$',
               transform=ax[0].transAxes, ha='right', va='top', fontsize=12,
               bbox=dict(facecolor='white', edgecolor='black', alpha=.8))
               
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(0, 8) 
    ax[0].legend(loc='best', fontsize=10, facecolor="white", framealpha=1.0, edgecolor='black')
    ax[0].set_box_aspect(1)

    # --- Plot 2: Relative Belief Ratio ---
    ax[1].plot(lam_plot_grid, rb_vec, '-', color='#cc0000', lw=2.5, label='RB Ratio')
    ax[1].axhline(1.0, color='black', linestyle=':', lw=1.5, label='Evidence Threshold')
    
    # Updated to match Plot 1's blue color for the Plausible Interval
    ax[1].axvline(lam_L, color='#0033cc', linestyle='--', alpha=0.6, label='Plausible Interval')
    ax[1].axvline(lam_U, color='#0033cc', linestyle='--', alpha=0.6)
    
    # Updated to match Plot 1's orange color for the FCCI Interval
    ax[1].axvline(FCCI_LL, color='#ff7f0e', linestyle='-.', alpha=0.6, label='FCCI Interval')
    ax[1].axvline(FCCI_UL, color='#ff7f0e', linestyle='-.', alpha=0.6)
    
    # ax[1].scatter([0], [rb_at_0], s=50, color='black', zorder=5, label=f'$RB(0)={rb_at_0:.2f}$')
    ax[1].set_xlabel(r'Signal Rate $\lambda$')
    ax[1].set_ylabel(r'RB Ratio')
    ax[1].set_xlim(0, 8)
    ax[1].legend(loc='upper left', fontsize=10, facecolor="white", framealpha=1.0, edgecolor='black')
    ax[1].set_box_aspect(1)
    
    plt.tight_layout()
    plt.show()
    # # 1. Signal Prior: Grey Dashed
    # ax[0].plot(lam_plot_grid, prior_dens, linestyle='--', color='gray', lw=2, 
    #            label=r'Signal Prior $\pi(\lambda)$')

    # # 2. Background: Red Dash-Dot (Distinct style)
    # # We plot b here to compare the uncertainty of b vs lambda
    # ax[0].axvline(B_BG_NOMINAL, color='#cc0000', linestyle='--', alpha=0.4,
    #               label=r'Background')

    # # 3. Posterior: Blue Solid
    # ax[0].plot(lam_plot_grid, post_dens, linestyle='-', color='#0033cc', lw=2.5, 
    #            label=r'Posterior $\pi(\lambda|\bar{{x}})$')

    # # 4. Fill Plausible Region
    # ax[0].fill_between(lam_plot_grid, 0, post_dens, where=mask_int, color='#0033cc', alpha=0.15, 
    #                    label='Plausible Region')
    # ax[0].fill_between(lam_plot_grid, 0, post_dens, where=FCCI_mask_int, color='gray', alpha=0.15, 
    #                    label='FCCI')

    # # Decoration
    # ax[0].set_xlabel(r'Parameters ($\lambda$ and $b$)') # Updated label since we plot both
    # ax[0].set_ylabel('Probability Density')
    # # ax[0].set_title(r'Bayesian Inference', loc='left')

    # # Text Box
    # ax[0].text(0.95, 0.95, 
    #            f'N = {N_SAMPLE:.0f}\n'
    #            f'$\\bar{{x}}={t_obs/N_SAMPLE}$\n'
    #            f'$B={B_BG_NOMINAL}$', 
    #            transform=ax[0].transAxes, ha='right', va='top', fontsize=12, 
    #            bbox=dict(facecolor='white', edgecolor='none', alpha=.8))

    # ax[0].set_ylim(bottom=0)
    # ax[0].set_xlim(0, 8) # Adjusted to see details
    # ax[0].legend(loc='best', fontsize=11)
    # ax[0].set_box_aspect(1)

    # # Plot 2
    # # Plot 2: Relative Belief Ratio
    # ax[1].plot(lam_plot_grid, rb_vec, '-', color='#cc0000', lw=2.5, label='RB Ratio')
    # ax[1].axhline(1.0, color='black', linestyle=':', lw=1.5, label='Evidence Thereshold')
    # ax[1].axvline(lam_L, color='#cc0000', linestyle='--', alpha=0.4, label='Plausible Interval')
    # ax[1].axvline(lam_U, color='#cc0000', linestyle='--', alpha=0.4)
    # ax[1].axvline(FCCI_LL, color='gray', linestyle='--', alpha=0.4, label='FCCI Interval')
    # ax[1].axvline(FCCI_UL, color='gray', linestyle='--', alpha=0.4)
    # ax[1].scatter([0], [rb_at_0], s=50, color='black', zorder=5, label=f'$RB(0)={rb_at_0:.2f}$')

    # ax[1].set_xlabel(r'Signal Rate $\lambda$')
    # ax[1].set_ylabel(r'RB Ratio')
    # # ax[1].set_title(r'Evidence Analysis', loc='right')
    # ax[1].set_xlim(0, 8)
    # ax[1].legend(loc='upper left', fontsize=11) # Added Legend
    # ax[1].set_box_aspect(1)

    # plt.tight_layout()
    # plt.show()

    # # ==========================================
    # # 6. REPORT
    # # ==========================================
    # evidence_str = "favoring" if rb_at_0 > 1 else "against"
    # conflict_str = "no significant conflict" if in_central else "potential conflict"

    # latex_paragraph = f"""
    # \\textbf{{Results Analysis:}} 
    # The experiment analyzed an experiment with sample size $N = {N_SAMPLE:.0f}$, 
    # observing $T(x) = {t_obs}$ events, $\\bar{{x}} = {t_obs/N_SAMPLE}$. The background $B = {B_BG_NOMINAL}$.

    # A signal Exponential prior $\\pi(\\lambda)$ was elicited using an interval with upper bound of $\\lambda < {B_BOUND_FOR_PRIOR}$ with $95\\%$ probability content.

    # The prior predictive check (one-sided p-value: $p={pval:.3f}$) indicates {conflict_str} between the model and data.

    # Accounting for the background nuisance parameter and assuming elicited priors are true, the bias against $H_0$ is {bias_against:.3f}, 
    # and the bias in favor of $H_0$ (practical significance $\\delta={DELTA}$) is {bias_favor:.3f} with a Baysian statistical power of {1-bias_favor:.3f}.

    # The Relative Belief ratio at the null hypothesis ($H_0: \\lambda=0$) is $RB(0) = {rb_at_0:.3f}$, providing evidence \\textbf{{{evidence_str}}} the null background-only hypothesis. 

    # The resulting plausible interval is $\\lambda \\in [{lam_L:.2f}, {lam_U:.2f}]$, 
    # containing ${post_content*100:.1f}\\%$ of the posterior mass.
    # """

    # print("\n" + "="*40 + "\nLATEX REPORT\n" + "="*40)
    # print(latex_paragraph)