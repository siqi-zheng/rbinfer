import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from .bias import compute_bif, compute_baga, _worker_bif, _worker_baga

def summarise_bif(df,output_dir, alpha, scale):
    """
    Calculates specific metrics from a DataFrame containing 'lambda_0' and 
    'bias_in_favor_fix_b' columns. It now calculates the area under the curve
    using the Trapezoidal rule for a more accurate integrated measure.

    Args:
        df: A pandas DataFrame with at least the columns 'lambda_0' and 
            'bias_in_favor_fix_b'.

    Returns:
        A dictionary containing:
        - 'bias_at_lambda_zero': The 'bias_in_favor_fix_b' value when 'lambda_0' is 0.
        - 'min_bias': The minimum value of 'bias_in_favor_fix_b'.
        - 'area_under_curve': The area under the bias curve with respect to lambda_0 
                            calculated using the Trapezoidal rule (AUC).
    """
    
    # --- 1. Bias in_favor at lambda_0 = 0 ---
    # We filter the DataFrame for rows where lambda_0 is exactly 0.
    # .iloc[0] takes the first matching row. We then select the value 
    # from the bias_in_favor_fix_b_max_d column.
    try:
        bias_at_zero = df['bias_in_favor_fix_b_max_d'].iloc[0]
    except IndexError:
        # Handle the case where no row has lambda_0 = 0
        bias_at_zero = None
        print("Warning: No data point found where lambda_0 is exactly 0.")

    # --- 2. Minimum bias_in_favor_fix_b_max_d ---
    min_bias = df['bias_in_favor_fix_b_max_d'].min()

    # --- 3. Area Under the Curve (Trapezoidal Rule) ---
    # np.trapezoid(y, x) calculates the integral of y with respect to x using 
    # the composite trapezoidal rule.
    prior_density_gamma = gamma.pdf(df['lambda_0'], a=alpha, scale=scale)
    weighted_bias = df['bias_in_favor_fix_b_max_d'] * prior_density_gamma
    
    # --- 2. Average Bias in_favor (Integration/Area under the weighted curve) ---
    # The integral is the expected value.
    # We integrate the weighted_bias function with respect to λ.
    average_bias_rb = np.trapezoid (
        weighted_bias, 
        df['lambda_0']
    )
    total_mass = np.trapezoid(prior_density_gamma, df['lambda_0'])
    average_bias_rb = average_bias_rb/ total_mass
    # --- Return results as a dictionary ---
    results = {
        'bias_at_zero': bias_at_zero,
        'min_bias': min_bias,
        'average_bias': average_bias_rb  
    }
    out_path = os.path.join(output_dir, "bias_in_favor_summary.csv")
    # The corrected line: Pass the dictionary and explicitly set the index to the keys
    df2 = pd.DataFrame(results.items(), columns=['Metric', 'Value'])
    df2.to_csv(out_path,index=False)
    print(f"Saved to {out_path}")

    return results

def run_bif_sweep(
    lambda_grid, alpha, scale, b, n_sample, delta, 
    output_dir="test_gemini_codes/output/bias_in_favor", t_max=None
):
    """
    Executes the bias in favor computation across a grid of lambda_0 values 
    using a multi-process pool, and saves the results to a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    if lambda_grid is None:
        epsilon = 1e-10
        low = max(epsilon, gamma.ppf(0.0001, a=alpha, scale=scale))
        high = gamma.ppf(0.9999, a=alpha, scale=scale)
        
        # 2. Create the lambda_0 grid
        lambda_grid = np.linspace(low, high, int((high-low)/delta))
    args_list = [
        (lambda_0, alpha, scale, b, n_sample, delta, t_max)
        for lambda_0 in lambda_grid
    ]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(_worker_bif, args_list),
            total=len(args_list),
            desc="Lambda Grid (parallel)"
        ))
    
    df = pd.DataFrame(
        results,
        columns=[
            "lambda_0",
            "bias_in_favor_fix_b_plus_d",
            "bias_in_favor_fix_b_minus_d",
            "bias_in_favor_fix_b_max_d"
        ]
    )
    
    out_path = os.path.join(output_dir, "bias_in_favor.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    summarise_bif(df,output_dir, alpha, scale)
    return df

def summarise_baga(df,output_dir, alpha, scale):
        """
        Calculates specific metrics from a DataFrame containing 'lambda_0' and 
        'bias_against' columns. It now calculates the area under the curve
        using the Trapezoidal rule for a more accurate integrated measure.

        Args:
            df: A pandas DataFrame with at least the columns 'lambda_0' and 
                'bias_against'.

        Returns:
            A dictionary containing:
            - 'bias_at_lambda_zero': The 'bias_against' value when 'lambda_0' is 0.
            - 'max_bias': The maximum value of 'bias_against'.
            - 'area_under_curve': The area under the bias curve with respect to lambda_0 
                                calculated using the Trapezoidal rule (AUC).
        """
        
        # --- 1. bias_against at lambda_0 = 0 ---
        # We filter the DataFrame for rows where lambda_0 is exactly 0.
        # .iloc[0] takes the first matching row. We then select the value 
        # from the bias_against column.
        try:
            bias_at_zero = df['bias_against'].iloc[0]
        except IndexError:
            # Handle the case where no row has lambda_0 = 0
            bias_at_zero = None
            print("Warning: No data point found where lambda_0 is exactly 0.")

        # --- 2. Max bias_against ---
        max_bias = df['bias_against'].max()

        # --- 3. Area Under the Curve (Trapezoidal Rule) ---
        # np.trapezoid(y, x) calculates the integral of y with respect to x using 
        # the composite trapezoidal rule.
        prior_density_gamma = gamma.pdf(df['lambda_0'], a=alpha, scale=scale)
        weighted_bias = df['bias_against'] * prior_density_gamma
        
        # --- 2. Average bias_against (Integration/Area under the weighted curve) ---
        # The integral is the expected value.
        # We integrate the weighted_bias function with respect to λ.
        
        average_bias_rb = np.trapezoid (
            weighted_bias, 
            df['lambda_0']
        )
        total_mass = np.trapezoid(prior_density_gamma, df['lambda_0'])
        average_bias_rb = average_bias_rb/ total_mass
        
        # --- Return results as a dictionary ---
        results = {
            'bias_at_zero': bias_at_zero,
            'max_bias': max_bias,
            'average_bias': average_bias_rb  
        }
        out_path = os.path.join(output_dir, "bias_against_summary.csv")
        # The corrected line: Pass the dictionary and explicitly set the index to the keys
        df2 = pd.DataFrame(results.items(), columns=['Metric', 'Value'])
        df2.to_csv(out_path,index=False)
        print(f"Saved to {out_path}")

        return results

def run_baga_sweep(
    lambda_grid, alpha, scale, b, n_sample, 
    output_dir="test_gemini_codes/output/bias_against", t_max=None
):
    """
    Executes the bias against computation across a grid of lambda_0 values 
    using a multi-process pool, and saves the results to a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    if lambda_grid is None:
        epsilon = 1e-10
        low = max(epsilon, gamma.ppf(0.0001, a=alpha, scale=scale))
        high = gamma.ppf(0.9999, a=alpha, scale=scale)
        
        # 2. Create the lambda_0 grid
        lambda_grid = np.linspace(low, high, int((high-low)/0.1))
    args_list = [
        (lambda_0, alpha, scale, b, n_sample, t_max)
        for lambda_0 in lambda_grid
    ]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(_worker_baga, args_list),
            total=len(args_list),
            desc="Lambda Grid (parallel)"
        ))
    
    df = pd.DataFrame(
        results,
        columns=[
            "lambda_0",
            "bias_against"
        ]
    )
    
    out_path = os.path.join(output_dir, "bias_against.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    summarise_baga(df,output_dir, alpha, scale)
    return df

def sweep_and_plot_bif(
    n_sample_list,
    lambda_grid,
    alpha,
    scale,
    b,
    delta,
    base_output_dir="test_gemini_codes/output/bias_in_favor",
    t_max=None,
    title="Bias In Favor vs. $\\lambda_0$ — Sample Size Comparison",
    save_plot=True,
):
    """
    Loop over multiple sample sizes, run the bias-in-favor experiment for each,
    then overlay all curves on a single plot.

    Parameters
    ----------
    n_sample_list    : list[int]   – sample sizes to sweep
    lambda_grid      : array-like  – λ0 values (shared across all runs)
    alpha, scale     : float       – Gamma prior parameters
    b                : float       – background rate (fixed)
    delta            : float       – perturbation magnitude
    base_output_dir  : str         – parent folder; one sub-folder created per n
    t_max            : int | None  – passed straight to the MC function
    n_mc_samples     : int         – MC draws per λ0 evaluation
    title            : str         – plot title
    save_plot        : bool        – if True, saves PNG next to base_output_dir

    Returns
    -------
    dict[int -> pd.DataFrame]  – keyed by n_sample, value is the result DataFrame
    """
    
    all_results: dict[int, pd.DataFrame] = {}
    b_str = f"{b:.4f}" if isinstance(b, (int, float, np.floating, np.integer)) else f"Gamma(α={b[0]:.2f}, scale={b[1]:.4f})"

    # ── 1. Run experiment for every n ────────────────────────────────────────
    for n_sample in n_sample_list:
        output_dir = os.path.join(
            base_output_dir,
            f"b_{b_str}_gamma(1,{np.round(scale, 2)})_delta{delta}_n{n_sample}",
        )
        print(f"\n{'='*60}")
        print(f"Running n_sample = {n_sample}")
        print(f"{'='*60}")

        run_bif_sweep(
    lambda_grid, alpha, scale, b, n_sample, delta, 
    output_dir=output_dir, t_max=t_max
)

        csv_path = os.path.join(output_dir, "bias_in_favor.csv")
        df = pd.read_csv(csv_path).sort_values("lambda_0")
        all_results[n_sample] = df

    # ── 2. Overlay plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    for n_sample, df in all_results.items():
        ax.plot(
            df["lambda_0"],
            df["bias_in_favor_fix_b_max_d"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"n = {n_sample}",
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Bias = 1")
    ax.set_xlabel(r"$\lambda_0$", fontsize=13)
    ax.set_ylabel("Bias In Favor (max over ±δ)", fontsize=12)

    ax.set_title(
        f"{title}\n"
        f"b={b_str}, δ={delta}, "
        fr"Gamma prior on $\lambda$: α={alpha:.2f}, scale={scale:.4f}",
        fontsize=11,
    )
    ax.legend(title="Sample size", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_plot:
        os.makedirs(base_output_dir, exist_ok=True)
        n_tag = "_".join(str(n) for n in n_sample_list)
        plot_path = os.path.join(
            base_output_dir, f"bias_vs_lambda_sample_sizes_{n_tag}.png"
        )
        fig.savefig(plot_path, dpi=150)
        print(f"\nOverlay plot saved to: {plot_path}")

    plt.show()
    return all_results

def sweep_and_plot_baga(
    n_sample_list,
    lambda_grid,
    alpha,
    scale,
    b,
    base_output_dir="test_gemini_codes/output/bias_against",
    t_max=None,
    title="Bias Against vs. $\\lambda_0$ — Sample Size Comparison",
    save_plot=True,
):
    """
    Loop over multiple sample sizes, run the bias-in-favor experiment for each,
    then overlay all curves on a single plot.

    Parameters
    ----------
    n_sample_list    : list[int]   – sample sizes to sweep
    lambda_grid      : array-like  – λ0 values (shared across all runs)
    alpha, scale     : float       – Gamma prior parameters
    b                : float       – background rate (fixed)
    base_output_dir  : str         – parent folder; one sub-folder created per n
    t_max            : int | None  – passed straight to the MC function
    n_mc_samples     : int         – MC draws per λ0 evaluation
    title            : str         – plot title
    save_plot        : bool        – if True, saves PNG next to base_output_dir

    Returns
    -------
    dict[int -> pd.DataFrame]  – keyed by n_sample, value is the result DataFrame
    """

    all_results: dict[int, pd.DataFrame] = {}
    b_str = f"{b:.4f}" if isinstance(b, (int, float, np.floating, np.integer)) else f"Gamma(α={b[0]:.2f}, scale={b[1]:.4f})"

    # ── 1. Run experiment for every n ────────────────────────────────────────
    for n_sample in n_sample_list:
        output_dir = os.path.join(
            base_output_dir,
            f"b_{b_str}_gamma(1,{np.round(scale, 2)})_n{n_sample}",
        )
        print(f"\n{'='*60}")
        print(f"Running n_sample = {n_sample}")
        print(f"{'='*60}")

        run_baga_sweep(
    lambda_grid, alpha, scale, b, n_sample, 
    output_dir=output_dir, t_max=t_max
)

        csv_path = os.path.join(output_dir, "bias_against.csv")
        df = pd.read_csv(csv_path).sort_values("lambda_0")
        all_results[n_sample] = df

    # ── 2. Overlay plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))

    for n_sample, df in all_results.items():
        ax.plot(
            df["lambda_0"],
            df["bias_against"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"n = {n_sample}",
        )

    ax.set_xlabel(r"$\lambda_0$", fontsize=13)
    ax.set_ylabel("Bias Against", fontsize=12)

    ax.set_title(
        f"{title}\n"
        f"b={b_str}, "
        fr"Gamma prior on $\lambda$: α={alpha:.2f}, scale={scale:.4f}",
        fontsize=11,
    )

    ax.legend(title="Sample size", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    if save_plot:
        os.makedirs(base_output_dir, exist_ok=True)
        plot_path = os.path.join(
            base_output_dir, f"bias_against_b_{b_str}_gamma(1,{np.round(scale, 2)})_n{n_sample}.png"
        )
        fig.savefig(plot_path, dpi=150)
        print(f"\nOverlay plot saved to: {plot_path}")

    plt.show()
    return all_results
