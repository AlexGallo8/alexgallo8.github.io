import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import poisson, expon, chi2_contingency, kstest
from scipy.special import factorial

def simulate_poisson_process(lambda_rate, T, n_intervals):
    """
    Simulate a single realization of a Poisson process.
    
    Parameters:
    -----------
    lambda_rate : float
        Rate parameter (expected events per unit time)
    T : float
        Total time interval
    n_intervals : int
        Number of subintervals to divide [0,T]
    
    Returns:
    --------
    times : ndarray
        Time grid [0, dt, 2dt, ..., T]
    counts : ndarray
        Cumulative count N(t) at each time point
    """
    dt = T / n_intervals
    times = np.linspace(0, T, n_intervals + 1)
    
    # Generate events: Bernoulli(lambda*dt) for each interval
    events = np.random.binomial(1, lambda_rate * dt, size=n_intervals)
    
    # Cumulative sum gives the counting process
    counts = np.concatenate([[0], np.cumsum(events)])
    
    return times, counts

def simulate_multiple_realizations(lambda_rate, T, n_intervals, n_sims):
    """
    Generate multiple independent realizations of the Poisson process.
    
    Returns:
    --------
    times : ndarray
        Time grid
    all_paths : ndarray
        Array of shape (n_sims, n_intervals+1) with sample paths
    final_counts : ndarray
        Final count N(T) for each realization
    """
    times = np.linspace(0, T, n_intervals + 1)
    all_paths = np.zeros((n_sims, n_intervals + 1))
    
    for i in range(n_sims):
        _, counts = simulate_poisson_process(lambda_rate, T, n_intervals)
        all_paths[i] = counts
    
    final_counts = all_paths[:, -1]
    
    return times, all_paths, final_counts

def theoretical_poisson_pmf(lambda_t, k_max):
    """
    Calculate theoretical Poisson PMF.
    
    Parameters:
    -----------
    lambda_t : float
        Parameter λt for Poisson(λt)
    k_max : int
        Maximum k value to compute
    
    Returns:
    --------
    k_values : ndarray
        Array [0, 1, 2, ..., k_max]
    probabilities : ndarray
        P(N = k) for each k
    """
    k_values = np.arange(0, k_max + 1)
    probabilities = poisson.pmf(k_values, lambda_t)
    
    return k_values, probabilities

def plot_sample_paths(times, paths, lambda_rate, T, n_paths_to_plot=50, 
                     output_dir='assets/images'):
    """
    Plot sample paths of the Poisson process.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Select subset of paths to plot
    n_total = paths.shape[0]
    indices = np.linspace(0, n_total - 1, min(n_paths_to_plot, n_total)).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for idx, path_idx in enumerate(indices):
        ax.step(times, paths[path_idx], where='post', 
               color=colors[idx], alpha=0.6, linewidth=1.5)
    
    # Add mean trajectory
    mean_path = np.mean(paths, axis=0)
    ax.plot(times, mean_path, 'r-', linewidth=3, 
           label=f'Mean trajectory (E[N(t)] = {lambda_rate}t)')
    
    # Theoretical mean
    ax.plot(times, lambda_rate * times, 'k--', linewidth=2.5,
           label=f'Theoretical mean: λt = {lambda_rate}t')
    
    ax.set_xlabel('Time t', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count N(t)', fontsize=13, fontweight='bold')
    ax.set_title(f'Poisson Process Sample Paths (λ = {lambda_rate}, T = {T})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw10_sample_paths_lambda_{lambda_rate:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_final_count_distribution(final_counts, lambda_rate, T, 
                                  output_dir='assets/images'):
    """
    Plot distribution of N(T) and compare with theoretical Poisson.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Histogram with PMF overlay
    lambda_t = lambda_rate * T
    k_max = int(max(final_counts)) + 5
    k_theory, p_theory = theoretical_poisson_pmf(lambda_t, k_max)
    
    # Empirical histogram
    counts, bins, _ = ax1.hist(final_counts, bins=range(int(min(final_counts)), 
                                int(max(final_counts)) + 2), 
                               density=True, alpha=0.6, color='steelblue',
                               edgecolor='black', linewidth=1.5,
                               label='Empirical')
    
    # Theoretical PMF
    ax1.plot(k_theory, p_theory, 'ro-', linewidth=2.5, markersize=8,
            label=f'Theoretical Poisson({lambda_t:.1f})')
    
    # Statistics
    sample_mean = np.mean(final_counts)
    sample_var = np.var(final_counts, ddof=1)
    
    stats_text = f'Sample mean: {sample_mean:.3f}\n'
    stats_text += f'Theoretical mean: {lambda_t:.3f}\n'
    stats_text += f'Sample variance: {sample_var:.3f}\n'
    stats_text += f'Theoretical variance: {lambda_t:.3f}\n'
    stats_text += f'Variance/Mean ratio: {sample_var/sample_mean:.3f}'
    
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Count N(T)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'Distribution of Final Count (λ = {lambda_rate}, T = {T})',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: Q-Q plot
    # Sort empirical quantiles
    sorted_counts = np.sort(final_counts)
    n = len(sorted_counts)
    empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n
    
    # Theoretical quantiles from Poisson
    theoretical_quantiles = poisson.ppf(empirical_quantiles, lambda_t)
    
    ax2.scatter(theoretical_quantiles, sorted_counts, alpha=0.5, s=20)
    
    # Perfect fit line
    min_val = min(theoretical_quantiles.min(), sorted_counts.min())
    max_val = max(theoretical_quantiles.max(), sorted_counts.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect fit')
    
    ax2.set_xlabel('Theoretical Quantiles (Poisson)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Empirical Quantiles', fontsize=12, fontweight='bold')
    ax2.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw10_distribution_lambda_{lambda_rate:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    
    return sample_mean, sample_var

def plot_convergence_analysis(lambda_rate, T, n_values, n_sims=1000,
                              output_dir='assets/images'):
    """
    Show convergence as n (number of intervals) increases.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    lambda_t = lambda_rate * T
    
    for idx, n in enumerate(n_values):
        ax = axes[idx]
        
        # Simulate
        _, _, final_counts = simulate_multiple_realizations(lambda_rate, T, n, n_sims)
        
        # Empirical histogram
        ax.hist(final_counts, bins=range(int(min(final_counts)), 
                                        int(max(final_counts)) + 2),
               density=True, alpha=0.6, color='steelblue',
               edgecolor='black', linewidth=1.2, label='Empirical')
        
        # Theoretical PMF
        k_max = int(max(final_counts)) + 5
        k_theory, p_theory = theoretical_poisson_pmf(lambda_t, k_max)
        ax.plot(k_theory, p_theory, 'ro-', linewidth=2, markersize=6,
               label=f'Poisson({lambda_t:.1f})')
        
        # Statistics
        dt = T / n
        ax.text(0.97, 0.97, f'n = {n}\nΔt = {dt:.4f}\nλΔt = {lambda_rate*dt:.4f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.set_xlabel('Count N(T)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(f'Convergence with n = {n} intervals', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw10_convergence_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_rate_comparison(T, n_intervals, lambda_values, n_sims=1000,
                        output_dir='assets/images'):
    """
    Compare processes with different rate parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top row: Sample paths
    for idx, lambda_rate in enumerate(lambda_values[:2]):
        ax = axes[0, idx]
        times, paths, _ = simulate_multiple_realizations(lambda_rate, T, 
                                                         n_intervals, 20)
        
        for i in range(paths.shape[0]):
            ax.step(times, paths[i], where='post', alpha=0.5, linewidth=1.5)
        
        mean_path = np.mean(paths, axis=0)
        ax.plot(times, mean_path, 'r-', linewidth=3, label='Mean')
        ax.plot(times, lambda_rate * times, 'k--', linewidth=2, 
               label=f'E[N(t)] = {lambda_rate}t')
        
        ax.set_xlabel('Time t', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count N(t)', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample Paths (λ = {lambda_rate})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Bottom row: Distributions
    for idx, lambda_rate in enumerate(lambda_values[:2]):
        ax = axes[1, idx]
        _, _, final_counts = simulate_multiple_realizations(lambda_rate, T,
                                                           n_intervals, n_sims)
        
        lambda_t = lambda_rate * T
        k_max = int(max(final_counts)) + 5
        k_theory, p_theory = theoretical_poisson_pmf(lambda_t, k_max)
        
        ax.hist(final_counts, bins=range(int(min(final_counts)),
                                        int(max(final_counts)) + 2),
               density=True, alpha=0.6, color='steelblue',
               edgecolor='black', linewidth=1.2, label='Empirical')
        
        ax.plot(k_theory, p_theory, 'ro-', linewidth=2, markersize=6,
               label=f'Poisson({lambda_t:.1f})')
        
        sample_mean = np.mean(final_counts)
        sample_var = np.var(final_counts, ddof=1)
        
        stats_text = f'Mean: {sample_mean:.2f}\nVar: {sample_var:.2f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Count N(T)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution (λ = {lambda_rate})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw10_rate_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def analyze_interarrival_times(paths, times, lambda_rate, T,
                               output_dir='assets/images'):
    """
    Extract and analyze interarrival times.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    interarrival_times = []
    
    for path in paths:
        # Find jump times (where count increases)
        jumps = np.where(np.diff(path) > 0)[0]
        if len(jumps) > 1:
            jump_times = times[jumps]
            # Calculate time differences
            diffs = np.diff(jump_times)
            interarrival_times.extend(diffs)
    
    if len(interarrival_times) == 0:
        print("No interarrival times found (too few events)")
        return
    
    interarrival_times = np.array(interarrival_times)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Histogram with exponential overlay
    ax1.hist(interarrival_times, bins=50, density=True, alpha=0.6,
            color='steelblue', edgecolor='black', linewidth=1.5,
            label='Empirical')
    
    # Theoretical exponential
    t_range = np.linspace(0, max(interarrival_times), 1000)
    ax1.plot(t_range, expon.pdf(t_range, scale=1/lambda_rate),
            'r-', linewidth=3, label=f'Exponential(λ={lambda_rate})')
    
    sample_mean = np.mean(interarrival_times)
    theoretical_mean = 1 / lambda_rate
    
    stats_text = f'Sample mean: {sample_mean:.4f}\n'
    stats_text += f'Theoretical mean: {theoretical_mean:.4f}\n'
    stats_text += f'Sample size: {len(interarrival_times)}'
    
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Interarrival Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Interarrival Time Distribution (λ = {lambda_rate})',
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Right: Q-Q plot
    sorted_times = np.sort(interarrival_times)
    n = len(sorted_times)
    empirical_quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = expon.ppf(empirical_quantiles, scale=1/lambda_rate)
    
    ax2.scatter(theoretical_quantiles, sorted_times, alpha=0.5, s=20)
    
    min_val = min(theoretical_quantiles.min(), sorted_times.min())
    max_val = max(theoretical_quantiles.max(), sorted_times.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect fit')
    
    ax2.set_xlabel('Theoretical Quantiles (Exponential)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Empirical Quantiles', fontsize=12, fontweight='bold')
    ax2.set_title('Q-Q Plot for Interarrival Times', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw10_interarrival_times_lambda_{lambda_rate:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    
    # KS test
    ks_stat, ks_pval = kstest(interarrival_times, 'expon', args=(0, 1/lambda_rate))
    print(f"\nKolmogorov-Smirnov Test for Exponential Fit:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4f}")
    print(f"  Result: {'PASS' if ks_pval > 0.05 else 'FAIL'} (α=0.05)")

def perform_statistical_tests(final_counts, lambda_rate, T):
    """
    Perform statistical tests for Poisson fit.
    """
    lambda_t = lambda_rate * T
    
    # Chi-square goodness of fit
    unique_counts, observed_freq = np.unique(final_counts, return_counts=True)
    
    # Expected frequencies
    k_min, k_max = int(min(unique_counts)), int(max(unique_counts))
    k_range = np.arange(k_min, k_max + 1)
    expected_prob = poisson.pmf(k_range, lambda_t)
    expected_freq = expected_prob * len(final_counts)
    
    # Match observed to expected
    observed_aligned = np.zeros(len(k_range))
    for i, k in enumerate(k_range):
        if k in unique_counts:
            idx = np.where(unique_counts == k)[0][0]
            observed_aligned[i] = observed_freq[idx]
    
    # Remove bins with expected frequency < 5
    valid = expected_freq >= 5
    if np.sum(valid) > 1:
        chi2_stat = np.sum((observed_aligned[valid] - expected_freq[valid])**2 / 
                          expected_freq[valid])
        dof = np.sum(valid) - 1 - 1  # -1 for constraint, -1 for estimated parameter
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi2_stat, dof)
        
        print("\nChi-Square Goodness-of-Fit Test:")
        print(f"  χ² statistic: {chi2_stat:.4f}")
        print(f"  Degrees of freedom: {dof}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Result: {'PASS' if p_value > 0.05 else 'FAIL'} (α=0.05)")
    else:
        print("\nChi-Square test: Insufficient data for valid test")
    
    # KS test
    ks_stat, ks_pval = kstest(final_counts, 'poisson', args=(lambda_t,))
    print("\nKolmogorov-Smirnov Test:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {ks_pval:.4f}")
    print(f"  Result: {'PASS' if ks_pval > 0.05 else 'FAIL'} (α=0.05)")

def main():
    print("="*70)
    print("HOMEWORK 10: POISSON PROCESS SIMULATION")
    print("="*70)
    
    # Parameters
    T = 1.0
    n_intervals = 5000
    n_sims = 1000
    
    # Scenario 1: Low rate
    print("\n" + "="*70)
    print("SCENARIO 1: Low Rate (λ = 1)")
    print("="*70)
    lambda_rate = 1.0
    print(f"Simulating {n_sims} realizations...")
    times, paths, final_counts = simulate_multiple_realizations(
        lambda_rate, T, n_intervals, n_sims)
    
    plot_sample_paths(times, paths, lambda_rate, T, n_paths_to_plot=50)
    sample_mean, sample_var = plot_final_count_distribution(final_counts, lambda_rate, T)
    
    print(f"\nStatistics:")
    print(f"  Sample mean: {sample_mean:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Sample variance: {sample_var:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Variance/Mean ratio: {sample_var/sample_mean:.4f} (should be ≈1)")
    
    perform_statistical_tests(final_counts, lambda_rate, T)
    analyze_interarrival_times(paths[:100], times, lambda_rate, T)
    
    # Scenario 2: Medium rate
    print("\n" + "="*70)
    print("SCENARIO 2: Medium Rate (λ = 10)")
    print("="*70)
    lambda_rate = 10.0
    print(f"Simulating {n_sims} realizations...")
    times, paths, final_counts = simulate_multiple_realizations(
        lambda_rate, T, n_intervals, n_sims)
    
    plot_sample_paths(times, paths, lambda_rate, T, n_paths_to_plot=50)
    sample_mean, sample_var = plot_final_count_distribution(final_counts, lambda_rate, T)
    
    print(f"\nStatistics:")
    print(f"  Sample mean: {sample_mean:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Sample variance: {sample_var:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Variance/Mean ratio: {sample_var/sample_mean:.4f} (should be ≈1)")
    
    perform_statistical_tests(final_counts, lambda_rate, T)
    analyze_interarrival_times(paths[:100], times, lambda_rate, T)
    
    # Scenario 3: High rate
    print("\n" + "="*70)
    print("SCENARIO 3: High Rate (λ = 50)")
    print("="*70)
    lambda_rate = 50.0
    print(f"Simulating {n_sims} realizations...")
    times, paths, final_counts = simulate_multiple_realizations(
        lambda_rate, T, n_intervals, n_sims)
    
    plot_sample_paths(times, paths, lambda_rate, T, n_paths_to_plot=50)
    sample_mean, sample_var = plot_final_count_distribution(final_counts, lambda_rate, T)
    
    print(f"\nStatistics:")
    print(f"  Sample mean: {sample_mean:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Sample variance: {sample_var:.4f} (theoretical: {lambda_rate*T:.4f})")
    print(f"  Variance/Mean ratio: {sample_var/sample_mean:.4f} (should be ≈1)")
    
    perform_statistical_tests(final_counts, lambda_rate, T)
    
    # Convergence analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    print("Testing convergence with different n values...")
    n_values = [10, 100, 1000, 5000]
    plot_convergence_analysis(10.0, T, n_values, n_sims=1000)
    
    # Rate comparison
    print("\n" + "="*70)
    print("RATE PARAMETER COMPARISON")
    print("="*70)
    print("Comparing different rate parameters...")
    lambda_values = [1.0, 10.0]
    plot_rate_comparison(T, n_intervals, lambda_values, n_sims=1000)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - hw10_sample_paths_lambda_*.png")
    print("  - hw10_distribution_lambda_*.png")
    print("  - hw10_convergence_analysis.png")
    print("  - hw10_rate_comparison.png")
    print("  - hw10_interarrival_times_lambda_*.png")

if __name__ == "__main__":
    main()
