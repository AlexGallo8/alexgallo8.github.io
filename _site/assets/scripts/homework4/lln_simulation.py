import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def simulate_lln(p, n, m):
    """
    Simulate the Law of Large Numbers.
    
    Parameters:
    -----------
    p : float
        True probability of success (0 < p < 1)
    n : int
        Number of trials per trajectory
    m : int
        Number of trajectories to simulate
    
    Returns:
    --------
    trajectories : ndarray
        Array of shape (m, n) containing relative frequencies
    """
    # Generate all Bernoulli trials at once (m trajectories, n trials each)
    trials = np.random.binomial(1, p, size=(m, n))
    
    # Calculate cumulative sums for each trajectory
    cumulative_successes = np.cumsum(trials, axis=1)
    
    # Calculate relative frequencies f(n) = successes / n
    trial_numbers = np.arange(1, n + 1)
    trajectories = cumulative_successes / trial_numbers
    
    return trajectories

def plot_lln_trajectories(p, n, m, output_dir='assets/images'):
    """
    Create the main LLN visualization with trajectories and histogram.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSimulating LLN with p={p}, n={n}, m={m} trajectories...")
    trajectories = simulate_lln(p, n, m)
    
    # Create figure with gridspec for main plot and histogram
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    # Main plot: trajectories
    ax_main = fig.add_subplot(gs[0])
    
    # Plot each trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, m))
    trial_numbers = np.arange(1, n + 1)
    
    for i in range(m):
        ax_main.plot(trial_numbers, trajectories[i], 
                    color=colors[i], alpha=0.6, linewidth=1.5)
    
    # Plot the true probability p as horizontal line
    ax_main.axhline(y=p, color='red', linestyle='--', 
                   linewidth=3, label=f'True probability p = {p}')
    
    ax_main.set_xlabel('Number of trials (n)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Relative frequency f(n)', fontsize=12, fontweight='bold')
    ax_main.set_title(f'Law of Large Numbers: {m} Trajectories with p = {p}', 
                     fontsize=14, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=11)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim([0, 1])
    
    # Histogram on the right (rotated)
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    
    # Get final frequencies (at n trials)
    final_frequencies = trajectories[:, -1]
    
    # Create histogram bins
    bins = np.linspace(0, 1, 30)
    counts, bin_edges = np.histogram(final_frequencies, bins=bins)
    
    # Plot horizontal bars (rotated histogram)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    for i in range(len(counts)):
        if counts[i] > 0:
            ax_hist.barh(bin_centers[i], counts[i], 
                        height=bin_widths[i], 
                        color='steelblue', alpha=0.7, 
                        edgecolor='black', linewidth=0.5)
    
    # Mark true probability on histogram
    ax_hist.axhline(y=p, color='red', linestyle='--', linewidth=3)
    
    ax_hist.set_xlabel('Count', fontsize=11, fontweight='bold')
    ax_hist.set_title(f'Distribution\nat n={n}', fontsize=11, fontweight='bold')
    ax_hist.yaxis.tick_right()
    ax_hist.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    filename = f'{output_dir}/lln_trajectories_p{p}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    
    # Calculate and return statistics
    mean_freq = np.mean(final_frequencies)
    std_freq = np.std(final_frequencies)
    min_freq = np.min(final_frequencies)
    max_freq = np.max(final_frequencies)
    
    stats = {
        'p': p,
        'n': n,
        'm': m,
        'mean': mean_freq,
        'std': std_freq,
        'min': min_freq,
        'max': max_freq,
        'bias': mean_freq - p
    }
    
    return trajectories, stats

def plot_lln_with_side_histogram(p, n, m, output_dir='assets/images'):
    """
    Create LLN visualization with vertical histogram on the right side.
    Similar to the professor's example with histogram showing distribution at n.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSimulating LLN with side histogram: p={p}, n={n}, m={m} trajectories...")
    trajectories = simulate_lln(p, n, m)
    
    # Create figure with gridspec for main plot and histogram
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1], wspace=0.05)
    
    # Main plot: trajectories
    ax_main = fig.add_subplot(gs[0])
    
    # Plot each trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, m))
    trial_numbers = np.arange(1, n + 1)
    
    for i in range(m):
        ax_main.plot(trial_numbers, trajectories[i], 
                    color=colors[i], alpha=0.5, linewidth=1.2)
    
    # Plot the true probability p as horizontal line
    ax_main.axhline(y=p, color='red', linestyle='--', 
                   linewidth=3, label=f'True probability p = {p}')
    
    ax_main.set_xlabel('Number of trials (n)', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Relative frequency f(n)', fontsize=13, fontweight='bold')
    ax_main.set_title(f'Law of Large Numbers: {m} Trajectories with p = {p}', 
                     fontsize=15, fontweight='bold')
    ax_main.legend(loc='upper right', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim([0, 1])
    ax_main.set_xlim([0, n])
    
    # Histogram on the right (vertical, aligned with y-axis)
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    
    # Get final frequencies (at n trials)
    final_frequencies = trajectories[:, -1]
    
    # Create histogram bins aligned with y-axis
    bins = np.linspace(0, 1, 60)
    counts, bin_edges = np.histogram(final_frequencies, bins=bins)
    
    # Plot horizontal bars (histogram rotated 90 degrees)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_heights = bin_edges[1:] - bin_edges[:-1]
    
    for i in range(len(counts)):
        if counts[i] > 0:
            ax_hist.barh(bin_centers[i], counts[i], 
                        height=bin_heights[i], 
                        color='steelblue', alpha=0.7, 
                        edgecolor='navy', linewidth=0.8)
    
    # Mark true probability on histogram
    ax_hist.axhline(y=p, color='red', linestyle='--', linewidth=3)
    
    # Histogram styling
    ax_hist.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax_hist.set_title(f'Distribution\nat n={n}', fontsize=12, fontweight='bold', pad=10)
    ax_hist.yaxis.set_label_position("right")
    ax_hist.yaxis.tick_right()
    ax_hist.grid(True, alpha=0.3, axis='x')
    ax_hist.set_ylim([0, 1])
    
    # Remove y-axis labels on histogram (shared with main plot)
    ax_hist.set_yticklabels([])
    
    plt.tight_layout()
    filename = f'{output_dir}/lln_with_histogram_p{p}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    
    return trajectories

def plot_convergence_analysis(p_values, n_values, m=100, output_dir='assets/images'):
    """
    Analyze how convergence depends on n for different p values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for ax_idx, n in enumerate(n_values):
        ax = axes[ax_idx]
        
        for p_idx, p in enumerate(p_values):
            print(f"Simulating p={p}, n={n}...")
            trajectories = simulate_lln(p, n, m)
            final_freqs = trajectories[:, -1]
            
            # Calculate standard deviation
            std = np.std(final_freqs)
            theoretical_std = np.sqrt(p * (1 - p) / n)
            
            # Plot histogram
            ax.hist(final_freqs, bins=20, alpha=0.5, 
                   label=f'p={p} (σ={std:.4f})', 
                   color=colors[p_idx], edgecolor='black')
            ax.axvline(p, color=colors[p_idx], linestyle='--', linewidth=2)
        
        ax.set_xlabel('Final Relative Frequency', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution at n = {n} trials', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/lln_convergence_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_variance_vs_n(p_values, n_range, m=100, output_dir='assets/images'):
    """
    Show how variance decreases as n increases (1/n relationship).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for p_idx, p in enumerate(p_values):
        variances_empirical = []
        variances_theoretical = []
        
        for n in n_range:
            trajectories = simulate_lln(p, n, m)
            final_freqs = trajectories[:, -1]
            
            var_emp = np.var(final_freqs)
            var_theory = p * (1 - p) / n
            
            variances_empirical.append(var_emp)
            variances_theoretical.append(var_theory)
        
        # Plot 1: Variance vs n
        ax1.plot(n_range, variances_empirical, 'o-', 
                label=f'p={p} (empirical)', color=colors[p_idx], 
                markersize=6, linewidth=2)
        ax1.plot(n_range, variances_theoretical, '--', 
                label=f'p={p} (theoretical)', color=colors[p_idx], 
                linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Number of trials (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance of f(n)', fontsize=12, fontweight='bold')
    ax1.set_title('Variance Decreases as n Increases', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: n × Variance (should be roughly constant)
    for p_idx, p in enumerate(p_values):
        scaled_vars = []
        
        for n in n_range:
            trajectories = simulate_lln(p, n, m)
            final_freqs = trajectories[:, -1]
            var_emp = np.var(final_freqs)
            scaled_vars.append(n * var_emp)
        
        theoretical_value = p * (1 - p)
        
        ax2.plot(n_range, scaled_vars, 'o-', 
                label=f'p={p} (n·Var)', color=colors[p_idx], 
                markersize=6, linewidth=2)
        ax2.axhline(theoretical_value, color=colors[p_idx], 
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'p={p} (theory: {theoretical_value:.3f})')
    
    ax2.set_xlabel('Number of trials (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('n × Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Scaled Variance: n·Var(f(n)) ≈ p(1-p)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    filename = f'{output_dir}/lln_variance_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_probability_comparison(output_dir='assets/images'):
    """
    Compare two different probabilities side by side.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    p_values = [0.5, 0.3]
    n = 500
    m = 30
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, p in enumerate(p_values):
        ax = axes[idx]
        
        print(f"\nSimulating for p={p}...")
        trajectories = simulate_lln(p, n, m)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, m))
        trial_numbers = np.arange(1, n + 1)
        
        for i in range(m):
            ax.plot(trial_numbers, trajectories[i], 
                   color=colors[i], alpha=0.6, linewidth=1.5)
        
        ax.axhline(y=p, color='red', linestyle='--', 
                  linewidth=3, label=f'True p = {p}')
        
        # Add confidence bands
        final_freqs = trajectories[:, -1]
        mean_final = np.mean(final_freqs)
        std_final = np.std(final_freqs)
        
        ax.axhline(y=mean_final, color='blue', linestyle=':', 
                  linewidth=2, alpha=0.7, label=f'Mean = {mean_final:.4f}')
        ax.fill_between([1, n], 
                       mean_final - 2*std_final, 
                       mean_final + 2*std_final,
                       alpha=0.2, color='blue', 
                       label=f'±2σ = ±{2*std_final:.4f}')
        
        ax.set_xlabel('Number of trials (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative frequency f(n)', fontsize=12, fontweight='bold')
        ax.set_title(f'LLN with p = {p}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = f'{output_dir}/lln_comparison_p05_p03.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def generate_statistics_table(stats_list, output_dir='assets/data'):
    """
    Generate a CSV with statistics from simulations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    import pandas as pd
    
    df = pd.DataFrame(stats_list)
    filename = f'{output_dir}/lln_statistics.csv'
    df.to_csv(filename, index=False)
    print(f"\nStatistics saved to: {filename}")
    
    return df

def main():
    print("="*70)
    print("LAW OF LARGE NUMBERS - SIMULATION AND VISUALIZATION")
    print("="*70)
    
    # Create output directories
    os.makedirs('assets/images', exist_ok=True)
    os.makedirs('assets/data', exist_ok=True)
    
    # Example 1: Fair coin (p = 0.5)
    print("\n" + "="*70)
    print("EXAMPLE 1: Fair Coin (p = 0.5)")
    print("="*70)
    trajectories1, stats1 = plot_lln_trajectories(p=0.5, n=1000, m=50)
    
    print(f"\nStatistics for p=0.5:")
    print(f"  Mean frequency: {stats1['mean']:.6f}")
    print(f"  Std deviation:  {stats1['std']:.6f}")
    print(f"  Bias (mean-p):  {stats1['bias']:.6f}")
    print(f"  Range: [{stats1['min']:.4f}, {stats1['max']:.4f}]")
    
    # Example 2: Biased coin (p = 0.3)
    print("\n" + "="*70)
    print("EXAMPLE 2: Biased Coin (p = 0.3)")
    print("="*70)
    trajectories2, stats2 = plot_lln_trajectories(p=0.3, n=1000, m=50)
    
    print(f"\nStatistics for p=0.3:")
    print(f"  Mean frequency: {stats2['mean']:.6f}")
    print(f"  Std deviation:  {stats2['std']:.6f}")
    print(f"  Bias (mean-p):  {stats2['bias']:.6f}")
    print(f"  Range: [{stats2['min']:.4f}, {stats2['max']:.4f}]")

    # Example with side histogram
    print("\n" + "="*70)
    print("VISUALIZATION WITH SIDE HISTOGRAM (Professor's Style)")
    print("="*70)
    plot_lln_with_side_histogram(p=0.5, n=1000, m=50)
    plot_lln_with_side_histogram(p=0.3, n=1000, m=50)
    
    # Side-by-side comparison
    print("\n" + "="*70)
    print("COMPARISON: p=0.5 vs p=0.3")
    print("="*70)
    plot_probability_comparison()
    
    # Convergence analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS: Effect of n on distribution")
    print("="*70)
    p_values = [0.5, 0.3, 0.7]
    n_values = [50, 200, 500, 1000]
    plot_convergence_analysis(p_values, n_values, m=100)
    
    # Variance analysis
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS: Var(f(n)) ∝ 1/n")
    print("="*70)
    n_range = [10, 20, 50, 100, 200, 500, 1000, 2000]
    plot_variance_vs_n(p_values, n_range, m=100)
    
    # Generate statistics table
    print("\n" + "="*70)
    print("GENERATING STATISTICS TABLE")
    print("="*70)
    
    all_stats = []
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for n in [100, 500, 1000]:
            trajectories = simulate_lln(p, n, m=100)
            final_freqs = trajectories[:, -1]
            
            stats = {
                'probability_p': p,
                'trials_n': n,
                'trajectories_m': 100,
                'mean_frequency': np.mean(final_freqs),
                'std_deviation': np.std(final_freqs),
                'theoretical_std': np.sqrt(p * (1-p) / n),
                'min_frequency': np.min(final_freqs),
                'max_frequency': np.max(final_freqs),
                'bias': np.mean(final_freqs) - p
            }
            all_stats.append(stats)
    
    df = generate_statistics_table(all_stats)
    
    print("\n" + "="*70)
    print("SUMMARY TABLE (Sample)")
    print("="*70)
    print(df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  Images: assets/images/")
    print("    - lln_trajectories_p0.5.png")
    print("    - lln_trajectories_p0.3.png")
    print("    - lln_comparison_p05_p03.png")
    print("    - lln_convergence_analysis.png")
    print("    - lln_variance_analysis.png")
    print("    - lln_with_histogram_p0.5.png")
    print("    - lln_with_histogram_p0.3.png")
    print("  Data: assets/data/")
    print("    - lln_statistics.csv")
    print("\nKey Insights:")
    print("  1. As n increases, f(n) converges to p (LLN)")
    print("  2. Variance decreases as 1/n")
    print("  3. Distribution becomes more concentrated around p")
    print("  4. Works for any probability p ∈ (0,1)")

if __name__ == "__main__":
    main()