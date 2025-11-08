import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.special import comb
from collections import Counter

def simulate_random_walk(p_secure, n_weeks, n_trajectories):
    """
    Simulate random walk for server security.
    
    Parameters:
    -----------
    p_secure : float
        Probability server remains secure in a week
    n_weeks : int
        Number of weeks (steps in random walk)
    n_trajectories : int
        Number of independent trajectories to simulate
    
    Returns:
    --------
    trajectories : ndarray
        Array of shape (n_trajectories, n_weeks+1) with cumulative scores
        Column 0 is initial state (0), columns 1-n are scores after each week
    """
    # Generate random steps: +1 (secure) or -1 (breached)
    # Use binomial(1, p_secure) then transform: 0→-1, 1→+1
    steps = 2 * np.random.binomial(1, p_secure, size=(n_trajectories, n_weeks)) - 1
    
    # Calculate cumulative sums (random walk positions)
    # Add initial position 0
    trajectories = np.concatenate([
        np.zeros((n_trajectories, 1)), 
        np.cumsum(steps, axis=1)
    ], axis=1)
    
    return trajectories

def theoretical_distribution(n, p_secure):
    """
    Calculate theoretical distribution of final scores.
    
    Returns:
    --------
    scores : array
        Possible final scores
    probabilities : array
        P(S(n) = s) for each score
    counts : array
        Number of paths reaching each score (binomial coefficients)
    """
    # Possible scores: -n, -n+2, ..., n-2, n (same parity as n)
    if n % 2 == 0:
        scores = np.arange(-n, n+1, 2)
    else:
        scores = np.arange(-n, n+1, 2)
    
    # For score s, we need k = (n+s)/2 successes
    k_values = ((n + scores) / 2).astype(int)
    
    # Number of paths (binomial coefficients)
    counts = np.array([comb(n, k, exact=True) for k in k_values])
    
    # Probabilities
    q = p_secure  # P(+1)
    p = 1 - p_secure  # P(-1)
    probabilities = np.array([
        comb(n, k, exact=True) * (q**k) * (p**(n-k)) 
        for k in k_values
    ])
    
    return scores, probabilities, counts

def count_final_positions(trajectories):
    """
    Count how many trajectories end at each position.
    
    Returns:
    --------
    unique_scores : array
        Unique final scores observed
    empirical_counts : array
        Number of trajectories ending at each score
    """
    final_scores = trajectories[:, -1]
    counter = Counter(final_scores)
    unique_scores = np.array(sorted(counter.keys()))
    empirical_counts = np.array([counter[s] for s in unique_scores])
    return unique_scores, empirical_counts

def plot_random_walk_trajectories(trajectories, p_secure, n_weeks, 
                                  title_suffix="", output_dir='assets/images'):
    """
    Plot all random walk trajectories.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: All trajectories
    n_traj = trajectories.shape[0]
    weeks = np.arange(trajectories.shape[1])
    
    # Sample trajectories to plot (if too many)
    plot_indices = np.linspace(0, n_traj-1, min(n_traj, 100)).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_indices)))
    
    for idx, traj_idx in enumerate(plot_indices):
        ax1.plot(weeks, trajectories[traj_idx], 
                color=colors[idx], alpha=0.5, linewidth=1)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, 
                label='Starting position')
    ax1.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Random Walk Trajectories (n={n_weeks}, p_secure={p_secure}){title_suffix}', 
                fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Final position histogram with theory
    final_scores = trajectories[:, -1]
    scores_theory, probs_theory, counts_theory = theoretical_distribution(n_weeks, p_secure)
    
    # Empirical histogram
    ax2.hist(final_scores, bins=len(scores_theory), alpha=0.6, 
            color='steelblue', edgecolor='black', linewidth=1.5,
            label='Empirical', density=True)
    
    # Theoretical overlay
    ax2.plot(scores_theory, probs_theory, 'ro-', linewidth=2, 
            markersize=8, label='Theoretical')
    
    ax2.set_xlabel('Final Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'Distribution of Final Scores{title_suffix}', 
                fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{output_dir}/hw7_random_walk_p{p_secure:.3f}{title_suffix.replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_trajectory_counts_comparison(n_weeks, p_secure, n_trajectories,
                                    output_dir='assets/images'):
    """
    Compare empirical trajectory counts with binomial coefficients.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    trajectories = simulate_random_walk(p_secure, n_weeks, n_trajectories)
    unique_scores, empirical_counts = count_final_positions(trajectories)
    scores_theory, probs_theory, counts_theory = theoretical_distribution(n_weeks, p_secure)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(scores_theory))
    width = 0.35
    
    # Match empirical to theoretical scores
    empirical_counts_aligned = np.zeros(len(scores_theory))
    for i, s in enumerate(scores_theory):
        if s in unique_scores:
            idx = np.where(unique_scores == s)[0][0]
            empirical_counts_aligned[i] = empirical_counts[idx]
    
    ax.bar(x - width/2, empirical_counts_aligned, width, 
        label='Empirical Counts', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, counts_theory * (n_trajectories / (2**n_weeks)), width,
        label='Theoretical (scaled)', color='coral', alpha=0.7)
    
    ax.set_xlabel('Final Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Trajectories', fontsize=12, fontweight='bold')
    ax.set_title(f'Trajectory Counts: Empirical vs Theoretical (n={n_weeks}, {n_trajectories} runs)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([int(s) for s in scores_theory])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{output_dir}/hw7_trajectory_counts_n{n_weeks}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def plot_pascals_triangle(max_n=10, output_dir='assets/images'):
    """
    Visualize Pascal's triangle and its connection to trajectory counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for n in range(max_n + 1):
        y = max_n - n
        coeffs = [comb(n, k, exact=True) for k in range(n + 1)]
        x_positions = np.linspace(-n/2, n/2, n + 1)
        
        for i, (x, c) in enumerate(zip(x_positions, coeffs)):
            # Color based on magnitude
            color = plt.cm.YlOrRd(c / max(coeffs) if max(coeffs) > 0 else 0)
            ax.scatter(x, y, s=800, c=[color], edgecolors='black', 
                    linewidths=2, zorder=3)
            ax.text(x, y, str(c), ha='center', va='center', 
                    fontsize=10, fontweight='bold', zorder=4)
    
    ax.set_xlim(-max_n/2 - 0.5, max_n/2 + 0.5)
    ax.set_ylim(-0.5, max_n + 0.5)
    ax.set_xlabel('Position (Random Walk Score / 2)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Steps', fontsize=12, fontweight='bold')
    ax.set_title("Pascal's Triangle: Binomial Coefficients and Trajectory Counts", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.invert_yaxis()
    
    plt.tight_layout()
    filename = f'{output_dir}/hw7_pascals_triangle.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def main():
    print("="*70)
    print("HOMEWORK 7: RANDOM WALK AND SECURITY SIMULATION")
    print("="*70)
    
    # Scenario 1: Symmetric walk
    print("\nScenario 1: Single attacker, p=0.5 (symmetric)")
    traj1 = simulate_random_walk(p_secure=0.5, n_weeks=50, n_trajectories=1000)
    plot_random_walk_trajectories(traj1, 0.5, 50, 
                                output_dir='assets/images')
    
    # Scenario 2: Secure system
    print("\nScenario 2: Single attacker, p=0.2 (secure system)")
    traj2 = simulate_random_walk(p_secure=0.8, n_weeks=50, n_trajectories=1000)
    plot_random_walk_trajectories(traj2, 0.8, 50,
                                output_dir='assets/images')
    
    # Scenario 3: Multiple attackers
    p_individual = 0.3
    m_attackers = 5
    q = (1 - p_individual)**m_attackers
    print(f"\nScenario 3: {m_attackers} attackers, p={p_individual} each")
    print(f"  System secure probability: {q:.4f}")
    traj3 = simulate_random_walk(p_secure=q, n_weeks=50, n_trajectories=1000)
    plot_random_walk_trajectories(traj3, q, 50,
                                output_dir='assets/images')
    
    # Trajectory count validation
    print("\nValidating trajectory counts against binomial coefficients")
    for n in [10, 20, 30]:
        print(f"  Analyzing n={n} weeks...")
        plot_trajectory_counts_comparison(n, 0.5, 10000,
                                        output_dir='assets/images')
    
    # Pascal's triangle visualization
    print("\nGenerating Pascal's triangle visualization")
    plot_pascals_triangle(max_n=15)
    
    print("\n" + "="*70)
    print("SIMULATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()