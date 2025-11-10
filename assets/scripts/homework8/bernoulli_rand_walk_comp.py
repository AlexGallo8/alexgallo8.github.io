import numpy as np
import matplotlib.pyplot as plt
import os

n = 100
p = 0.6
trajectories = np.random.binomial(1, p, size=(50, n))

# Homework 4: Cumulative successes
K = np.cumsum(trajectories, axis=1)
f_n = K / np.arange(1, n+1)

# Homework 7: Random walk (transform 0→-1, 1→+1)
steps = 2 * trajectories - 1
S = np.cumsum(steps, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot HW4
for i in range(50):
    ax1.plot(range(1, n+1), f_n[i], alpha=0.5)
ax1.axhline(p, color='red', linewidth=2, label=f'p={p}')
ax1.set_title('Homework 4: Relative Frequency')
ax1.set_xlabel('Trial n')
ax1.set_ylabel('f(n) = K/n')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot HW7
for i in range(50):
    ax2.plot(range(1, n+1), S[i], alpha=0.5)
ax2.axhline(0, color='red', linewidth=2, label='Starting position')
ax2.axhline(n*(2*p-1), color='orange', linewidth=2, 
           linestyle='--', label=f'Expected: {n*(2*p-1):.0f}')
ax2.set_title('Homework 7: Random Walk Score')
ax2.set_xlabel('Step n')
ax2.set_ylabel('S(n)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Create output directory and save
output_dir = 'assets/images'
os.makedirs(output_dir, exist_ok=True)
filename = f'{output_dir}/hw8_comparison_bernoulli_vs_random_walk.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()