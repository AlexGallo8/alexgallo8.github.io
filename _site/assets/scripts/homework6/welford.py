import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class OnlineStatistics:
    """
    Online algorithm for computing mean and variance using Welford's method.
    
    This implementation uses O(1) memory and provides numerically stable
    updates for streaming data.
    """
    
    def __init__(self):
        """Initialize the online statistics tracker."""
        self.n = 0          # Number of observations
        self.mean = 0.0     # Running mean
        self.M2 = 0.0       # Sum of squared deviations (S in derivation)
        
    def update(self, x: float) -> None:
        """
        Update statistics with a new observation.
        
        Parameters:
        -----------
        x : float
            New observation value
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def get_mean(self) -> float:
        """Return the current mean."""
        return self.mean
    
    def get_variance(self, ddof: int = 1) -> float:
        """
        Return the current variance.
        
        Parameters:
        -----------
        ddof : int, default=1
            Degrees of freedom (0 for population, 1 for sample variance)
        
        Returns:
        --------
        float
            Variance estimate
        """
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - ddof)
    
    def get_std(self, ddof: int = 1) -> float:
        """Return the current standard deviation."""
        return np.sqrt(self.get_variance(ddof))
    
    def get_count(self) -> int:
        """Return the number of observations processed."""
        return self.n
    
    def __repr__(self) -> str:
        """String representation of current statistics."""
        return (f"OnlineStatistics(n={self.n}, mean={self.mean:.6f}, "
                f"variance={self.get_variance():.6f}, std={self.get_std():.6f})")


class BatchStatistics:
    """
    Traditional batch algorithm for comparison (LESS STABLE).
    
    WARNING: This naive implementation stores all data and uses
    the numerically unstable formula: Var(X) = E[X²] - (E[X])²
    """
    
    def __init__(self):
        """Initialize the batch statistics tracker."""
        self.data = []
        
    def update(self, x: float) -> None:
        """Store new observation."""
        self.data.append(x)
        
    def get_mean(self) -> float:
        """Compute mean from all stored data."""
        if not self.data:
            return 0.0
        return sum(self.data) / len(self.data)
    
    def get_variance(self, ddof: int = 1) -> float:
        """Compute variance using naive (unstable) formula."""
        if len(self.data) < 2:
            return 0.0
        
        mean = self.get_mean()
        n = len(self.data)
        
        # Unstable formula: Var = E[X²] - (E[X])²
        sum_squares = sum(x**2 for x in self.data)
        variance = (sum_squares / n) - (mean ** 2)
        
        # Apply Bessel's correction if needed
        if ddof == 1:
            variance *= n / (n - 1)
        
        return variance
    
    def get_std(self, ddof: int = 1) -> float:
        """Return standard deviation."""
        return np.sqrt(max(0, self.get_variance(ddof)))
    
    def get_count(self) -> int:
        """Return number of observations."""
        return len(self.data)


def compare_algorithms(data: List[float]) -> Tuple[OnlineStatistics, BatchStatistics]:
    """
    Compare online and batch algorithms on the same data.
    
    Parameters:
    -----------
    data : list of float
        Input data sequence
    
    Returns:
    --------
    tuple
        (online_stats, batch_stats) objects
    """
    online = OnlineStatistics()
    batch = BatchStatistics()
    
    for x in data:
        online.update(x)
        batch.update(x)
    
    return online, batch


def test_numerical_stability():
    """
    Test numerical stability with data that has large mean and small variance.
    
    This is where naive algorithms often fail due to catastrophic cancellation.
    """
    print("="*70)
    print("TEST 1: NUMERICAL STABILITY")
    print("="*70)
    print("\nData: [1e9, 1e9 + 1, 1e9 + 2, ..., 1e9 + 99]")
    print("(Large mean ~1e9, small variance ~833.33)")
    print()
    
    # Generate data with large mean, small variance
    base = 1e9
    data = [base + i for i in range(100)]
    
    # True statistics
    true_mean = np.mean(data)
    true_var = np.var(data, ddof=1)
    true_std = np.std(data, ddof=1)
    
    # Online algorithm (Welford)
    online = OnlineStatistics()
    for x in data:
        online.update(x)
    
    # Batch algorithm (naive)
    batch = BatchStatistics()
    for x in data:
        batch.update(x)
    
    # Compare results
    print("TRUE VALUES (numpy with higher precision):")
    print(f"  Mean:     {true_mean:.10f}")
    print(f"  Variance: {true_var:.10f}")
    print(f"  Std Dev:  {true_std:.10f}")
    print()
    
    print("ONLINE ALGORITHM (Welford):")
    print(f"  Mean:     {online.get_mean():.10f}")
    print(f"  Variance: {online.get_variance():.10f}")
    print(f"  Std Dev:  {online.get_std():.10f}")
    print()
    
    print("BATCH ALGORITHM (Naive E[X²]-(E[X])²):")
    print(f"  Mean:     {batch.get_mean():.10f}")
    print(f"  Variance: {batch.get_variance():.10f}")
    print(f"  Std Dev:  {batch.get_std():.10f}")
    print()
    
    print("ERRORS:")
    print(f"  Online mean error:  {abs(online.get_mean() - true_mean):.2e}")
    print(f"  Batch mean error:   {abs(batch.get_mean() - true_mean):.2e}")
    print(f"  Online var error:   {abs(online.get_variance() - true_var):.2e}")
    print(f"  Batch var error:    {abs(batch.get_variance() - true_var):.2e}")
    print()
    
    # Verdict
    online_var_error = abs(online.get_variance() - true_var)
    batch_var_error = abs(batch.get_variance() - true_var)
    
    if batch_var_error > online_var_error * 10:
        print("âš ï¸  CATASTROPHIC CANCELLATION DETECTED in batch algorithm!")
        print(f"   Batch algorithm variance error is {batch_var_error/online_var_error:.1f}x larger")
    else:
        print("âœ… Both algorithms performed similarly on this test")


def test_correctness():
    """Test correctness on simple known data."""
    print("\n" + "="*70)
    print("TEST 2: CORRECTNESS ON SIMPLE DATA")
    print("="*70)
    
    data = [2, 4, 4, 4, 5, 5, 7, 9]
    
    print(f"\nData: {data}")
    print()
    
    # NumPy (ground truth)
    np_mean = np.mean(data)
    np_var = np.var(data, ddof=1)
    np_std = np.std(data, ddof=1)
    
    # Our algorithms
    online, batch = compare_algorithms(data)
    
    print("RESULTS:")
    print(f"  NumPy:   mean={np_mean:.6f}, var={np_var:.6f}, std={np_std:.6f}")
    print(f"  Online:  mean={online.get_mean():.6f}, var={online.get_variance():.6f}, std={online.get_std():.6f}")
    print(f"  Batch:   mean={batch.get_mean():.6f}, var={batch.get_variance():.6f}, std={batch.get_std():.6f}")
    print()
    
    # Check accuracy
    tol = 1e-10
    assert abs(online.get_mean() - np_mean) < tol, "Online mean incorrect"
    assert abs(online.get_variance() - np_var) < tol, "Online variance incorrect"
    print("âœ… All correctness tests passed!")


def test_incremental_updates():
    """Visualize how statistics evolve as data arrives."""
    print("\n" + "="*70)
    print("TEST 3: INCREMENTAL UPDATES VISUALIZATION")
    print("="*70)
    
    # Generate data with changing mean and variance
    np.random.seed(42)
    n_points = 100
    
    # Phase 1: Low variance around 5
    data1 = np.random.normal(5, 1, n_points // 2)
    # Phase 2: Higher variance around 10
    data2 = np.random.normal(10, 3, n_points // 2)
    
    data = np.concatenate([data1, data2])
    
    # Track statistics at each step
    online = OnlineStatistics()
    means = []
    stds = []
    
    for x in data:
        online.update(x)
        means.append(online.get_mean())
        stds.append(online.get_std())
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Data points
    ax1.plot(data, 'o-', alpha=0.6, markersize=4, label='Data points')
    ax1.axvline(n_points // 2, color='red', linestyle='--', label='Phase change')
    ax1.set_xlabel('Observation number', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Incoming Data Stream', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Running mean
    ax2.plot(means, 'b-', linewidth=2, label='Running mean')
    ax2.axvline(n_points // 2, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(np.mean(data1), color='green', linestyle=':', alpha=0.5, label='Phase 1 true mean')
    ax2.axhline(np.mean(data2), color='orange', linestyle=':', alpha=0.5, label='Phase 2 true mean')
    ax2.set_xlabel('Observation number', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean', fontsize=11, fontweight='bold')
    ax2.set_title('Evolution of Running Mean', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Running standard deviation
    ax3.plot(stds, 'r-', linewidth=2, label='Running std dev')
    ax3.axvline(n_points // 2, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(np.std(data1, ddof=1), color='green', linestyle=':', alpha=0.5, label='Phase 1 true std')
    ax3.axhline(np.std(data2, ddof=1), color='orange', linestyle=':', alpha=0.5, label='Phase 2 true std')
    ax3.set_xlabel('Observation number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax3.set_title('Evolution of Running Standard Deviation', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assets/images/online_statistics_evolution.png', dpi=300, bbox_inches='tight')
    print("\nâœ… Visualization saved: assets/images/online_statistics_evolution.png")
    plt.close()


def test_memory_efficiency():
    """Compare memory usage between online and batch methods."""
    print("\n" + "="*70)
    print("TEST 4: MEMORY EFFICIENCY")
    print("="*70)
    
    import sys
    
    n = 1000000  # 1 million points
    
    print(f"\nProcessing {n:,} data points...")
    print()
    
    # Online algorithm
    online = OnlineStatistics()
    online_memory_before = sys.getsizeof(online.__dict__)
    
    for i in range(n):
        online.update(float(i))
    
    online_memory_after = sys.getsizeof(online.__dict__)
    
    # Batch algorithm
    batch = BatchStatistics()
    batch_memory_before = sys.getsizeof(batch.data)
    
    for i in range(n):
        batch.update(float(i))
    
    batch_memory_after = sys.getsizeof(batch.data)
    
    print("MEMORY USAGE:")
    print(f"  Online algorithm: ~{online_memory_after - online_memory_before} bytes (constant)")
    print(f"  Batch algorithm:  ~{(batch_memory_after - batch_memory_before) / 1024 / 1024:.2f} MB (linear growth)")
    print()
    print(f"  Memory ratio: {(batch_memory_after - batch_memory_before) / max(1, online_memory_after - online_memory_before):.0f}:1")
    print()
    print("âš ï¸  Batch algorithm memory scales with n; online algorithm uses O(1) memory!")


def main():
    """Run all tests."""
    print("="*70)
    print("ONLINE ALGORITHMS FOR MEAN AND VARIANCE")
    print("="*70)
    
    # Create output directory
    import os
    os.makedirs('assets/images', exist_ok=True)
    
    # Run tests
    test_correctness()
    test_numerical_stability()
    test_incremental_updates()
    test_memory_efficiency()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()