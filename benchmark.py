"""
Generate performance comparison figure for Tier 1 documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from flock import Flock, SimulationParams
from flock_optimized import FlockOptimized


def run_benchmark():
    """Run benchmark and return results."""
    results = {
        'n': [],
        'naive_ms': [],
        'optimized_ms': []
    }
    
    flock_sizes = [50, 100, 150, 200, 300, 400, 500]
    num_steps = 100
    
    for n in flock_sizes:
        print(f"Benchmarking N={n}...")
        np.random.seed(42)
        params = SimulationParams()
        
        # Create flocks
        naive_flock = Flock(num_boids=n, params=params)
        np.random.seed(42)
        optimized_flock = FlockOptimized(num_boids=n, params=params)
        
        # Warm up
        for _ in range(10):
            naive_flock.update()
            optimized_flock.update()
        
        # Time naive
        start = time.perf_counter()
        for _ in range(num_steps):
            naive_flock.update()
        naive_time = (time.perf_counter() - start) / num_steps * 1000
        
        # Time optimized
        start = time.perf_counter()
        for _ in range(num_steps):
            optimized_flock.update()
        optimized_time = (time.perf_counter() - start) / num_steps * 1000
        
        results['n'].append(n)
        results['naive_ms'].append(naive_time)
        results['optimized_ms'].append(optimized_time)
    
    return results


def create_figure(results):
    """Generate performance comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n = results['n']
    naive = results['naive_ms']
    optimized = results['optimized_ms']
    speedup = [naive[i] / optimized[i] for i in range(len(n))]
    
    # Plot 1: Execution time comparison
    ax1 = axes[0]
    ax1.plot(n, naive, 'o-', label='Naive O(n²)', linewidth=2, markersize=8, color='#e74c3c')
    ax1.plot(n, optimized, 's-', label='KDTree O(n log n)', linewidth=2, markersize=8, color='#27ae60')
    ax1.set_xlabel('Number of Boids', fontsize=12)
    ax1.set_ylabel('Frame Time (ms)', fontsize=12)
    ax1.set_title('Execution Time vs Flock Size', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(n) + 50)
    ax1.set_ylim(0, max(naive) * 1.1)
    
    # Add 60fps and 30fps reference lines
    ax1.axhline(y=16.67, color='gray', linestyle='--', alpha=0.7, label='60 FPS')
    ax1.axhline(y=33.33, color='gray', linestyle=':', alpha=0.7, label='30 FPS')
    ax1.text(max(n) - 30, 18, '60 FPS', fontsize=9, color='gray')
    ax1.text(max(n) - 30, 35, '30 FPS', fontsize=9, color='gray')
    
    # Plot 2: Speedup factor
    ax2 = axes[1]
    bars = ax2.bar(n, speedup, color='#3498db', width=30, edgecolor='black')
    ax2.set_xlabel('Number of Boids', fontsize=12)
    ax2.set_ylabel('Speedup Factor (x)', fontsize=12)
    ax2.set_title('KDTree Speedup vs Naive Implementation', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, s in zip(bars, speedup):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                 f'{s:.1f}x', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("Figure saved to benchmark_results.png")
    
    return fig


def print_table(results):
    """Print markdown table for documentation."""
    print("\n### Performance Results\n")
    print("| Boids | Naive (ms) | KDTree (ms) | Speedup |")
    print("|------:|-----------:|------------:|--------:|")
    
    for i in range(len(results['n'])):
        n = results['n'][i]
        naive = results['naive_ms'][i]
        opt = results['optimized_ms'][i]
        speedup = naive / opt
        print(f"| {n} | {naive:.2f} | {opt:.2f} | {speedup:.2f}x |")


if __name__ == "__main__":
    print("Running performance benchmark...")
    results = run_benchmark()
    
    print("\nGenerating figure...")
    create_figure(results)
    
    print_table(results)