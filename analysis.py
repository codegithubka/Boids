"""
Parameter sweep and analysis for predator-prey dynamics.

Runs experiments varying predator speed and avoidance strength,
collects metrics, and generates heatmap visualizations (Tier 3).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

from flock import SimulationParams
from flock_optimized import FlockOptimized
from metrics import run_simulation_with_metrics, RunMetrics


@dataclass
class ExperimentConfig:
    """Configuration for a parameter sweep experiment."""
    
    # Parameter 1: Predator speed
    predator_speeds: List[float]
    
    # Parameter 2: Avoidance strength
    avoidance_strengths: List[float]
    
    # Experiment settings
    num_boids: int = 50
    num_frames: int = 500
    num_repetitions: int = 5
    
    # Base simulation parameters (others kept at defaults)
    base_params: SimulationParams = None
    
    def __post_init__(self):
        if self.base_params is None:
            self.base_params = SimulationParams()


@dataclass
class ExperimentResults:
    """Results from a parameter sweep experiment."""
    
    # Parameter values (axes)
    predator_speeds: np.ndarray
    avoidance_strengths: np.ndarray
    
    # Metric grids (shape: len(speeds) x len(strengths))
    mean_avg_distance: np.ndarray  # Mean of avg distance across repetitions
    std_avg_distance: np.ndarray   # Std of avg distance across repetitions
    
    mean_min_distance: np.ndarray  # Mean of min distance across repetitions
    std_min_distance: np.ndarray
    
    mean_cohesion: np.ndarray      # Mean cohesion across repetitions
    std_cohesion: np.ndarray
    
    # Metadata
    num_repetitions: int = 0
    num_frames: int = 0
    total_runs: int = 0
    elapsed_time: float = 0.0


def run_single_experiment(
    predator_speed: float,
    avoidance_strength: float,
    config: ExperimentConfig,
    seed: int
) -> RunMetrics:
    """
    Run a single simulation with given parameters.
    
    Args:
        predator_speed: Predator speed parameter
        avoidance_strength: Predator avoidance strength parameter
        config: Experiment configuration
        seed: Random seed for reproducibility
        
    Returns:
        RunMetrics from the simulation
    """
    np.random.seed(seed)
    
    # Create params with modified predator settings
    params = SimulationParams(
        width=config.base_params.width,
        height=config.base_params.height,
        visual_range=config.base_params.visual_range,
        protected_range=config.base_params.protected_range,
        max_speed=config.base_params.max_speed,
        min_speed=config.base_params.min_speed,
        cohesion_factor=config.base_params.cohesion_factor,
        alignment_factor=config.base_params.alignment_factor,
        separation_strength=config.base_params.separation_strength,
        margin=config.base_params.margin,
        turn_factor=config.base_params.turn_factor,
        # Varied parameters:
        predator_speed=predator_speed,
        predator_avoidance_strength=avoidance_strength,
        predator_detection_range=config.base_params.predator_detection_range,
        predator_hunting_strength=config.base_params.predator_hunting_strength
    )
    
    # Create flock with predator
    flock = FlockOptimized(
        num_boids=config.num_boids,
        params=params,
        enable_predator=True
    )
    
    # Run simulation and collect metrics
    return run_simulation_with_metrics(flock, num_frames=config.num_frames)


def run_parameter_sweep(config: ExperimentConfig, verbose: bool = True) -> ExperimentResults:
    """
    Run full parameter sweep experiment.
    
    Args:
        config: Experiment configuration
        verbose: If True, print progress
        
    Returns:
        ExperimentResults with all metrics
    """
    start_time = time.time()
    
    n_speeds = len(config.predator_speeds)
    n_strengths = len(config.avoidance_strengths)
    n_reps = config.num_repetitions
    
    total_runs = n_speeds * n_strengths * n_reps
    
    if verbose:
        print(f"Running parameter sweep: {n_speeds} speeds × {n_strengths} strengths × {n_reps} reps = {total_runs} runs")
    
    # Storage for all repetitions
    all_avg_dist = np.zeros((n_speeds, n_strengths, n_reps))
    all_min_dist = np.zeros((n_speeds, n_strengths, n_reps))
    all_cohesion = np.zeros((n_speeds, n_strengths, n_reps))
    
    run_count = 0
    
    for i, speed in enumerate(config.predator_speeds):
        for j, strength in enumerate(config.avoidance_strengths):
            for rep in range(n_reps):
                seed = 1000 * i + 100 * j + rep  # Unique seed per run
                
                metrics = run_single_experiment(speed, strength, config, seed)
                
                all_avg_dist[i, j, rep] = metrics.mean_avg_distance
                all_min_dist[i, j, rep] = metrics.mean_min_distance
                all_cohesion[i, j, rep] = metrics.mean_cohesion
                
                run_count += 1
                
                if verbose and run_count % 10 == 0:
                    elapsed = time.time() - start_time
                    remaining = elapsed / run_count * (total_runs - run_count)
                    print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.0f}%) "
                          f"- ETA: {remaining:.0f}s")
    
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Completed {total_runs} runs in {elapsed_time:.1f}s")
    
    # Compute mean and std across repetitions
    results = ExperimentResults(
        predator_speeds=np.array(config.predator_speeds),
        avoidance_strengths=np.array(config.avoidance_strengths),
        mean_avg_distance=np.mean(all_avg_dist, axis=2),
        std_avg_distance=np.std(all_avg_dist, axis=2),
        mean_min_distance=np.mean(all_min_dist, axis=2),
        std_min_distance=np.std(all_min_dist, axis=2),
        mean_cohesion=np.mean(all_cohesion, axis=2),
        std_cohesion=np.std(all_cohesion, axis=2),
        num_repetitions=n_reps,
        num_frames=config.num_frames,
        total_runs=total_runs,
        elapsed_time=elapsed_time
    )
    
    return results


def create_heatmap(
    results: ExperimentResults,
    metric: str = 'avg_distance',
    title: str = None,
    filename: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create heatmap visualization of experiment results.
    
    Args:
        results: ExperimentResults from parameter sweep
        metric: Which metric to plot ('avg_distance', 'min_distance', 'cohesion')
        title: Plot title (auto-generated if None)
        filename: If provided, save figure to this path
        show: If True, display the figure
        
    Returns:
        matplotlib Figure object
    """
    # Select metric data
    if metric == 'avg_distance':
        data = results.mean_avg_distance
        std_data = results.std_avg_distance
        default_title = 'Average Distance to Predator'
        cmap = 'viridis'  # Higher is better (escape)
        label = 'Avg Distance (px)'
    elif metric == 'min_distance':
        data = results.mean_min_distance
        std_data = results.std_min_distance
        default_title = 'Minimum Distance to Predator'
        cmap = 'viridis'  # Higher is better (survival)
        label = 'Min Distance (px)'
    elif metric == 'cohesion':
        data = results.mean_cohesion
        std_data = results.std_cohesion
        default_title = 'Flock Cohesion (Dispersion)'
        cmap = 'viridis_r'  # Lower is better (tighter flock)
        label = 'Position Std Dev (px)'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    title = title or default_title
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(
        data.T,  # Transpose so x=speed, y=strength
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[
            results.predator_speeds[0],
            results.predator_speeds[-1],
            results.avoidance_strengths[0],
            results.avoidance_strengths[-1]
        ]
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label, fontsize=12)
    
    # Add contour lines
    X, Y = np.meshgrid(results.predator_speeds, results.avoidance_strengths)
    contours = ax.contour(X, Y, data.T, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
    
    # Labels and title
    ax.set_xlabel('Predator Speed', fontsize=12)
    ax.set_ylabel('Avoidance Strength', fontsize=12)
    ax.set_title(f'{title}\n(mean over {results.num_repetitions} runs, {results.num_frames} frames each)', 
                 fontsize=14)
    
    # Add grid
    ax.set_xticks(results.predator_speeds)
    ax.set_yticks(results.avoidance_strengths)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show:
        plt.show()
    
    return fig


def create_combined_figure(
    results: ExperimentResults,
    filename: str = None,
    show: bool = True
) -> plt.Figure:
    """
    Create combined figure with all three metrics.
    
    Args:
        results: ExperimentResults from parameter sweep
        filename: If provided, save figure to this path
        show: If True, display the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [
        ('avg_distance', 'Mean Avg Distance to Predator', 'viridis', 'Avg Distance (px)'),
        ('min_distance', 'Mean Min Distance to Predator', 'viridis', 'Min Distance (px)'),
        ('cohesion', 'Flock Dispersion', 'viridis_r', 'Position Std Dev (px)')
    ]
    
    data_sources = [
        results.mean_avg_distance,
        results.mean_min_distance,
        results.mean_cohesion
    ]
    
    for ax, (metric, title, cmap, label), data in zip(axes, metrics, data_sources):
        im = ax.imshow(
            data.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            extent=[
                results.predator_speeds[0],
                results.predator_speeds[-1],
                results.avoidance_strengths[0],
                results.avoidance_strengths[-1]
            ]
        )
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontsize=10)
        
        # Contour lines
        X, Y = np.meshgrid(results.predator_speeds, results.avoidance_strengths)
        contours = ax.contour(X, Y, data.T, colors='white', alpha=0.5, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=7, fmt='%.0f')
        
        ax.set_xlabel('Predator Speed', fontsize=10)
        ax.set_ylabel('Avoidance Strength', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(results.predator_speeds)
        ax.set_yticks(results.avoidance_strengths)
    
    fig.suptitle(
        f'Predator-Prey Dynamics: Parameter Sweep Results\n'
        f'({results.num_repetitions} repetitions × {results.num_frames} frames, {results.total_runs} total runs)',
        fontsize=13
    )
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    if show:
        plt.show()
    
    return fig


def print_results_table(results: ExperimentResults) -> None:
    """Print results as formatted table."""
    print("\n" + "="*70)
    print("PARAMETER SWEEP RESULTS")
    print("="*70)
    
    print(f"\nExperiment: {results.total_runs} runs in {results.elapsed_time:.1f}s")
    print(f"Settings: {results.num_repetitions} reps × {results.num_frames} frames")
    
    print("\n--- Mean Average Distance to Predator ---")
    print("            ", end="")
    for s in results.avoidance_strengths:
        print(f"  str={s:.1f}", end="")
    print()
    
    for i, speed in enumerate(results.predator_speeds):
        print(f"speed={speed:.1f}  ", end="")
        for j in range(len(results.avoidance_strengths)):
            val = results.mean_avg_distance[i, j]
            print(f"  {val:6.1f}", end="")
        print()
    
    print("\n--- Mean Minimum Distance to Predator ---")
    print("            ", end="")
    for s in results.avoidance_strengths:
        print(f"  str={s:.1f}", end="")
    print()
    
    for i, speed in enumerate(results.predator_speeds):
        print(f"speed={speed:.1f}  ", end="")
        for j in range(len(results.avoidance_strengths)):
            val = results.mean_min_distance[i, j]
            print(f"  {val:6.1f}", end="")
        print()
    
    print("\n" + "="*70)


def run_default_experiment(
    num_reps: int = 5,
    num_frames: int = 500,
    verbose: bool = True
) -> ExperimentResults:
    """
    Run the default parameter sweep experiment.
    
    Args:
        num_reps: Number of repetitions per parameter combination
        num_frames: Frames per simulation run
        verbose: Print progress
        
    Returns:
        ExperimentResults
    """
    config = ExperimentConfig(
        predator_speeds=[1.5, 2.0, 2.5, 3.0, 3.5],
        avoidance_strengths=[0.1, 0.3, 0.5, 0.7, 0.9],
        num_boids=50,
        num_frames=num_frames,
        num_repetitions=num_reps
    )
    
    return run_parameter_sweep(config, verbose=verbose)


if __name__ == "__main__":
    print("Running Tier 3 parameter sweep experiment...")
    print("This may take a few minutes.\n")
    
    # Run experiment
    results = run_default_experiment(num_reps=5, num_frames=500)
    
    # Print table
    print_results_table(results)
    
    # Generate figures
    create_combined_figure(results, filename='parameter_sweep_results.png', show=False)
    
    print("\nExperiment complete!")