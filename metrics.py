"""
Metrics for quantitative analysis of predator-prey dynamics.

Provides functions to measure flock behavior and predator interaction
for parameter sweep experiments (Tier 3).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from boid import Boid
from predator import Predator


def compute_distance_to_predator(boid: Boid, predator: Predator) -> float:
    """
    Compute Euclidean distance from a boid to the predator.
    
    Args:
        boid: The boid
        predator: The predator
        
    Returns:
        Distance in pixels
    """
    dx = boid.x - predator.x
    dy = boid.y - predator.y
    return np.sqrt(dx * dx + dy * dy)


def compute_avg_distance_to_predator(boids: List[Boid], predator: Predator) -> float:
    """
    Compute average distance from all boids to the predator.
    
    Args:
        boids: List of all boids
        predator: The predator
        
    Returns:
        Mean distance in pixels, or 0.0 if no boids
    """
    if not boids:
        return 0.0
    
    total_distance = sum(compute_distance_to_predator(b, predator) for b in boids)
    return total_distance / len(boids)


def compute_min_distance_to_predator(boids: List[Boid], predator: Predator) -> float:
    """
    Compute minimum distance from any boid to the predator.
    
    This represents the "closest call" - how close the predator got
    to catching any boid.
    
    Args:
        boids: List of all boids
        predator: The predator
        
    Returns:
        Minimum distance in pixels, or float('inf') if no boids
    """
    if not boids:
        return float('inf')
    
    return min(compute_distance_to_predator(b, predator) for b in boids)


def compute_flock_center(boids: List[Boid]) -> Tuple[float, float]:
    """
    Compute center of mass of the flock.
    
    Args:
        boids: List of all boids
        
    Returns:
        Tuple (x, y) of flock center, or (0, 0) if no boids
    """
    if not boids:
        return (0.0, 0.0)
    
    sum_x = sum(b.x for b in boids)
    sum_y = sum(b.y for b in boids)
    n = len(boids)
    
    return (sum_x / n, sum_y / n)


def compute_flock_cohesion(boids: List[Boid]) -> float:
    """
    Compute flock cohesion as the standard deviation of boid positions.
    
    Lower values indicate a tighter, more cohesive flock.
    Higher values indicate a dispersed flock.
    
    Uses the average of std(x) and std(y) for a single scalar metric.
    
    Args:
        boids: List of all boids
        
    Returns:
        Cohesion metric (std of positions), or 0.0 if < 2 boids
    """
    if len(boids) < 2:
        return 0.0
    
    x_positions = np.array([b.x for b in boids])
    y_positions = np.array([b.y for b in boids])
    
    std_x = np.std(x_positions)
    std_y = np.std(y_positions)
    
    # Return average of x and y standard deviations
    return (std_x + std_y) / 2


def compute_flock_spread(boids: List[Boid]) -> float:
    """
    Compute flock spread as the maximum distance between any two boids.
    
    Alternative cohesion metric - diameter of the flock.
    
    Args:
        boids: List of all boids
        
    Returns:
        Maximum pairwise distance, or 0.0 if < 2 boids
    """
    if len(boids) < 2:
        return 0.0
    
    max_dist = 0.0
    for i, b1 in enumerate(boids):
        for b2 in boids[i+1:]:
            dx = b1.x - b2.x
            dy = b1.y - b2.y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > max_dist:
                max_dist = dist
    
    return max_dist


@dataclass
class FrameMetrics:
    """Metrics captured for a single simulation frame."""
    avg_distance_to_predator: float
    min_distance_to_predator: float
    flock_cohesion: float


@dataclass
class RunMetrics:
    """Aggregated metrics for a complete simulation run."""
    # Mean values across all frames
    mean_avg_distance: float = 0.0
    mean_min_distance: float = 0.0
    mean_cohesion: float = 0.0
    
    # Minimum values (worst case)
    overall_min_distance: float = float('inf')
    
    # Standard deviations
    std_avg_distance: float = 0.0
    std_min_distance: float = 0.0
    std_cohesion: float = 0.0
    
    # Count
    num_frames: int = 0


class MetricsCollector:
    """
    Collects metrics during a simulation run.
    
    Usage:
        collector = MetricsCollector()
        for frame in simulation:
            collector.record_frame(flock.boids, flock.predator)
        results = collector.summarize()
    """
    
    def __init__(self):
        """Initialize empty metrics collector."""
        self.frame_metrics: List[FrameMetrics] = []
    
    def reset(self):
        """Clear all recorded metrics."""
        self.frame_metrics = []
    
    def record_frame(self, boids: List[Boid], predator: Optional[Predator]) -> None:
        """
        Record metrics for the current frame.
        
        Args:
            boids: List of all boids
            predator: The predator (or None if disabled)
        """
        if predator is None:
            # Skip frames where predator is disabled
            return
        
        metrics = FrameMetrics(
            avg_distance_to_predator=compute_avg_distance_to_predator(boids, predator),
            min_distance_to_predator=compute_min_distance_to_predator(boids, predator),
            flock_cohesion=compute_flock_cohesion(boids)
        )
        
        self.frame_metrics.append(metrics)
    
    def summarize(self) -> RunMetrics:
        """
        Compute summary statistics for the run.
        
        Returns:
            RunMetrics with mean, std, and extreme values
        """
        if not self.frame_metrics:
            return RunMetrics()
        
        avg_distances = [m.avg_distance_to_predator for m in self.frame_metrics]
        min_distances = [m.min_distance_to_predator for m in self.frame_metrics]
        cohesions = [m.flock_cohesion for m in self.frame_metrics]
        
        return RunMetrics(
            mean_avg_distance=np.mean(avg_distances),
            mean_min_distance=np.mean(min_distances),
            mean_cohesion=np.mean(cohesions),
            overall_min_distance=min(min_distances),
            std_avg_distance=np.std(avg_distances),
            std_min_distance=np.std(min_distances),
            std_cohesion=np.std(cohesions),
            num_frames=len(self.frame_metrics)
        )


def run_simulation_with_metrics(
    flock,
    num_frames: int = 500
) -> RunMetrics:
    """
    Run a simulation and collect metrics.
    
    Args:
        flock: Flock or FlockOptimized instance (must have predator enabled)
        num_frames: Number of frames to simulate
        
    Returns:
        RunMetrics summarizing the run
    """
    collector = MetricsCollector()
    
    for _ in range(num_frames):
        # Record metrics before update
        collector.record_frame(flock.boids, flock.predator)
        
        # Advance simulation
        flock.update()
    
    return collector.summarize()