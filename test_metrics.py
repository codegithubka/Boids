"""
Unit tests for metrics module.
"""

import pytest
import numpy as np
from boid import Boid
from predator import Predator
from metrics import (
    compute_distance_to_predator,
    compute_avg_distance_to_predator,
    compute_min_distance_to_predator,
    compute_flock_center,
    compute_flock_cohesion,
    compute_flock_spread,
    FrameMetrics,
    RunMetrics,
    MetricsCollector,
    run_simulation_with_metrics
)
from flock import SimulationParams
from flock_optimized import FlockOptimized


class TestDistanceMetrics:
    """Tests for distance-to-predator metrics."""
    
    def test_distance_to_predator_horizontal(self):
        """Distance computed correctly for horizontal separation."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        predator = Predator(x=200, y=100, vx=0, vy=0)
        
        dist = compute_distance_to_predator(boid, predator)
        
        assert dist == 100.0
    
    def test_distance_to_predator_diagonal(self):
        """Distance computed correctly for diagonal separation."""
        boid = Boid(x=0, y=0, vx=0, vy=0)
        predator = Predator(x=3, y=4, vx=0, vy=0)
        
        dist = compute_distance_to_predator(boid, predator)
        
        assert dist == 5.0  # 3-4-5 triangle
    
    def test_distance_to_predator_same_position(self):
        """Distance is zero when at same position."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        predator = Predator(x=100, y=100, vx=0, vy=0)
        
        dist = compute_distance_to_predator(boid, predator)
        
        assert dist == 0.0
    
    def test_avg_distance_empty_flock(self):
        """Average distance is 0 for empty flock."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        
        avg_dist = compute_avg_distance_to_predator([], predator)
        
        assert avg_dist == 0.0
    
    def test_avg_distance_single_boid(self):
        """Average distance equals single boid distance."""
        boid = Boid(x=0, y=0, vx=0, vy=0)
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        avg_dist = compute_avg_distance_to_predator([boid], predator)
        
        assert avg_dist == 100.0
    
    def test_avg_distance_multiple_boids(self):
        """Average distance computed correctly for multiple boids."""
        boids = [
            Boid(x=0, y=0, vx=0, vy=0),    # 100 from predator
            Boid(x=200, y=0, vx=0, vy=0),  # 100 from predator
        ]
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        avg_dist = compute_avg_distance_to_predator(boids, predator)
        
        assert avg_dist == 100.0  # (100 + 100) / 2
    
    def test_min_distance_empty_flock(self):
        """Min distance is infinity for empty flock."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        
        min_dist = compute_min_distance_to_predator([], predator)
        
        assert min_dist == float('inf')
    
    def test_min_distance_multiple_boids(self):
        """Min distance finds closest boid."""
        boids = [
            Boid(x=0, y=0, vx=0, vy=0),    # 100 from predator
            Boid(x=150, y=0, vx=0, vy=0),  # 50 from predator (closest)
            Boid(x=300, y=0, vx=0, vy=0),  # 200 from predator
        ]
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        min_dist = compute_min_distance_to_predator(boids, predator)
        
        assert min_dist == 50.0


class TestFlockMetrics:
    """Tests for flock cohesion metrics."""
    
    def test_flock_center_empty(self):
        """Flock center is (0,0) for empty flock."""
        center = compute_flock_center([])
        
        assert center == (0.0, 0.0)
    
    def test_flock_center_single_boid(self):
        """Flock center is boid position for single boid."""
        boid = Boid(x=100, y=200, vx=0, vy=0)
        
        center = compute_flock_center([boid])
        
        assert center == (100.0, 200.0)
    
    def test_flock_center_multiple_boids(self):
        """Flock center is average position."""
        boids = [
            Boid(x=0, y=0, vx=0, vy=0),
            Boid(x=100, y=100, vx=0, vy=0),
            Boid(x=200, y=200, vx=0, vy=0),
        ]
        
        center = compute_flock_center(boids)
        
        assert center == (100.0, 100.0)
    
    def test_cohesion_empty_flock(self):
        """Cohesion is 0 for empty flock."""
        cohesion = compute_flock_cohesion([])
        
        assert cohesion == 0.0
    
    def test_cohesion_single_boid(self):
        """Cohesion is 0 for single boid."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        cohesion = compute_flock_cohesion([boid])
        
        assert cohesion == 0.0
    
    def test_cohesion_tight_flock(self):
        """Tight flock has low cohesion value."""
        boids = [
            Boid(x=100, y=100, vx=0, vy=0),
            Boid(x=101, y=100, vx=0, vy=0),
            Boid(x=100, y=101, vx=0, vy=0),
        ]
        
        cohesion = compute_flock_cohesion(boids)
        
        assert cohesion < 1.0  # Very small spread
    
    def test_cohesion_dispersed_flock(self):
        """Dispersed flock has high cohesion value."""
        boids = [
            Boid(x=0, y=0, vx=0, vy=0),
            Boid(x=400, y=0, vx=0, vy=0),
            Boid(x=800, y=0, vx=0, vy=0),
        ]
        
        cohesion = compute_flock_cohesion(boids)
        
        assert cohesion > 100.0  # Large spread
    
    def test_spread_empty_flock(self):
        """Spread is 0 for empty flock."""
        spread = compute_flock_spread([])
        
        assert spread == 0.0
    
    def test_spread_single_boid(self):
        """Spread is 0 for single boid."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        spread = compute_flock_spread([boid])
        
        assert spread == 0.0
    
    def test_spread_two_boids(self):
        """Spread equals distance between two boids."""
        boids = [
            Boid(x=0, y=0, vx=0, vy=0),
            Boid(x=100, y=0, vx=0, vy=0),
        ]
        
        spread = compute_flock_spread(boids)
        
        assert spread == 100.0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""
    
    def test_empty_collector(self):
        """Empty collector returns zero metrics."""
        collector = MetricsCollector()
        
        results = collector.summarize()
        
        assert results.num_frames == 0
        assert results.mean_avg_distance == 0.0
    
    def test_record_single_frame(self):
        """Single frame recorded correctly."""
        collector = MetricsCollector()
        boids = [Boid(x=0, y=0, vx=0, vy=0)]
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        collector.record_frame(boids, predator)
        results = collector.summarize()
        
        assert results.num_frames == 1
        assert results.mean_avg_distance == 100.0
        assert results.mean_min_distance == 100.0
    
    def test_record_multiple_frames(self):
        """Multiple frames aggregated correctly."""
        collector = MetricsCollector()
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        # Frame 1: boid at distance 100
        boids1 = [Boid(x=0, y=0, vx=0, vy=0)]
        collector.record_frame(boids1, predator)
        
        # Frame 2: boid at distance 50
        boids2 = [Boid(x=50, y=0, vx=0, vy=0)]
        collector.record_frame(boids2, predator)
        
        results = collector.summarize()
        
        assert results.num_frames == 2
        assert results.mean_avg_distance == 75.0  # (100 + 50) / 2
        assert results.overall_min_distance == 50.0
    
    def test_skip_frame_without_predator(self):
        """Frames without predator are skipped."""
        collector = MetricsCollector()
        boids = [Boid(x=0, y=0, vx=0, vy=0)]
        
        collector.record_frame(boids, None)  # No predator
        
        results = collector.summarize()
        
        assert results.num_frames == 0
    
    def test_reset_clears_data(self):
        """Reset clears all recorded metrics."""
        collector = MetricsCollector()
        boids = [Boid(x=0, y=0, vx=0, vy=0)]
        predator = Predator(x=100, y=0, vx=0, vy=0)
        
        collector.record_frame(boids, predator)
        assert collector.summarize().num_frames == 1
        
        collector.reset()
        
        assert collector.summarize().num_frames == 0


class TestRunSimulationWithMetrics:
    """Tests for full simulation with metrics collection."""
    
    def test_run_with_predator(self):
        """Simulation with predator produces valid metrics."""
        np.random.seed(42)
        params = SimulationParams()
        flock = FlockOptimized(num_boids=10, params=params, enable_predator=True)
        
        results = run_simulation_with_metrics(flock, num_frames=50)
        
        assert results.num_frames == 50
        assert results.mean_avg_distance > 0
        assert results.mean_min_distance > 0
        assert results.overall_min_distance > 0
        assert results.overall_min_distance <= results.mean_min_distance
    
    def test_run_without_predator(self):
        """Simulation without predator produces no metrics."""
        np.random.seed(42)
        params = SimulationParams()
        flock = FlockOptimized(num_boids=10, params=params, enable_predator=False)
        
        results = run_simulation_with_metrics(flock, num_frames=50)
        
        assert results.num_frames == 0  # No frames recorded (no predator)
    
    def test_metrics_vary_with_seed(self):
        """Different seeds produce different metrics."""
        params = SimulationParams()
        
        np.random.seed(42)
        flock1 = FlockOptimized(num_boids=10, params=params, enable_predator=True)
        results1 = run_simulation_with_metrics(flock1, num_frames=50)
        
        np.random.seed(123)
        flock2 = FlockOptimized(num_boids=10, params=params, enable_predator=True)
        results2 = run_simulation_with_metrics(flock2, num_frames=50)
        
        # Results should differ (different random initializations)
        assert results1.mean_avg_distance != results2.mean_avg_distance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])