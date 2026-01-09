"""
Tests for KDTree optimization.

Verifies that optimized implementation produces identical results
to naive implementation, and benchmarks performance.
"""

import pytest
import numpy as np
import time
from typing import List, Tuple

from boid import Boid
from flock import Flock, SimulationParams
from flock_optimized import FlockOptimized
from rules import compute_separation, compute_alignment, compute_cohesion
from rules_optimized import (
    FlockState, 
    compute_separation_kdtree,
    compute_alignment_kdtree,
    compute_cohesion_kdtree,
    compute_all_rules_kdtree
)


class TestFlockStateKDTree:
    """Tests for FlockState spatial indexing."""
    
    def test_empty_flock(self):
        """FlockState handles empty boid list."""
        state = FlockState([])
        assert state.positions.shape == (0, 2)
        assert state.query_neighbors(0, 100) == []
    
    def test_single_boid(self):
        """FlockState handles single boid."""
        boid = Boid(x=100, y=100, vx=1, vy=0)
        state = FlockState([boid])
        
        assert state.positions.shape == (1, 2)
        assert state.query_neighbors(0, 100) == []  # No neighbors
    
    def test_neighbor_query_finds_close_boids(self):
        """query_neighbors finds boids within radius."""
        boids = [
            Boid(x=100, y=100, vx=0, vy=0),  # Index 0
            Boid(x=110, y=100, vx=0, vy=0),  # Index 1, 10px away
            Boid(x=200, y=100, vx=0, vy=0),  # Index 2, 100px away
        ]
        state = FlockState(boids)
        
        neighbors = state.query_neighbors(0, 50)
        
        assert 1 in neighbors
        assert 2 not in neighbors
        assert 0 not in neighbors  # Self excluded
    
    def test_update_rebuilds_tree(self):
        """update() rebuilds tree with new positions."""
        boids = [
            Boid(x=100, y=100, vx=0, vy=0),
            Boid(x=200, y=100, vx=0, vy=0),  # 100px away
        ]
        state = FlockState(boids)
        
        # Initially not neighbors
        assert state.query_neighbors(0, 50) == []
        
        # Move boid 1 closer
        boids[1].x = 110
        state.update()
        
        # Now should be neighbors
        assert 1 in state.query_neighbors(0, 50)


class TestRulesEquivalence:
    """Verify KDTree rules produce identical results to naive rules."""
    
    def create_test_flock(self, n: int = 20, seed: int = 42) -> List[Boid]:
        """Create reproducible test flock."""
        np.random.seed(seed)
        return [
            Boid(
                x=np.random.uniform(0, 800),
                y=np.random.uniform(0, 600),
                vx=np.random.uniform(-3, 3),
                vy=np.random.uniform(-3, 3)
            )
            for _ in range(n)
        ]
    
    def test_separation_equivalence(self):
        """KDTree separation matches naive separation."""
        boids = self.create_test_flock(20)
        state = FlockState(boids)
        
        protected_range = 12
        strength = 0.15
        
        for i, boid in enumerate(boids):
            naive_dv = compute_separation(boid, boids, protected_range, strength)
            kdtree_dv = compute_separation_kdtree(i, state, protected_range, strength)
            
            assert naive_dv[0] == pytest.approx(kdtree_dv[0], abs=1e-10)
            assert naive_dv[1] == pytest.approx(kdtree_dv[1], abs=1e-10)
    
    def test_alignment_equivalence(self):
        """KDTree alignment matches naive alignment."""
        boids = self.create_test_flock(20)
        state = FlockState(boids)
        
        visual_range = 50
        protected_range = 12
        matching_factor = 0.06
        
        for i, boid in enumerate(boids):
            naive_dv = compute_alignment(boid, boids, visual_range, protected_range, matching_factor)
            kdtree_dv = compute_alignment_kdtree(i, state, visual_range, protected_range, matching_factor)
            
            assert naive_dv[0] == pytest.approx(kdtree_dv[0], abs=1e-10)
            assert naive_dv[1] == pytest.approx(kdtree_dv[1], abs=1e-10)
    
    def test_cohesion_equivalence(self):
        """KDTree cohesion matches naive cohesion."""
        boids = self.create_test_flock(20)
        state = FlockState(boids)
        
        visual_range = 50
        protected_range = 12
        centering_factor = 0.002
        
        for i, boid in enumerate(boids):
            naive_dv = compute_cohesion(boid, boids, visual_range, protected_range, centering_factor)
            kdtree_dv = compute_cohesion_kdtree(i, state, visual_range, protected_range, centering_factor)
            
            assert naive_dv[0] == pytest.approx(kdtree_dv[0], abs=1e-10)
            assert naive_dv[1] == pytest.approx(kdtree_dv[1], abs=1e-10)
    
    def test_combined_rules_equivalence(self):
        """Combined KDTree rules match sum of naive rules."""
        boids = self.create_test_flock(20)
        state = FlockState(boids)
        
        visual_range = 50
        protected_range = 12
        cohesion_factor = 0.002
        alignment_factor = 0.06
        separation_strength = 0.15
        
        for i, boid in enumerate(boids):
            # Naive: sum of individual rules
            sep_dv = compute_separation(boid, boids, protected_range, separation_strength)
            align_dv = compute_alignment(boid, boids, visual_range, protected_range, alignment_factor)
            coh_dv = compute_cohesion(boid, boids, visual_range, protected_range, cohesion_factor)
            
            naive_total = (
                sep_dv[0] + align_dv[0] + coh_dv[0],
                sep_dv[1] + align_dv[1] + coh_dv[1]
            )
            
            # KDTree: combined function
            kdtree_total = compute_all_rules_kdtree(
                i, state, visual_range, protected_range,
                cohesion_factor, alignment_factor, separation_strength
            )
            
            assert naive_total[0] == pytest.approx(kdtree_total[0], abs=1e-10)
            assert naive_total[1] == pytest.approx(kdtree_total[1], abs=1e-10)


class TestFlockOptimizedEquivalence:
    """Verify FlockOptimized produces same behavior as Flock."""
    
    def test_single_step_positions_similar(self):
        """After one step, optimized flock has similar positions.
        
        Note: Not exactly identical because naive Flock uses sequential
        updates while FlockOptimized uses parallel semantics.
        """
        np.random.seed(42)
        params = SimulationParams()
        
        # Create identical starting conditions
        naive_flock = Flock(num_boids=0, params=params)
        optimized_flock = FlockOptimized(num_boids=0, params=params)
        
        # Add same boids to both
        for _ in range(20):
            np.random.seed(42 + _)  # Same seed for each boid
            boid_naive = Boid.create_random(
                width=params.width,
                height=params.height,
                max_speed=params.max_speed
            )
            np.random.seed(42 + _)
            boid_opt = Boid.create_random(
                width=params.width,
                height=params.height,
                max_speed=params.max_speed
            )
            naive_flock.boids.append(boid_naive)
            optimized_flock.boids.append(boid_opt)
        
        # Verify starting positions match
        naive_pos = naive_flock.get_positions()
        opt_pos = optimized_flock.get_positions()
        np.testing.assert_array_almost_equal(naive_pos, opt_pos)
        
        # Run one step
        naive_flock.update()
        optimized_flock.update()
        
        # Positions should be similar (not exact due to sequential vs parallel)
        naive_pos = naive_flock.get_positions()
        opt_pos = optimized_flock.get_positions()
        
        # Check that they're in the same ballpark (within 1 pixel)
        diff = np.abs(naive_pos - opt_pos)
        assert np.max(diff) < 1.0, f"Max position difference: {np.max(diff)}"
    
    def test_long_run_stability(self):
        """Optimized flock remains stable over many steps."""
        np.random.seed(42)
        params = SimulationParams()
        flock = FlockOptimized(num_boids=50, params=params)
        
        # Run for many steps
        for _ in range(500):
            flock.update()
        
        # All boids should still be roughly within bounds
        positions = flock.get_positions()
        assert np.all(positions[:, 0] > -50)
        assert np.all(positions[:, 0] < params.width + 50)
        assert np.all(positions[:, 1] > -50)
        assert np.all(positions[:, 1] < params.height + 50)


class TestBenchmarks:
    """Performance benchmarks comparing naive vs KDTree."""
    
    def time_flock_update(self, flock, num_steps: int = 100) -> float:
        """Time average update duration."""
        # Warm up
        for _ in range(10):
            flock.update()
        
        # Timed run
        start = time.perf_counter()
        for _ in range(num_steps):
            flock.update()
        elapsed = time.perf_counter() - start
        
        return elapsed / num_steps
    
    def test_benchmark_50_boids(self):
        """Benchmark with 50 boids."""
        np.random.seed(42)
        params = SimulationParams()
        
        naive_flock = Flock(num_boids=50, params=params)
        np.random.seed(42)
        optimized_flock = FlockOptimized(num_boids=50, params=params)
        
        naive_time = self.time_flock_update(naive_flock)
        optimized_time = self.time_flock_update(optimized_flock)
        
        print(f"\n50 boids: naive={naive_time*1000:.3f}ms, optimized={optimized_time*1000:.3f}ms")
        
        # Just verify both run
        assert naive_time > 0
        assert optimized_time > 0
    
    def test_benchmark_200_boids(self):
        """Benchmark with 200 boids."""
        np.random.seed(42)
        params = SimulationParams()
        
        naive_flock = Flock(num_boids=200, params=params)
        np.random.seed(42)
        optimized_flock = FlockOptimized(num_boids=200, params=params)
        
        naive_time = self.time_flock_update(naive_flock)
        optimized_time = self.time_flock_update(optimized_flock)
        
        print(f"\n200 boids: naive={naive_time*1000:.3f}ms, optimized={optimized_time*1000:.3f}ms")
        
        # Optimized should be faster at this scale
        # (might not be true for very small flocks due to overhead)
        assert naive_time > 0
        assert optimized_time > 0
    
    def test_benchmark_500_boids(self):
        """Benchmark with 500 boids."""
        np.random.seed(42)
        params = SimulationParams()
        
        naive_flock = Flock(num_boids=500, params=params)
        np.random.seed(42)
        optimized_flock = FlockOptimized(num_boids=500, params=params)
        
        naive_time = self.time_flock_update(naive_flock, num_steps=50)
        optimized_time = self.time_flock_update(optimized_flock, num_steps=50)
        
        print(f"\n500 boids: naive={naive_time*1000:.3f}ms, optimized={optimized_time*1000:.3f}ms")
        print(f"Speedup: {naive_time/optimized_time:.2f}x")
        
        # Optimized should definitely be faster at 500 boids
        assert optimized_time < naive_time


def run_full_benchmark():
    """Run comprehensive benchmark and generate report."""
    print("\n" + "="*60)
    print("BOIDS PERFORMANCE BENCHMARK: Naive vs KDTree")
    print("="*60)
    
    results = []
    flock_sizes = [50, 100, 200, 300, 500]
    num_steps = 100
    
    for n in flock_sizes:
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
        
        speedup = naive_time / optimized_time if optimized_time > 0 else 0
        
        results.append({
            'n': n,
            'naive_ms': naive_time,
            'optimized_ms': optimized_time,
            'speedup': speedup
        })
        
        print(f"N={n:4d}: Naive={naive_time:7.3f}ms, KDTree={optimized_time:7.3f}ms, Speedup={speedup:.2f}x")
    
    print("\n" + "-"*60)
    print("SUMMARY TABLE")
    print("-"*60)
    print(f"{'Boids':>8} | {'Naive (ms)':>12} | {'KDTree (ms)':>12} | {'Speedup':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['n']:>8} | {r['naive_ms']:>12.3f} | {r['optimized_ms']:>12.3f} | {r['speedup']:>7.2f}x")
    print("-"*60)
    
    return results


if __name__ == "__main__":
    # Run pytest tests
    pytest.main([__file__, "-v", "-s"])
    
    # Run full benchmark
    run_full_benchmark()