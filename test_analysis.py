"""
Unit tests for analysis module.
"""

import pytest
import numpy as np
from analysis import (
    ExperimentConfig,
    ExperimentResults,
    run_single_experiment,
    run_parameter_sweep
)
from flock import SimulationParams


class TestExperimentConfig:
    """Tests for ExperimentConfig."""
    
    def test_default_base_params(self):
        """Default base params are created if not provided."""
        config = ExperimentConfig(
            predator_speeds=[1.5, 2.5],
            avoidance_strengths=[0.3, 0.7]
        )
        
        assert config.base_params is not None
        assert isinstance(config.base_params, SimulationParams)
    
    def test_custom_settings(self):
        """Custom settings are stored correctly."""
        config = ExperimentConfig(
            predator_speeds=[1.0, 2.0, 3.0],
            avoidance_strengths=[0.1, 0.5, 0.9],
            num_boids=100,
            num_frames=200,
            num_repetitions=3
        )
        
        assert len(config.predator_speeds) == 3
        assert len(config.avoidance_strengths) == 3
        assert config.num_boids == 100
        assert config.num_frames == 200
        assert config.num_repetitions == 3


class TestRunSingleExperiment:
    """Tests for single experiment runs."""
    
    def test_produces_valid_metrics(self):
        """Single experiment produces valid metrics."""
        config = ExperimentConfig(
            predator_speeds=[2.5],
            avoidance_strengths=[0.5],
            num_boids=10,
            num_frames=50,
            num_repetitions=1
        )
        
        metrics = run_single_experiment(
            predator_speed=2.5,
            avoidance_strength=0.5,
            config=config,
            seed=42
        )
        
        assert metrics.num_frames == 50
        assert metrics.mean_avg_distance > 0
        assert metrics.mean_min_distance > 0
    
    def test_different_seeds_produce_different_results(self):
        """Different seeds produce different metrics."""
        config = ExperimentConfig(
            predator_speeds=[2.5],
            avoidance_strengths=[0.5],
            num_boids=10,
            num_frames=50,
            num_repetitions=1
        )
        
        metrics1 = run_single_experiment(2.5, 0.5, config, seed=42)
        metrics2 = run_single_experiment(2.5, 0.5, config, seed=123)
        
        # Different seeds should produce different results
        assert metrics1.mean_avg_distance != metrics2.mean_avg_distance
    
    def test_same_seed_produces_same_results(self):
        """Same seed produces reproducible results."""
        config = ExperimentConfig(
            predator_speeds=[2.5],
            avoidance_strengths=[0.5],
            num_boids=10,
            num_frames=50,
            num_repetitions=1
        )
        
        metrics1 = run_single_experiment(2.5, 0.5, config, seed=42)
        metrics2 = run_single_experiment(2.5, 0.5, config, seed=42)
        
        assert metrics1.mean_avg_distance == metrics2.mean_avg_distance


class TestParameterSweep:
    """Tests for full parameter sweep."""
    
    def test_small_sweep_completes(self):
        """Small parameter sweep completes without error."""
        config = ExperimentConfig(
            predator_speeds=[2.0, 3.0],
            avoidance_strengths=[0.3, 0.7],
            num_boids=10,
            num_frames=20,
            num_repetitions=2
        )
        
        results = run_parameter_sweep(config, verbose=False)
        
        assert results.total_runs == 2 * 2 * 2  # 8 runs
        assert results.num_repetitions == 2
        assert results.num_frames == 20
    
    def test_result_shapes_correct(self):
        """Result arrays have correct shapes."""
        config = ExperimentConfig(
            predator_speeds=[1.5, 2.5, 3.5],
            avoidance_strengths=[0.2, 0.5],
            num_boids=10,
            num_frames=20,
            num_repetitions=2
        )
        
        results = run_parameter_sweep(config, verbose=False)
        
        # Shape should be (n_speeds, n_strengths)
        assert results.mean_avg_distance.shape == (3, 2)
        assert results.std_avg_distance.shape == (3, 2)
        assert results.mean_min_distance.shape == (3, 2)
        assert results.mean_cohesion.shape == (3, 2)
    
    def test_higher_avoidance_increases_distance(self):
        """Higher avoidance strength should generally increase distance to predator."""
        config = ExperimentConfig(
            predator_speeds=[2.5],
            avoidance_strengths=[0.1, 0.9],  # Low vs high avoidance
            num_boids=20,
            num_frames=100,
            num_repetitions=3
        )
        
        results = run_parameter_sweep(config, verbose=False)
        
        low_avoidance_dist = results.mean_avg_distance[0, 0]
        high_avoidance_dist = results.mean_avg_distance[0, 1]
        
        # Higher avoidance should lead to greater distance (usually)
        # This is a statistical tendency, not absolute
        assert high_avoidance_dist >= low_avoidance_dist * 0.8  # Allow some variance


class TestExperimentResults:
    """Tests for ExperimentResults data structure."""
    
    def test_results_contain_metadata(self):
        """Results contain experiment metadata."""
        config = ExperimentConfig(
            predator_speeds=[2.0, 3.0],
            avoidance_strengths=[0.5],
            num_boids=10,
            num_frames=30,
            num_repetitions=2
        )
        
        results = run_parameter_sweep(config, verbose=False)
        
        assert results.num_repetitions == 2
        assert results.num_frames == 30
        assert results.total_runs == 4
        assert results.elapsed_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])