"""
Unit tests for the Flock class.
"""

import pytest
import numpy as np
from boid import Boid
from flock import Flock, SimulationParams


class TestFlockInitialization:
    """Tests for flock creation."""
    
    def test_create_flock_with_default_params(self):
        """Flock can be created with default parameters."""
        flock = Flock(num_boids=10)
        
        assert len(flock.boids) == 10
        assert flock.params.width == 800
        assert flock.params.height == 600
    
    def test_create_flock_with_custom_params(self):
        """Flock can be created with custom parameters."""
        params = SimulationParams(width=1024, height=768, max_speed=10)
        flock = Flock(num_boids=5, params=params)
        
        assert len(flock.boids) == 5
        assert flock.params.width == 1024
        assert flock.params.max_speed == 10
    
    def test_boids_within_bounds(self):
        """All boids start within simulation bounds."""
        np.random.seed(42)
        flock = Flock(num_boids=100)
        
        for boid in flock.boids:
            assert 0 <= boid.x <= flock.params.width
            assert 0 <= boid.y <= flock.params.height


class TestBoundaryHandling:
    """Tests for boundary steering behavior."""
    
    def test_no_steering_in_center(self):
        """Boid in center of screen has no boundary steering."""
        params = SimulationParams(width=800, height=600, margin=100)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=400, y=300, vx=0, vy=0)  # Center of 800x600
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_left_margin_steering(self):
        """Boid near left edge steers right."""
        params = SimulationParams(width=800, height=600, margin=100, turn_factor=0.5)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=50, y=300, vx=0, vy=0)  # Inside left margin (100)
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx > 0  # Steers right
        assert dvy == 0.0
    
    def test_right_margin_steering(self):
        """Boid near right edge steers left."""
        params = SimulationParams(width=800, height=600, margin=100, turn_factor=0.5)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=750, y=300, vx=0, vy=0)  # Inside right margin (800-100=700)
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx < 0  # Steers left
        assert dvy == 0.0
    
    def test_top_margin_steering(self):
        """Boid near top edge steers down (increasing y)."""
        params = SimulationParams(width=800, height=600, margin=100, turn_factor=0.5)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=400, y=50, vx=0, vy=0)  # Inside top margin
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx == 0.0
        assert dvy > 0  # Steers down (screen coordinates)
    
    def test_bottom_margin_steering(self):
        """Boid near bottom edge steers up (decreasing y)."""
        params = SimulationParams(width=800, height=600, margin=100, turn_factor=0.5)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=400, y=550, vx=0, vy=0)  # Inside bottom margin (600-100=500)
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx == 0.0
        assert dvy < 0  # Steers up (screen coordinates)
    
    def test_corner_steering(self):
        """Boid in corner gets steering in both dimensions."""
        params = SimulationParams(width=800, height=600, margin=100, turn_factor=0.5)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=50, y=50, vx=0, vy=0)  # Top-left corner
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx > 0  # Steers right
        assert dvy > 0  # Steers down
    
    def test_turn_factor_magnitude(self):
        """Boundary steering magnitude matches turn_factor parameter."""
        params = SimulationParams(margin=100, turn_factor=0.8)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=50, y=300, vx=0, vy=0)
        
        dvx, dvy = flock.apply_boundary_steering(boid)
        
        assert dvx == 0.8


class TestSpeedLimits:
    """Tests for speed enforcement."""
    
    def test_speed_within_limits_unchanged(self):
        """Speed within limits is not modified."""
        params = SimulationParams(min_speed=2.0, max_speed=6.0)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=100, y=100, vx=3.0, vy=4.0)  # Speed = 5
        original_vx, original_vy = boid.vx, boid.vy
        
        flock.enforce_speed_limits(boid)
        
        assert boid.vx == original_vx
        assert boid.vy == original_vy
    
    def test_speed_above_max_clamped(self):
        """Speed above max_speed is clamped."""
        params = SimulationParams(max_speed=6.0)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=100, y=100, vx=8.0, vy=6.0)  # Speed = 10
        
        flock.enforce_speed_limits(boid)
        
        assert boid.speed == pytest.approx(6.0)
    
    def test_speed_below_min_boosted(self):
        """Speed below min_speed is boosted."""
        params = SimulationParams(min_speed=2.0)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=100, y=100, vx=0.6, vy=0.8)  # Speed = 1
        
        flock.enforce_speed_limits(boid)
        
        assert boid.speed == pytest.approx(2.0)
    
    def test_direction_preserved_when_clamped(self):
        """Velocity direction is preserved when speed is clamped."""
        params = SimulationParams(max_speed=5.0)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=100, y=100, vx=8.0, vy=6.0)  # Speed = 10, ratio 4:3
        
        flock.enforce_speed_limits(boid)
        
        # Direction should still be 4:3 ratio
        assert boid.vx / boid.vy == pytest.approx(8.0 / 6.0)
    
    def test_zero_speed_handled(self):
        """Zero speed gets random direction at min_speed."""
        np.random.seed(42)
        params = SimulationParams(min_speed=2.0)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        flock.enforce_speed_limits(boid)
        
        assert boid.speed == pytest.approx(2.0)


class TestUpdateBoid:
    """Tests for single boid update."""
    
    def test_isolated_boid_maintains_velocity(self):
        """Isolated boid in center maintains its velocity direction."""
        params = SimulationParams(
            width=800, height=600, margin=100,
            min_speed=2.0, max_speed=6.0
        )
        flock = Flock(num_boids=0, params=params)
        # Place boid in center with valid speed
        boid = Boid(x=400, y=300, vx=3.0, vy=4.0)  # Speed = 5
        flock.boids = [boid]
        
        original_speed = boid.speed
        flock.update_boid(boid)
        
        # Speed should be preserved (within limits)
        assert boid.speed == pytest.approx(original_speed)
    
    def test_position_updates_by_velocity(self):
        """Position updates by velocity after update."""
        params = SimulationParams(width=800, height=600, margin=100)
        flock = Flock(num_boids=0, params=params)
        boid = Boid(x=400, y=300, vx=3.0, vy=4.0)
        flock.boids = [boid]
        
        flock.update_boid(boid)
        
        # Position should have moved (approximately by velocity, 
        # though rules may modify velocity slightly first)
        assert boid.x != 400 or boid.y != 300
    
    def test_boid_near_edge_steered_inward(self):
        """Boid near edge is steered back inward."""
        params = SimulationParams(
            width=800, height=600, margin=100, turn_factor=0.5,
            min_speed=2.0, max_speed=6.0
        )
        flock = Flock(num_boids=0, params=params)
        # Boid moving toward left edge
        boid = Boid(x=50, y=300, vx=-3.0, vy=0)
        flock.boids = [boid]
        
        initial_vx = boid.vx
        flock.update_boid(boid)
        
        # Velocity should have been adjusted rightward
        assert boid.vx > initial_vx


class TestFlockUpdate:
    """Tests for full flock update."""
    
    def test_all_boids_updated(self):
        """All boids have their positions updated."""
        np.random.seed(42)
        flock = Flock(num_boids=10)
        
        initial_positions = [(b.x, b.y) for b in flock.boids]
        flock.update()
        final_positions = [(b.x, b.y) for b in flock.boids]
        
        # All positions should have changed
        for initial, final in zip(initial_positions, final_positions):
            assert initial != final
    
    def test_simulation_runs_multiple_steps(self):
        """Simulation can run for multiple time steps without error."""
        np.random.seed(42)
        flock = Flock(num_boids=20)
        
        for _ in range(100):
            flock.update()
        
        # Boids should still be reasonably near the screen
        # (boundary handling prevents escape)
        for boid in flock.boids:
            assert -100 < boid.x < 900  # Some margin for overshooting
            assert -100 < boid.y < 700


class TestFlockHelpers:
    """Tests for helper methods."""
    
    def test_get_positions(self):
        """get_positions returns correct array."""
        flock = Flock(num_boids=0)
        flock.boids = [
            Boid(x=100, y=200, vx=1, vy=2),
            Boid(x=300, y=400, vx=3, vy=4)
        ]
        
        positions = flock.get_positions()
        
        assert positions.shape == (2, 2)
        assert positions[0, 0] == 100
        assert positions[0, 1] == 200
        assert positions[1, 0] == 300
        assert positions[1, 1] == 400
    
    def test_get_velocities(self):
        """get_velocities returns correct array."""
        flock = Flock(num_boids=0)
        flock.boids = [
            Boid(x=100, y=200, vx=1, vy=2),
            Boid(x=300, y=400, vx=3, vy=4)
        ]
        
        velocities = flock.get_velocities()
        
        assert velocities.shape == (2, 2)
        assert velocities[0, 0] == 1
        assert velocities[0, 1] == 2
        assert velocities[1, 0] == 3
        assert velocities[1, 1] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])