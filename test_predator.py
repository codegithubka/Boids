"""
Unit tests for the Predator class and predator avoidance rule.
"""

import pytest
import numpy as np
from boid import Boid
from predator import Predator
from rules import compute_predator_avoidance
from rules_optimized import (
    FlockState,
    compute_predator_avoidance_kdtree,
    compute_all_rules_with_predator_kdtree
)


class TestPredatorInstantiation:
    """Tests for Predator creation."""
    
    def test_explicit_instantiation(self):
        """Predator can be created with explicit values."""
        predator = Predator(x=100, y=200, vx=1.5, vy=-1.5)
        
        assert predator.x == 100
        assert predator.y == 200
        assert predator.vx == 1.5
        assert predator.vy == -1.5
    
    def test_create_at_position(self):
        """Predator can be created at specific position."""
        np.random.seed(42)
        predator = Predator.create_at_position(x=400, y=300, speed=2.5)
        
        assert predator.x == 400
        assert predator.y == 300
        assert predator.speed == pytest.approx(2.5)
    
    def test_create_random(self):
        """Predator can be created at random position."""
        np.random.seed(42)
        predator = Predator.create_random(width=800, height=600, speed=2.5)
        
        assert 0 <= predator.x <= 800
        assert 0 <= predator.y <= 600
        assert predator.speed == pytest.approx(2.5)
    
    def test_attributes_are_mutable(self):
        """Predator attributes can be modified."""
        predator = Predator(x=0, y=0, vx=0, vy=0)
        
        predator.x = 100
        predator.y = 200
        predator.vx = 1.0
        predator.vy = 2.0
        
        assert predator.x == 100
        assert predator.y == 200
        assert predator.vx == 1.0
        assert predator.vy == 2.0


class TestPredatorProperties:
    """Tests for Predator computed properties."""
    
    def test_speed_calculation(self):
        """Speed is calculated correctly."""
        predator = Predator(x=0, y=0, vx=3.0, vy=4.0)
        
        assert predator.speed == 5.0
    
    def test_position_property(self):
        """Position property returns numpy array."""
        predator = Predator(x=100, y=200, vx=0, vy=0)
        
        pos = predator.position
        
        assert isinstance(pos, np.ndarray)
        assert pos[0] == 100
        assert pos[1] == 200
    
    def test_velocity_property(self):
        """Velocity property returns numpy array."""
        predator = Predator(x=0, y=0, vx=1.5, vy=-2.5)
        
        vel = predator.velocity
        
        assert isinstance(vel, np.ndarray)
        assert vel[0] == 1.5
        assert vel[1] == -2.5


class TestPredatorFlockTracking:
    """Tests for predator's flock tracking abilities."""
    
    def test_compute_flock_center_empty(self):
        """Flock center returns None for empty flock."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        
        center = predator.compute_flock_center([])
        
        assert center is None
    
    def test_compute_flock_center_single_boid(self):
        """Flock center is boid position for single boid."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        boid = Boid(x=200, y=300, vx=0, vy=0)
        
        center = predator.compute_flock_center([boid])
        
        assert center[0] == 200
        assert center[1] == 300
    
    def test_compute_flock_center_multiple_boids(self):
        """Flock center is average of all boid positions."""
        predator = Predator(x=0, y=0, vx=0, vy=0)
        boids = [
            Boid(x=100, y=100, vx=0, vy=0),
            Boid(x=200, y=100, vx=0, vy=0),
            Boid(x=150, y=200, vx=0, vy=0),
        ]
        
        center = predator.compute_flock_center(boids)
        
        # Center: (100+200+150)/3 = 150, (100+100+200)/3 = 133.33
        assert center[0] == pytest.approx(150.0)
        assert center[1] == pytest.approx(133.33, rel=0.01)
    
    def test_compute_nearest_boid_empty(self):
        """Nearest boid returns None for empty flock."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        
        nearest = predator.compute_nearest_boid([])
        
        assert nearest is None
    
    def test_compute_nearest_boid_single(self):
        """Nearest boid returns the only boid."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        boid = Boid(x=200, y=200, vx=0, vy=0)
        
        nearest = predator.compute_nearest_boid([boid])
        
        assert nearest is boid
    
    def test_compute_nearest_boid_multiple(self):
        """Nearest boid returns closest boid."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        far_boid = Boid(x=500, y=500, vx=0, vy=0)
        close_boid = Boid(x=110, y=110, vx=0, vy=0)
        medium_boid = Boid(x=200, y=200, vx=0, vy=0)
        
        nearest = predator.compute_nearest_boid([far_boid, close_boid, medium_boid])
        
        assert nearest is close_boid


class TestPredatorSteering:
    """Tests for predator steering behavior."""
    
    def test_steer_toward_target(self):
        """Predator steers toward target position."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        target = np.array([200, 150])
        
        dvx, dvy = predator.steer_toward(target, hunting_strength=0.1)
        
        # dx = 200-100 = 100, dy = 150-100 = 50
        assert dvx == 10.0  # 100 * 0.1
        assert dvy == 5.0   # 50 * 0.1
    
    def test_steer_toward_target_behind(self):
        """Predator steers backward if target is behind."""
        predator = Predator(x=200, y=200, vx=0, vy=0)
        target = np.array([100, 100])
        
        dvx, dvy = predator.steer_toward(target, hunting_strength=0.1)
        
        assert dvx == -10.0
        assert dvy == -10.0
    
    def test_update_velocity_toward_center(self):
        """Predator velocity updates toward flock center."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        boids = [
            Boid(x=200, y=200, vx=0, vy=0),
            Boid(x=300, y=200, vx=0, vy=0),
        ]
        
        predator.update_velocity_toward_center(boids, hunting_strength=0.1)
        
        # Center is (250, 200), dx = 150, dy = 100
        assert predator.vx == 15.0
        assert predator.vy == 10.0
    
    def test_update_velocity_toward_center_empty_flock(self):
        """Predator velocity unchanged for empty flock."""
        predator = Predator(x=100, y=100, vx=1.0, vy=2.0)
        
        predator.update_velocity_toward_center([], hunting_strength=0.1)
        
        assert predator.vx == 1.0
        assert predator.vy == 2.0
    
    def test_update_velocity_toward_nearest(self):
        """Predator velocity updates toward nearest boid."""
        predator = Predator(x=100, y=100, vx=0, vy=0)
        close_boid = Boid(x=150, y=100, vx=0, vy=0)
        far_boid = Boid(x=500, y=500, vx=0, vy=0)
        
        predator.update_velocity_toward_nearest([far_boid, close_boid], hunting_strength=0.1)
        
        # Nearest is close_boid at (150, 100), dx = 50, dy = 0
        assert predator.vx == 5.0
        assert predator.vy == 0.0


class TestPredatorBoundaryHandling:
    """Tests for predator boundary behavior."""
    
    def test_no_steering_in_center(self):
        """Predator in center has no boundary steering."""
        predator = Predator(x=400, y=300, vx=1.0, vy=1.0)
        
        predator.apply_boundary_steering(
            width=800, height=600, margin=100, turn_factor=0.5
        )
        
        assert predator.vx == 1.0
        assert predator.vy == 1.0
    
    def test_left_margin_steering(self):
        """Predator near left edge steers right."""
        predator = Predator(x=50, y=300, vx=0, vy=0)
        
        predator.apply_boundary_steering(
            width=800, height=600, margin=100, turn_factor=0.5
        )
        
        assert predator.vx == 0.5
        assert predator.vy == 0.0
    
    def test_corner_steering(self):
        """Predator in corner steers in both dimensions."""
        predator = Predator(x=50, y=50, vx=0, vy=0)
        
        predator.apply_boundary_steering(
            width=800, height=600, margin=100, turn_factor=0.5
        )
        
        assert predator.vx == 0.5
        assert predator.vy == 0.5


class TestPredatorSpeedLimits:
    """Tests for predator speed enforcement."""
    
    def test_speed_within_limits_unchanged(self):
        """Speed within limits is not modified."""
        predator = Predator(x=0, y=0, vx=1.5, vy=2.0)
        original_vx, original_vy = predator.vx, predator.vy
        
        predator.enforce_speed_limits(max_speed=5.0, min_speed=1.0)
        
        assert predator.vx == original_vx
        assert predator.vy == original_vy
    
    def test_speed_above_max_clamped(self):
        """Speed above max is clamped."""
        predator = Predator(x=0, y=0, vx=8.0, vy=6.0)  # Speed = 10
        
        predator.enforce_speed_limits(max_speed=5.0, min_speed=1.0)
        
        assert predator.speed == pytest.approx(5.0)
    
    def test_speed_below_min_boosted(self):
        """Speed below min is boosted."""
        predator = Predator(x=0, y=0, vx=0.3, vy=0.4)  # Speed = 0.5
        
        predator.enforce_speed_limits(max_speed=5.0, min_speed=2.0)
        
        assert predator.speed == pytest.approx(2.0)
    
    def test_zero_speed_handled(self):
        """Zero speed gets random direction."""
        np.random.seed(42)
        predator = Predator(x=0, y=0, vx=0, vy=0)
        
        predator.enforce_speed_limits(max_speed=5.0, min_speed=2.0)
        
        assert predator.speed == pytest.approx(2.0)


class TestPredatorPositionUpdate:
    """Tests for predator position updates."""
    
    def test_position_updates_by_velocity(self):
        """Position updates correctly."""
        predator = Predator(x=100, y=100, vx=3.0, vy=-2.0)
        
        predator.update_position()
        
        assert predator.x == 103.0
        assert predator.y == 98.0
    
    def test_multiple_updates(self):
        """Multiple updates accumulate correctly."""
        predator = Predator(x=100, y=100, vx=1.0, vy=1.0)
        
        for _ in range(10):
            predator.update_position()
        
        assert predator.x == 110.0
        assert predator.y == 110.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPredatorAvoidanceRule:
    """Tests for compute_predator_avoidance function."""
    
    def test_predator_outside_detection_range(self):
        """No avoidance when predator is outside detection range."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid,
            predator_x=500, predator_y=500,  # Far away
            detection_range=100,
            avoidance_strength=0.5
        )
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_predator_at_edge_of_detection_range(self):
        """Minimal avoidance at edge of detection range."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid,
            predator_x=199, predator_y=100,  # Just inside detection range
            detection_range=100,
            avoidance_strength=0.5
        )
        
        # Should have small negative dvx (flee left from predator on right)
        assert dvx < 0
        assert abs(dvy) < abs(dvx)  # Mostly horizontal
    
    def test_predator_very_close(self):
        """Strong avoidance when predator is very close."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid,
            predator_x=110, predator_y=100,  # 10px away
            detection_range=100,
            avoidance_strength=0.5
        )
        
        # Should have strong negative dvx (flee left)
        assert dvx < -10  # Significant avoidance
        assert abs(dvy) < 0.01  # Purely horizontal
    
    def test_avoidance_direction_correct(self):
        """Boid flees in opposite direction from predator."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        # Predator to the right
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=150, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        assert dvx < 0  # Flee left
        
        # Predator below
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=100, predator_y=150,
            detection_range=100, avoidance_strength=0.5
        )
        assert dvy < 0  # Flee up
        
        # Predator to the left
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=50, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        assert dvx > 0  # Flee right
    
    def test_avoidance_scales_with_strength(self):
        """Stronger avoidance_strength produces larger adjustment."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        weak_dvx, _ = compute_predator_avoidance(
            boid, predator_x=150, predator_y=100,
            detection_range=100, avoidance_strength=0.1
        )
        
        strong_dvx, _ = compute_predator_avoidance(
            boid, predator_x=150, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        
        assert abs(strong_dvx) > abs(weak_dvx)
    
    def test_predator_at_same_position_handled(self):
        """Edge case: predator at exact same position as boid."""
        np.random.seed(42)
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=100, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        
        # Should return non-zero (random direction)
        magnitude = (dvx**2 + dvy**2) ** 0.5
        assert magnitude > 0


class TestPredatorAvoidanceKDTree:
    """Tests for KDTree-based predator avoidance."""
    
    def create_test_flock_state(self, boids):
        """Helper to create FlockState from boid list."""
        return FlockState(boids)
    
    def test_kdtree_matches_naive(self):
        """KDTree avoidance produces same results as naive."""
        boids = [
            Boid(x=100, y=100, vx=1, vy=0),
            Boid(x=200, y=200, vx=0, vy=1),
            Boid(x=300, y=150, vx=-1, vy=0),
        ]
        state = self.create_test_flock_state(boids)
        
        predator_x, predator_y = 150, 120
        detection_range = 100
        avoidance_strength = 0.5
        
        for i, boid in enumerate(boids):
            naive_dv = compute_predator_avoidance(
                boid, predator_x, predator_y,
                detection_range, avoidance_strength
            )
            kdtree_dv = compute_predator_avoidance_kdtree(
                i, state, predator_x, predator_y,
                detection_range, avoidance_strength
            )
            
            assert naive_dv[0] == pytest.approx(kdtree_dv[0], abs=1e-10)
            assert naive_dv[1] == pytest.approx(kdtree_dv[1], abs=1e-10)
    
    def test_combined_rules_with_predator(self):
        """Combined rules include predator avoidance."""
        boids = [
            Boid(x=100, y=100, vx=1, vy=0),
            Boid(x=120, y=100, vx=1, vy=0),  # Neighbor
        ]
        state = self.create_test_flock_state(boids)
        
        # Without predator
        dv_no_pred = compute_all_rules_with_predator_kdtree(
            boid_index=0, flock_state=state,
            visual_range=50, protected_range=12,
            cohesion_factor=0.002, alignment_factor=0.06,
            separation_strength=0.15,
            predator_x=None, predator_y=None,
            predator_detection_range=100, predator_avoidance_strength=0.5
        )
        
        # With predator nearby
        dv_with_pred = compute_all_rules_with_predator_kdtree(
            boid_index=0, flock_state=state,
            visual_range=50, protected_range=12,
            cohesion_factor=0.002, alignment_factor=0.06,
            separation_strength=0.15,
            predator_x=150, predator_y=100,
            predator_detection_range=100, predator_avoidance_strength=0.5
        )
        
        # With predator, boid should have different (more negative x) velocity
        assert dv_with_pred[0] < dv_no_pred[0]


class TestPredatorAvoidanceEdgeCases:
    """Tests for edge cases in predator avoidance."""
    
    def test_single_boid_vs_predator(self):
        """Single boid flees directly away from predator."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=150, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        
        # Should flee directly left (negative x, zero y)
        assert dvx < 0
        assert abs(dvy) < 0.01
    
    def test_diagonal_predator(self):
        """Boid flees diagonally from diagonal predator."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=150, predator_y=150,
            detection_range=100, avoidance_strength=0.5
        )
        
        # Should flee up-left (negative x, negative y)
        assert dvx < 0
        assert dvy < 0
        # Should be roughly equal magnitude
        assert abs(dvx) == pytest.approx(abs(dvy), rel=0.01)
    
    def test_predator_exactly_at_detection_boundary(self):
        """Predator exactly at detection range boundary."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        # Predator exactly 100px away (detection_range=100)
        dvx, dvy = compute_predator_avoidance(
            boid, predator_x=200, predator_y=100,
            detection_range=100, avoidance_strength=0.5
        )
        
        # At exactly the boundary, should return zero
        assert dvx == 0.0
        assert dvy == 0.0