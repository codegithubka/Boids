"""
Unit tests for flocking rules.
"""

import pytest
import math
from boid import Boid
from rules import compute_separation, compute_alignment, compute_cohesion


class TestSeparation:
    """Tests for the separation rule."""
    
    def test_no_neighbors(self):
        """With no other boids, separation returns zero."""
        boid = Boid(x=100, y=100, vx=1, vy=0)
        all_boids = [boid]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=0.05)
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_neighbor_outside_protected_range(self):
        """Boids outside protected range don't cause separation."""
        boid = Boid(x=100, y=100, vx=1, vy=0)
        other = Boid(x=150, y=100, vx=1, vy=0)  # 50 pixels away
        all_boids = [boid, other]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=0.05)
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_single_neighbor_in_range(self):
        """Single neighbor in protected range causes repulsion."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=105, y=100, vx=0, vy=0)  # 5 pixels to the right
        all_boids = [boid, other]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=1.0)
        
        # Boid should be pushed left (negative x direction)
        # dx = 100 - 105 = -5, so repel_x = -5, dvx = -5 * 1.0 = -5
        assert dvx == -5.0
        assert dvy == 0.0
    
    def test_repulsion_direction(self):
        """Repulsion pushes boid away from neighbor."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=95, y=100, vx=0, vy=0)  # 5 pixels to the left
        all_boids = [boid, other]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=1.0)
        
        # Boid should be pushed right (positive x direction)
        assert dvx == 5.0
        assert dvy == 0.0
    
    def test_multiple_neighbors(self):
        """Multiple neighbors in range accumulate repulsion forces."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        left = Boid(x=95, y=100, vx=0, vy=0)   # 5 left
        right = Boid(x=105, y=100, vx=0, vy=0) # 5 right
        all_boids = [boid, left, right]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=1.0)
        
        # Forces should cancel: (100-95) + (100-105) = 5 + (-5) = 0
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_asymmetric_neighbors(self):
        """Asymmetric neighbor positions create net force."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        close = Boid(x=98, y=100, vx=0, vy=0)  # 2 pixels left
        far = Boid(x=108, y=100, vx=0, vy=0)   # 8 pixels right
        all_boids = [boid, close, far]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=1.0)
        
        # (100-98) + (100-108) = 2 + (-8) = -6
        assert dvx == -6.0
        assert dvy == 0.0
    
    def test_strength_factor(self):
        """Strength factor scales the repulsion."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=110, y=100, vx=0, vy=0)  # 10 pixels right
        all_boids = [boid, other]
        
        dvx1, _ = compute_separation(boid, all_boids, protected_range=20, strength=0.1)
        dvx2, _ = compute_separation(boid, all_boids, protected_range=20, strength=0.5)
        
        # Ratio should match strength ratio
        assert dvx2 / dvx1 == pytest.approx(5.0)
    
    def test_diagonal_neighbor(self):
        """Diagonal neighbor creates force in both dimensions."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=103, y=104, vx=0, vy=0)  # 3 right, 4 down (5 total distance)
        all_boids = [boid, other]
        
        dvx, dvy = compute_separation(boid, all_boids, protected_range=20, strength=1.0)
        
        assert dvx == -3.0
        assert dvy == -4.0


class TestAlignment:
    """Tests for the alignment rule."""
    
    def test_no_neighbors(self):
        """With no visible neighbors, alignment returns zero."""
        boid = Boid(x=100, y=100, vx=1, vy=0)
        all_boids = [boid]
        
        dvx, dvy = compute_alignment(
            boid, all_boids, 
            visual_range=75, protected_range=12, matching_factor=0.05
        )
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_neighbor_in_protected_range_excluded(self):
        """Neighbors in protected range don't contribute to alignment."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        close = Boid(x=105, y=100, vx=10, vy=0)  # 5 pixels away (in protected range)
        all_boids = [boid, close]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=1.0
        )
        
        # Should return zero because neighbor is in protected range
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_neighbor_outside_visual_range_excluded(self):
        """Neighbors outside visual range don't contribute to alignment."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        far = Boid(x=200, y=100, vx=10, vy=0)  # 100 pixels away
        all_boids = [boid, far]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=1.0
        )
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_single_neighbor_same_velocity(self):
        """When neighbor has same velocity, no adjustment needed."""
        boid = Boid(x=100, y=100, vx=5, vy=3)
        other = Boid(x=130, y=100, vx=5, vy=3)  # 30 pixels away, same velocity
        all_boids = [boid, other]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=1.0
        )
        
        assert dvx == pytest.approx(0.0)
        assert dvy == pytest.approx(0.0)
    
    def test_single_neighbor_different_velocity(self):
        """Boid adjusts toward neighbor's velocity."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=130, y=100, vx=10, vy=0)  # Moving right
        all_boids = [boid, other]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=1.0
        )
        
        # Should fully match neighbor velocity with factor=1.0
        assert dvx == 10.0
        assert dvy == 0.0
    
    def test_matching_factor(self):
        """Matching factor scales the velocity adjustment."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=130, y=100, vx=10, vy=0)
        all_boids = [boid, other]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=0.5
        )
        
        assert dvx == 5.0  # Half of difference
        assert dvy == 0.0
    
    def test_opposing_velocities_average(self):
        """With opposing neighbors, alignment tends toward average (zero)."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        left_mover = Boid(x=80, y=100, vx=-10, vy=0)   # Moving left
        right_mover = Boid(x=120, y=100, vx=10, vy=0)  # Moving right
        all_boids = [boid, left_mover, right_mover]
        
        dvx, dvy = compute_alignment(
            boid, all_boids,
            visual_range=75, protected_range=12, matching_factor=1.0
        )
        
        # Average velocity is 0, boid velocity is 0, so no adjustment
        assert dvx == pytest.approx(0.0)
        assert dvy == pytest.approx(0.0)


class TestCohesion:
    """Tests for the cohesion rule."""
    
    def test_no_neighbors(self):
        """With no visible neighbors, cohesion returns zero."""
        boid = Boid(x=100, y=100, vx=1, vy=0)
        all_boids = [boid]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=0.005
        )
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_neighbor_in_protected_range_excluded(self):
        """Neighbors in protected range don't contribute to cohesion."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        close = Boid(x=105, y=100, vx=0, vy=0)  # 5 pixels away
        all_boids = [boid, close]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=1.0
        )
        
        assert dvx == 0.0
        assert dvy == 0.0
    
    def test_single_neighbor_attraction(self):
        """Boid is attracted toward single neighbor."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=150, y=100, vx=0, vy=0)  # 50 pixels to the right
        all_boids = [boid, other]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=1.0
        )
        
        # Should steer right toward neighbor
        assert dvx == 50.0
        assert dvy == 0.0
    
    def test_centering_factor(self):
        """Centering factor scales the attraction."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        other = Boid(x=150, y=100, vx=0, vy=0)
        all_boids = [boid, other]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=0.1
        )
        
        assert dvx == 5.0  # 50 * 0.1
        assert dvy == 0.0
    
    def test_center_of_mass_multiple_neighbors(self):
        """Boid steers toward center of mass of multiple neighbors."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        # Three neighbors forming a triangle
        n1 = Boid(x=130, y=100, vx=0, vy=0)  # Right
        n2 = Boid(x=100, y=130, vx=0, vy=0)  # Below
        n3 = Boid(x=130, y=130, vx=0, vy=0)  # Diagonal
        all_boids = [boid, n1, n2, n3]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=1.0
        )
        
        # Center of mass: ((130+100+130)/3, (100+130+130)/3) = (120, 120)
        # Difference from boid: (120-100, 120-100) = (20, 20)
        assert dvx == pytest.approx(20.0)
        assert dvy == pytest.approx(20.0)
    
    def test_equilateral_triangle_centered(self):
        """Boid at center of equilateral triangle has balanced cohesion."""
        # Boid at centroid of triangle should have small/zero net force
        # Triangle with vertices at (100, 100), (140, 100), (120, 100+34.64)
        boid = Boid(x=120, y=100 + 34.64/3, vx=0, vy=0)  # Near centroid
        n1 = Boid(x=100, y=100, vx=0, vy=0)
        n2 = Boid(x=140, y=100, vx=0, vy=0)
        n3 = Boid(x=120, y=100 + 34.64, vx=0, vy=0)
        all_boids = [boid, n1, n2, n3]
        
        dvx, dvy = compute_cohesion(
            boid, all_boids,
            visual_range=75, protected_range=12, centering_factor=1.0
        )
        
        # Should be very close to zero (boid is at centroid)
        assert dvx == pytest.approx(0.0, abs=1e-10)
        assert dvy == pytest.approx(0.0, abs=0.01)  # Small numerical tolerance


class TestRulesInteraction:
    """Tests for how rules should work together."""
    
    def test_protected_range_inside_visual_range(self):
        """Protected range should be smaller than visual range."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        # Neighbor at 20 pixels - inside visual range (75) but outside protected (12)
        neighbor = Boid(x=120, y=100, vx=5, vy=0)
        all_boids = [boid, neighbor]
        
        sep_dv = compute_separation(boid, all_boids, protected_range=12, strength=1.0)
        align_dv = compute_alignment(boid, all_boids, visual_range=75, protected_range=12, matching_factor=1.0)
        cohesion_dv = compute_cohesion(boid, all_boids, visual_range=75, protected_range=12, centering_factor=1.0)
        
        # Separation: not triggered (outside protected range)
        assert sep_dv == (0.0, 0.0)
        
        # Alignment: triggered (in visual range, outside protected)
        assert align_dv == (5.0, 0.0)
        
        # Cohesion: triggered (in visual range, outside protected)
        assert cohesion_dv == (20.0, 0.0)
    
    def test_very_close_neighbor(self):
        """Very close neighbor triggers separation but not alignment/cohesion."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        # Neighbor at 5 pixels - inside protected range
        neighbor = Boid(x=105, y=100, vx=5, vy=0)
        all_boids = [boid, neighbor]
        
        sep_dv = compute_separation(boid, all_boids, protected_range=12, strength=1.0)
        align_dv = compute_alignment(boid, all_boids, visual_range=75, protected_range=12, matching_factor=1.0)
        cohesion_dv = compute_cohesion(boid, all_boids, visual_range=75, protected_range=12, centering_factor=1.0)
        
        # Separation: triggered (inside protected range)
        assert sep_dv == (-5.0, 0.0)
        
        # Alignment: NOT triggered (inside protected range)
        assert align_dv == (0.0, 0.0)
        
        # Cohesion: NOT triggered (inside protected range)
        assert cohesion_dv == (0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])