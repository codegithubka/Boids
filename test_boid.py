"""
Unit tests for the Boid class.
"""

import pytest
import numpy as np
from boid import Boid


class TestBoidInstantiation:
    """Tests for Boid creation and attribute access."""
    
    def test_explicit_instantiation(self):
        """Boid can be created with explicit position and velocity."""
        boid = Boid(x=100, y=200, vx=3.0, vy=-4.0)
        
        assert boid.x == 100
        assert boid.y == 200
        assert boid.vx == 3.0
        assert boid.vy == -4.0
    
    def test_attributes_are_mutable(self):
        """Boid attributes can be modified after creation."""
        boid = Boid(x=0, y=0, vx=0, vy=0)
        
        boid.x = 50
        boid.y = 75
        boid.vx = 2.0
        boid.vy = 3.0
        
        assert boid.x == 50
        assert boid.y == 75
        assert boid.vx == 2.0
        assert boid.vy == 3.0
    
    def test_zero_velocity(self):
        """Boid can have zero velocity."""
        boid = Boid(x=100, y=100, vx=0, vy=0)
        
        assert boid.vx == 0
        assert boid.vy == 0
        assert boid.speed == 0


class TestBoidRandomCreation:
    """Tests for random boid generation."""
    
    def test_random_position_within_bounds(self):
        """Random boid position is within specified bounds."""
        np.random.seed(42)  # Reproducibility
        
        width, height = 800, 600
        
        for _ in range(100):
            boid = Boid.create_random(width=width, height=height)
            
            assert 0 <= boid.x <= width
            assert 0 <= boid.y <= height
    
    def test_random_velocity_within_speed_limit(self):
        """Random boid speed is within specified limits."""
        np.random.seed(42)
        
        max_speed = 6.0
        
        for _ in range(100):
            boid = Boid.create_random(max_speed=max_speed)
            
            assert boid.speed <= max_speed
            assert boid.speed >= max_speed / 2  # Minimum is half max
    
    def test_random_boids_are_different(self):
        """Multiple random boids have different positions."""
        np.random.seed(42)
        
        boid1 = Boid.create_random()
        boid2 = Boid.create_random()
        
        # Extremely unlikely to be identical
        assert boid1.x != boid2.x or boid1.y != boid2.y


class TestBoidProperties:
    """Tests for computed properties."""
    
    def test_speed_calculation(self):
        """Speed is calculated correctly from velocity components."""
        # Classic 3-4-5 triangle
        boid = Boid(x=0, y=0, vx=3.0, vy=4.0)
        
        assert boid.speed == 5.0
    
    def test_speed_with_negative_velocity(self):
        """Speed is positive regardless of velocity direction."""
        boid = Boid(x=0, y=0, vx=-3.0, vy=-4.0)
        
        assert boid.speed == 5.0
    
    def test_position_property(self):
        """Position property returns numpy array."""
        boid = Boid(x=100, y=200, vx=0, vy=0)
        
        pos = boid.position
        
        assert isinstance(pos, np.ndarray)
        assert len(pos) == 2
        assert pos[0] == 100
        assert pos[1] == 200
    
    def test_velocity_property(self):
        """Velocity property returns numpy array."""
        boid = Boid(x=0, y=0, vx=3.0, vy=-4.0)
        
        vel = boid.velocity
        
        assert isinstance(vel, np.ndarray)
        assert len(vel) == 2
        assert vel[0] == 3.0
        assert vel[1] == -4.0


class TestBoidEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_negative_position(self):
        """Boid can have negative position (before boundary correction)."""
        boid = Boid(x=-10, y=-20, vx=1, vy=1)
        
        assert boid.x == -10
        assert boid.y == -20
    
    def test_very_small_velocity(self):
        """Boid handles very small velocities without numerical issues."""
        boid = Boid(x=0, y=0, vx=1e-10, vy=1e-10)
        
        assert boid.speed == pytest.approx(np.sqrt(2) * 1e-10)
    
    def test_large_values(self):
        """Boid handles large position/velocity values."""
        boid = Boid(x=1e6, y=1e6, vx=1000, vy=1000)
        
        assert boid.x == 1e6
        assert boid.speed == pytest.approx(np.sqrt(2) * 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])