"""
Boid class for flocking simulation.

A Boid represents a single bird-oid agent with position and velocity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Boid:
    """
    A single boid agent in the flocking simulation.
    
    Attributes:
        x: horizontal position (pixels)
        y: vertical position (pixels)
        vx: horizontal velocity (pixels/frame)
        vy: vertical velocity (pixels/frame)
    """
    x: float
    y: float
    vx: float
    vy: float
    
    @classmethod
    def create_random(
        cls,
        width: float = 800,
        height: float = 600,
        max_speed: float = 6.0
    ) -> "Boid":
        """
        Factory method to create a boid with random position and velocity.
        
        Args:
            width: simulation width in pixels
            height: simulation height in pixels
            max_speed: maximum initial speed magnitude
            
        Returns:
            A new Boid with random position within bounds and random velocity.
        """
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        # Random angle for velocity direction
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(max_speed / 2, max_speed)
        
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        return cls(x=x, y=y, vx=vx, vy=vy)
    
    @property
    def speed(self) -> float:
        """Calculate current speed magnitude."""
        return np.sqrt(self.vx**2 + self.vy**2)
    
    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array([self.x, self.y])
    
    @property
    def velocity(self) -> np.ndarray:
        """Return velocity as numpy array."""
        return np.array([self.vx, self.vy])