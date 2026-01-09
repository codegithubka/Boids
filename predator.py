"""
Predator class for the Boids simulation.

A predator is an antagonistic agent that hunts the flock,
causing boids to scatter and avoid it.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from boid import Boid


@dataclass
class Predator:
    """
    A predator agent that hunts boids.
    
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
    def create_at_position(
        cls,
        x: float,
        y: float,
        speed: float = 2.5
    ) -> "Predator":
        """
        Factory method to create a predator at a specific position.
        
        Args:
            x: horizontal position
            y: vertical position
            speed: initial speed magnitude
            
        Returns:
            A new Predator with random velocity direction.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        
        return cls(x=x, y=y, vx=vx, vy=vy)
    
    @classmethod
    def create_random(
        cls,
        width: float = 800,
        height: float = 600,
        speed: float = 2.5
    ) -> "Predator":
        """
        Factory method to create a predator at random position.
        
        Args:
            width: simulation width in pixels
            height: simulation height in pixels
            speed: initial speed magnitude
            
        Returns:
            A new Predator with random position and velocity.
        """
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        
        return cls.create_at_position(x, y, speed)
    
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
    
    def compute_flock_center(self, boids: List["Boid"]) -> Optional[np.ndarray]:
        """
        Compute center of mass of the flock.
        
        Args:
            boids: List of all boids
            
        Returns:
            numpy array [x, y] of flock center, or None if no boids
        """
        if not boids:
            return None
        
        sum_x = sum(b.x for b in boids)
        sum_y = sum(b.y for b in boids)
        n = len(boids)
        
        return np.array([sum_x / n, sum_y / n])
    
    def compute_nearest_boid(self, boids: List["Boid"]) -> Optional["Boid"]:
        """
        Find the nearest boid to the predator.
        
        Args:
            boids: List of all boids
            
        Returns:
            The nearest Boid, or None if no boids
        """
        if not boids:
            return None
        
        nearest = None
        min_dist_sq = float('inf')
        
        for boid in boids:
            dx = self.x - boid.x
            dy = self.y - boid.y
            dist_sq = dx * dx + dy * dy
            
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = boid
        
        return nearest
    
    def steer_toward(
        self,
        target: np.ndarray,
        hunting_strength: float = 0.05
    ) -> tuple:
        """
        Compute steering adjustment toward a target position.
        
        Args:
            target: numpy array [x, y] of target position
            hunting_strength: multiplier for steering force
            
        Returns:
            Tuple (dvx, dvy) — velocity adjustment
        """
        dx = target[0] - self.x
        dy = target[1] - self.y
        
        return (dx * hunting_strength, dy * hunting_strength)
    
    def update_velocity_toward_center(
        self,
        boids: List["Boid"],
        hunting_strength: float = 0.05
    ) -> None:
        """
        Adjust velocity to move toward flock center of mass.
        
        Args:
            boids: List of all boids
            hunting_strength: multiplier for steering force
        """
        center = self.compute_flock_center(boids)
        
        if center is None:
            return
        
        dvx, dvy = self.steer_toward(center, hunting_strength)
        self.vx += dvx
        self.vy += dvy
    
    def update_velocity_toward_nearest(
        self,
        boids: List["Boid"],
        hunting_strength: float = 0.05
    ) -> None:
        """
        Adjust velocity to move toward nearest boid.
        
        Args:
            boids: List of all boids
            hunting_strength: multiplier for steering force
        """
        nearest = self.compute_nearest_boid(boids)
        
        if nearest is None:
            return
        
        target = np.array([nearest.x, nearest.y])
        dvx, dvy = self.steer_toward(target, hunting_strength)
        self.vx += dvx
        self.vy += dvy
    
    def apply_boundary_steering(
        self,
        width: float,
        height: float,
        margin: float,
        turn_factor: float
    ) -> None:
        """
        Apply boundary steering to keep predator in bounds.
        
        Uses same logic as boids for consistency.
        
        Args:
            width: simulation width
            height: simulation height
            margin: distance from edge to start turning
            turn_factor: steering strength at boundaries
        """
        if self.x < margin:
            self.vx += turn_factor
        if self.x > width - margin:
            self.vx -= turn_factor
        if self.y < margin:
            self.vy += turn_factor
        if self.y > height - margin:
            self.vy -= turn_factor
    
    def enforce_speed_limits(
        self,
        max_speed: float,
        min_speed: float
    ) -> None:
        """
        Clamp predator speed to within bounds.
        
        Args:
            max_speed: maximum speed
            min_speed: minimum speed
        """
        speed = self.speed
        
        if speed == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            self.vx = min_speed * np.cos(angle)
            self.vy = min_speed * np.sin(angle)
            return
        
        if speed > max_speed:
            self.vx = (self.vx / speed) * max_speed
            self.vy = (self.vy / speed) * max_speed
        elif speed < min_speed:
            self.vx = (self.vx / speed) * min_speed
            self.vy = (self.vy / speed) * min_speed
    
    def update_position(self) -> None:
        """Update position based on current velocity."""
        self.x += self.vx
        self.y += self.vy