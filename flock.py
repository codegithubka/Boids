"""
Flock class for managing the Boids simulation.

Handles initialization, update loop, and parameter configuration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from boid import Boid
from rules import compute_separation, compute_alignment, compute_cohesion


@dataclass
class SimulationParams:
    """Configuration parameters for the simulation."""
    
    # Simulation bounds
    width: float = 800
    height: float = 600
    
    # Perception ranges
    visual_range: float = 75
    protected_range: float = 12
    
    # Speed constraints
    max_speed: float = 6.0
    min_speed: float = 2.0
    
    # Rule weights
    cohesion_factor: float = 0.005
    alignment_factor: float = 0.05
    separation_strength: float = 0.05
    
    # Boundary handling
    margin: float = 100
    turn_factor: float = 0.5


class Flock:
    """
    Manages a collection of boids and runs the simulation.
    
    Attributes:
        boids: List of all boids in the flock
        params: Simulation parameters
    """
    
    def __init__(self, num_boids: int, params: SimulationParams = None):
        """
        Initialize the flock with random boids.
        
        Args:
            num_boids: Number of boids to create
            params: Simulation parameters (uses defaults if None)
        """
        self.params = params or SimulationParams()
        self.boids: List[Boid] = []
        
        for _ in range(num_boids):
            boid = Boid.create_random(
                width=self.params.width,
                height=self.params.height,
                max_speed=self.params.max_speed
            )
            self.boids.append(boid)
    
    def apply_boundary_steering(self, boid: Boid) -> tuple:
        """
        Compute velocity adjustment to keep boid within bounds.
        
        Uses soft boundaries: boids are steered away from edges
        when they enter the margin zone.
        
        Args:
            boid: The boid to check
            
        Returns:
            Tuple (dvx, dvy) — velocity adjustment
            
        Note: Uses screen coordinates where (0,0) is top-left,
        y increases downward.
        """
        dvx = 0.0
        dvy = 0.0
        
        p = self.params
        
        # Left margin
        if boid.x < p.margin:
            dvx += p.turn_factor
        
        # Right margin
        if boid.x > p.width - p.margin:
            dvx -= p.turn_factor
        
        # Top margin (small y values)
        if boid.y < p.margin:
            dvy += p.turn_factor
        
        # Bottom margin (large y values)
        if boid.y > p.height - p.margin:
            dvy -= p.turn_factor
        
        return (dvx, dvy)
    
    def enforce_speed_limits(self, boid: Boid) -> None:
        """
        Clamp boid speed to within min/max bounds.
        
        Modifies boid velocity in-place to maintain direction
        while constraining speed magnitude.
        
        Args:
            boid: The boid to constrain
        """
        speed = boid.speed
        
        if speed == 0:
            # Avoid division by zero; give random direction at min speed
            angle = np.random.uniform(0, 2 * np.pi)
            boid.vx = self.params.min_speed * np.cos(angle)
            boid.vy = self.params.min_speed * np.sin(angle)
            return
        
        if speed > self.params.max_speed:
            boid.vx = (boid.vx / speed) * self.params.max_speed
            boid.vy = (boid.vy / speed) * self.params.max_speed
        elif speed < self.params.min_speed:
            boid.vx = (boid.vx / speed) * self.params.min_speed
            boid.vy = (boid.vy / speed) * self.params.min_speed
    
    def update_boid(self, boid: Boid) -> None:
        """
        Apply all rules and update a single boid's state.
        
        This implements the combined update loop from Phase 1:
        1. Compute separation, alignment, cohesion adjustments
        2. Apply boundary steering
        3. Update velocity
        4. Enforce speed limits
        5. Update position
        
        Args:
            boid: The boid to update (modified in-place)
        """
        p = self.params
        
        # Compute rule contributions
        sep_dv = compute_separation(
            boid, self.boids,
            protected_range=p.protected_range,
            strength=p.separation_strength
        )
        
        align_dv = compute_alignment(
            boid, self.boids,
            visual_range=p.visual_range,
            protected_range=p.protected_range,
            matching_factor=p.alignment_factor
        )
        
        cohesion_dv = compute_cohesion(
            boid, self.boids,
            visual_range=p.visual_range,
            protected_range=p.protected_range,
            centering_factor=p.cohesion_factor
        )
        
        # Compute boundary steering
        boundary_dv = self.apply_boundary_steering(boid)
        
        # Apply all velocity adjustments
        boid.vx += sep_dv[0] + align_dv[0] + cohesion_dv[0] + boundary_dv[0]
        boid.vy += sep_dv[1] + align_dv[1] + cohesion_dv[1] + boundary_dv[1]
        
        # Enforce speed limits
        self.enforce_speed_limits(boid)
        
        # Update position
        boid.x += boid.vx
        boid.y += boid.vy
    
    def update(self) -> None:
        """
        Advance the simulation by one time step.
        
        Updates all boids. Note: Currently updates boids sequentially,
        which means later boids see partially-updated state of earlier
        boids. This is a known simplification; a more accurate approach
        would compute all adjustments first, then apply them.
        """
        for boid in self.boids:
            self.update_boid(boid)
    
    def get_positions(self) -> np.ndarray:
        """
        Get all boid positions as a numpy array.
        
        Returns:
            Array of shape (n_boids, 2) with [x, y] positions
        """
        return np.array([[b.x, b.y] for b in self.boids])
    
    def get_velocities(self) -> np.ndarray:
        """
        Get all boid velocities as a numpy array.
        
        Returns:
            Array of shape (n_boids, 2) with [vx, vy] velocities
        """
        return np.array([[b.vx, b.vy] for b in self.boids])