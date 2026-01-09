"""
Optimized Flock class using KDTree for spatial queries.

This module provides FlockOptimized which uses scipy.spatial.KDTree
for O(n log n) neighbor finding instead of O(n²) naive iteration.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from boid import Boid
from flock import SimulationParams
from rules_optimized import FlockState, compute_all_rules_kdtree


class FlockOptimized:
    """
    Optimized flock manager using KDTree for spatial queries.
    
    Drop-in replacement for Flock class with identical behavior
    but better performance for large numbers of boids.
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
        
        # Initialize spatial index
        self._flock_state = FlockState(self.boids)
    
    def apply_boundary_steering(self, boid: Boid) -> tuple:
        """
        Compute velocity adjustment to keep boid within bounds.
        
        Identical to Flock.apply_boundary_steering.
        """
        dvx = 0.0
        dvy = 0.0
        
        p = self.params
        
        if boid.x < p.margin:
            dvx += p.turn_factor
        if boid.x > p.width - p.margin:
            dvx -= p.turn_factor
        if boid.y < p.margin:
            dvy += p.turn_factor
        if boid.y > p.height - p.margin:
            dvy -= p.turn_factor
        
        return (dvx, dvy)
    
    def enforce_speed_limits(self, boid: Boid) -> None:
        """
        Clamp boid speed to within min/max bounds.
        
        Identical to Flock.enforce_speed_limits.
        """
        speed = boid.speed
        
        if speed == 0:
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
    
    def update(self) -> None:
        """
        Advance the simulation by one time step using KDTree optimization.
        
        Key difference from naive Flock:
        1. Rebuild spatial index once at start of frame
        2. Use KDTree queries for neighbor finding
        3. Compute all velocity adjustments first
        4. Apply all updates at end (parallel semantics)
        """
        p = self.params
        
        # Rebuild spatial index with current positions
        self._flock_state.update()
        
        # Compute all velocity adjustments first (parallel semantics)
        adjustments = []
        
        for i, boid in enumerate(self.boids):
            # Compute flocking rules using KDTree
            rules_dv = compute_all_rules_kdtree(
                boid_index=i,
                flock_state=self._flock_state,
                visual_range=p.visual_range,
                protected_range=p.protected_range,
                cohesion_factor=p.cohesion_factor,
                alignment_factor=p.alignment_factor,
                separation_strength=p.separation_strength
            )
            
            # Compute boundary steering
            boundary_dv = self.apply_boundary_steering(boid)
            
            adjustments.append((
                rules_dv[0] + boundary_dv[0],
                rules_dv[1] + boundary_dv[1]
            ))
        
        # Apply all adjustments
        for i, boid in enumerate(self.boids):
            boid.vx += adjustments[i][0]
            boid.vy += adjustments[i][1]
            
            self.enforce_speed_limits(boid)
            
            boid.x += boid.vx
            boid.y += boid.vy
    
    def get_positions(self) -> np.ndarray:
        """Get all boid positions as a numpy array."""
        return np.array([[b.x, b.y] for b in self.boids])
    
    def get_velocities(self) -> np.ndarray:
        """Get all boid velocities as a numpy array."""
        return np.array([[b.vx, b.vy] for b in self.boids])