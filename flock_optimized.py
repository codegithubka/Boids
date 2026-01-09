"""
Optimized Flock class using KDTree for spatial queries.

This module provides FlockOptimized which uses scipy.spatial.KDTree
for O(n log n) neighbor finding instead of O(n²) naive iteration.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from boid import Boid
from predator import Predator
from flock import SimulationParams
from rules_optimized import FlockState, compute_all_rules_kdtree, compute_all_rules_with_predator_kdtree


class FlockOptimized:
    """
    Optimized flock manager using KDTree for spatial queries.
    
    Drop-in replacement for Flock class with identical behavior
    but better performance for large numbers of boids.
    """
    
    def __init__(self, num_boids: int, params: SimulationParams = None,
                 enable_predator: bool = False):
        """
        Initialize the flock with random boids.
        
        Args:
            num_boids: Number of boids to create
            params: Simulation parameters (uses defaults if None)
            enable_predator: If True, create a predator (Tier 2)
        """
        self.params = params or SimulationParams()
        self.boids: List[Boid] = []
        self.predator: Optional[Predator] = None
        
        for _ in range(num_boids):
            boid = Boid.create_random(
                width=self.params.width,
                height=self.params.height,
                max_speed=self.params.max_speed
            )
            self.boids.append(boid)
        
        # Initialize spatial index
        self._flock_state = FlockState(self.boids)
        
        # Initialize predator if enabled
        if enable_predator:
            self.predator = Predator.create_random(
                width=self.params.width,
                height=self.params.height,
                speed=self.params.predator_speed
            )
    
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
        5. Update predator if present (Tier 2)
        """
        p = self.params
        
        # Rebuild spatial index with current positions
        self._flock_state.update()
        
        # Get predator position (or None)
        predator_x = self.predator.x if self.predator else None
        predator_y = self.predator.y if self.predator else None
        
        # Compute all velocity adjustments first (parallel semantics)
        adjustments = []
        
        for i, boid in enumerate(self.boids):
            # Compute flocking rules using KDTree (including predator avoidance)
            rules_dv = compute_all_rules_with_predator_kdtree(
                boid_index=i,
                flock_state=self._flock_state,
                visual_range=p.visual_range,
                protected_range=p.protected_range,
                cohesion_factor=p.cohesion_factor,
                alignment_factor=p.alignment_factor,
                separation_strength=p.separation_strength,
                predator_x=predator_x,
                predator_y=predator_y,
                predator_detection_range=p.predator_detection_range,
                predator_avoidance_strength=p.predator_avoidance_strength
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
        
        # Update predator (Tier 2)
        if self.predator is not None:
            self.update_predator()
    
    def update_predator(self) -> None:
        """
        Update the predator's state.
        
        The predator tracks the flock center of mass and moves toward it.
        Uses same boundary handling and speed limits as boids.
        """
        if self.predator is None:
            return
        
        p = self.params
        
        # Predator steers toward flock center
        self.predator.update_velocity_toward_center(
            self.boids,
            hunting_strength=p.predator_hunting_strength
        )
        
        # Apply boundary steering
        self.predator.apply_boundary_steering(
            width=p.width,
            height=p.height,
            margin=p.margin,
            turn_factor=p.turn_factor
        )
        
        # Enforce speed limits (predator has own speed)
        self.predator.enforce_speed_limits(
            max_speed=p.predator_speed,
            min_speed=p.predator_speed * 0.5  # Min is half of max
        )
        
        # Update position
        self.predator.update_position()
    
    def toggle_predator(self) -> bool:
        """
        Toggle predator on/off.
        
        Returns:
            True if predator is now enabled, False if disabled
        """
        if self.predator is None:
            self.predator = Predator.create_random(
                width=self.params.width,
                height=self.params.height,
                speed=self.params.predator_speed
            )
            return True
        else:
            self.predator = None
            return False
    
    def get_positions(self) -> np.ndarray:
        """Get all boid positions as a numpy array."""
        return np.array([[b.x, b.y] for b in self.boids])
    
    def get_velocities(self) -> np.ndarray:
        """Get all boid velocities as a numpy array."""
        return np.array([[b.vx, b.vy] for b in self.boids])