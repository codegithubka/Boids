"""
Flock class for managing the Boids simulation.

Handles initialization, update loop, and parameter configuration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from boid import Boid
from predator import Predator
from rules import compute_separation, compute_alignment, compute_cohesion, compute_predator_avoidance


@dataclass
class SimulationParams:
    """Configuration parameters for the simulation.
    
    Default values tuned through iterative testing (Steps 10a-10c).
    Predator parameters added in Tier 2.
    """
    
    # Simulation bounds
    width: float = 800
    height: float = 600
    
    # Perception ranges
    visual_range: float = 50
    protected_range: float = 12
    
    # Speed constraints
    max_speed: float = 3.0
    min_speed: float = 2.0
    
    # Rule weights
    cohesion_factor: float = 0.002
    alignment_factor: float = 0.06
    separation_strength: float = 0.15
    
    # Boundary handling
    margin: float = 75
    turn_factor: float = 0.2
    
    # Predator parameters (Tier 2)
    predator_detection_range: float = 100  # 2x visual_range
    predator_avoidance_strength: float = 0.5  # Strong avoidance
    predator_speed: float = 2.5  # Slightly slower than boids
    predator_hunting_strength: float = 0.05  # How aggressively predator tracks flock


class Flock:
    """
    Manages a collection of boids and runs the simulation.
    
    Attributes:
        boids: List of all boids in the flock
        params: Simulation parameters
        predator: Optional predator (Tier 2)
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
        2. Compute predator avoidance (Tier 2)
        3. Apply boundary steering
        4. Update velocity
        5. Enforce speed limits
        6. Update position
        
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
        
        # Compute predator avoidance (Tier 2)
        predator_dv = (0.0, 0.0)
        if self.predator is not None:
            predator_dv = compute_predator_avoidance(
                boid,
                predator_x=self.predator.x,
                predator_y=self.predator.y,
                detection_range=p.predator_detection_range,
                avoidance_strength=p.predator_avoidance_strength
            )
        
        # Compute boundary steering
        boundary_dv = self.apply_boundary_steering(boid)
        
        # Apply all velocity adjustments
        boid.vx += sep_dv[0] + align_dv[0] + cohesion_dv[0] + predator_dv[0] + boundary_dv[0]
        boid.vy += sep_dv[1] + align_dv[1] + cohesion_dv[1] + predator_dv[1] + boundary_dv[1]
        
        # Enforce speed limits
        self.enforce_speed_limits(boid)
        
        # Update position
        boid.x += boid.vx
        boid.y += boid.vy
    
    def update(self) -> None:
        """
        Advance the simulation by one time step.
        
        Updates all boids and the predator (if present).
        Note: Currently updates boids sequentially,
        which means later boids see partially-updated state of earlier
        boids. This is a known simplification; a more accurate approach
        would compute all adjustments first, then apply them.
        """
        # Update all boids
        for boid in self.boids:
            self.update_boid(boid)
        
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