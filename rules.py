"""
Flocking rules for the Boids simulation.

Each rule computes a steering adjustment based on nearby boids.
Rules follow the specification from Phase 1 documentation.
"""

from typing import List, Tuple
from boid import Boid


def compute_separation(
    boid: Boid,
    all_boids: List[Boid],
    protected_range: float,
    strength: float
) -> Tuple[float, float]:
    """
    Compute separation steering: boid moves away from neighbors within protected range.
    
    Boids that are too close cause a repulsion force proportional to the 
    displacement vector. Forces from all nearby boids are accumulated.
    
    Args:
        boid: The current boid to compute steering for
        all_boids: List of all boids in the simulation
        protected_range: Distance threshold for separation (pixels)
        strength: Multiplier for the separation force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment to apply
        
    Algorithm (from Phase 1 spec):
        repelX, repelY = 0
        for each other boid o:
            if distance(b, o) < protected_range:
                repelX += (b.x - o.x)
                repelY += (b.y - o.y)
        return (repelX * strength, repelY * strength)
    """
    repel_x = 0.0
    repel_y = 0.0
    
    for other in all_boids:
        # Skip self-comparison
        if other is boid:
            continue
        
        # Compute displacement
        dx = boid.x - other.x
        dy = boid.y - other.y
        
        # Compute squared distance (avoid sqrt for comparison)
        squared_distance = dx * dx + dy * dy
        protected_range_squared = protected_range * protected_range
        
        # If within protected range, accumulate repulsion
        if squared_distance < protected_range_squared:
            repel_x += dx
            repel_y += dy
    
    # Apply strength factor
    return (repel_x * strength, repel_y * strength)


def compute_alignment(
    boid: Boid,
    all_boids: List[Boid],
    visual_range: float,
    protected_range: float,
    matching_factor: float
) -> Tuple[float, float]:
    """
    Compute alignment steering: boid matches velocity of visible neighbors.
    
    Boids within the protected range are excluded (they're being avoided,
    not flocked with). Only boids in the visual range but outside the
    protected range contribute to alignment.
    
    Args:
        boid: The current boid to compute steering for
        all_boids: List of all boids in the simulation
        visual_range: Distance threshold for neighbor visibility (pixels)
        protected_range: Distance threshold for separation (pixels)
        matching_factor: Multiplier for velocity matching (0 to 1)
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment to apply
    """
    sum_vx = 0.0
    sum_vy = 0.0
    neighbor_count = 0
    
    visual_range_squared = visual_range * visual_range
    protected_range_squared = protected_range * protected_range
    
    for other in all_boids:
        if other is boid:
            continue
        
        dx = boid.x - other.x
        dy = boid.y - other.y
        squared_distance = dx * dx + dy * dy
        
        # Only consider boids in visual range but outside protected range
        if protected_range_squared <= squared_distance < visual_range_squared:
            sum_vx += other.vx
            sum_vy += other.vy
            neighbor_count += 1
    
    if neighbor_count == 0:
        return (0.0, 0.0)
    
    # Compute average velocity of neighbors
    avg_vx = sum_vx / neighbor_count
    avg_vy = sum_vy / neighbor_count
    
    # Steer toward average velocity
    dvx = (avg_vx - boid.vx) * matching_factor
    dvy = (avg_vy - boid.vy) * matching_factor
    
    return (dvx, dvy)


def compute_cohesion(
    boid: Boid,
    all_boids: List[Boid],
    visual_range: float,
    protected_range: float,
    centering_factor: float
) -> Tuple[float, float]:
    """
    Compute cohesion steering: boid moves toward center of mass of visible neighbors.
    
    Boids within the protected range are excluded. Only boids in the visual 
    range but outside the protected range contribute to the center of mass.
    
    Args:
        boid: The current boid to compute steering for
        all_boids: List of all boids in the simulation
        visual_range: Distance threshold for neighbor visibility (pixels)
        protected_range: Distance threshold for separation (pixels)
        centering_factor: Multiplier for centering force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment to apply
    """
    sum_x = 0.0
    sum_y = 0.0
    neighbor_count = 0
    
    visual_range_squared = visual_range * visual_range
    protected_range_squared = protected_range * protected_range
    
    for other in all_boids:
        if other is boid:
            continue
        
        dx = boid.x - other.x
        dy = boid.y - other.y
        squared_distance = dx * dx + dy * dy
        
        # Only consider boids in visual range but outside protected range
        if protected_range_squared <= squared_distance < visual_range_squared:
            sum_x += other.x
            sum_y += other.y
            neighbor_count += 1
    
    if neighbor_count == 0:
        return (0.0, 0.0)
    
    # Compute center of mass of neighbors
    avg_x = sum_x / neighbor_count
    avg_y = sum_y / neighbor_count
    
    # Steer toward center of mass
    dvx = (avg_x - boid.x) * centering_factor
    dvy = (avg_y - boid.y) * centering_factor
    
    return (dvx, dvy)


def compute_predator_avoidance(
    boid: Boid,
    predator_x: float,
    predator_y: float,
    detection_range: float,
    avoidance_strength: float
) -> Tuple[float, float]:
    """
    Compute predator avoidance steering: boid flees from nearby predator.
    
    This is the fourth rule, added for Tier 2. Boids detect the predator
    from a greater distance than they detect each other, and avoidance
    has high priority to override cohesion when necessary.
    
    The avoidance force scales inversely with distance — closer predator
    means stronger avoidance.
    
    Args:
        boid: The current boid to compute steering for
        predator_x: Predator x position
        predator_y: Predator y position
        detection_range: Distance at which boid detects predator (pixels)
        avoidance_strength: Base multiplier for avoidance force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment to apply
        
    Edge cases:
        - Predator outside detection range: returns (0, 0)
        - Predator at exact same position: returns strong random direction
    """
    # Compute displacement from predator to boid (flee direction)
    dx = boid.x - predator_x
    dy = boid.y - predator_y
    
    # Compute distance
    squared_distance = dx * dx + dy * dy
    detection_range_squared = detection_range * detection_range
    
    # No avoidance if predator outside detection range
    if squared_distance >= detection_range_squared:
        return (0.0, 0.0)
    
    # Handle edge case: predator at exact same position
    # Apply strong avoidance in random direction
    if squared_distance < 1e-10:
        import numpy as np
        angle = np.random.uniform(0, 2 * np.pi)
        return (
            avoidance_strength * 10 * np.cos(angle),
            avoidance_strength * 10 * np.sin(angle)
        )
    
    # Scale avoidance inversely with distance
    # Closer predator = stronger avoidance
    distance = squared_distance ** 0.5
    scale = (detection_range - distance) / detection_range
    
    # Normalize direction and apply scaled strength
    dvx = (dx / distance) * avoidance_strength * scale * detection_range
    dvy = (dy / distance) * avoidance_strength * scale * detection_range
    
    return (dvx, dvy)