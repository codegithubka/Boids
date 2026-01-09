"""
Optimized flocking rules using KDTree for spatial queries.

This module provides the same interface as rules.py but uses
scipy.spatial.KDTree for O(n log n) neighbor finding instead
of O(n²) naive iteration.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Tuple, Optional
from boid import Boid


class FlockState:
    """
    Maintains spatial index for efficient neighbor queries.
    
    Rebuilds KDTree each frame from current boid positions.
    """
    
    def __init__(self, boids: List[Boid]):
        """
        Initialize flock state with spatial index.
        
        Args:
            boids: List of all boids in the simulation
        """
        self.boids = boids
        self._positions: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None
        self._tree: Optional[KDTree] = None
        self._rebuild()
    
    def _rebuild(self) -> None:
        """Rebuild spatial index from current boid positions."""
        if len(self.boids) == 0:
            self._positions = np.empty((0, 2))
            self._velocities = np.empty((0, 2))
            self._tree = None
            return
        
        self._positions = np.array([[b.x, b.y] for b in self.boids])
        self._velocities = np.array([[b.vx, b.vy] for b in self.boids])
        self._tree = KDTree(self._positions)
    
    def update(self) -> None:
        """Call after boid positions change to rebuild spatial index."""
        self._rebuild()
    
    def query_neighbors(self, index: int, radius: float) -> List[int]:
        """
        Find all neighbors within radius of boid at given index.
        
        Args:
            index: Index of the query boid
            radius: Search radius
            
        Returns:
            List of indices of neighbors (excluding self)
        """
        if self._tree is None:
            return []
        
        position = self._positions[index]
        neighbor_indices = self._tree.query_ball_point(position, radius)
        
        # Remove self from results
        return [i for i in neighbor_indices if i != index]
    
    @property
    def positions(self) -> np.ndarray:
        """Get positions array."""
        return self._positions
    
    @property
    def velocities(self) -> np.ndarray:
        """Get velocities array."""
        return self._velocities


def compute_separation_kdtree(
    boid_index: int,
    flock_state: FlockState,
    protected_range: float,
    strength: float
) -> Tuple[float, float]:
    """
    Compute separation steering using KDTree for neighbor queries.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        protected_range: Distance threshold for separation
        strength: Multiplier for separation force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment
    """
    neighbors = flock_state.query_neighbors(boid_index, protected_range)
    
    if not neighbors:
        return (0.0, 0.0)
    
    boid_pos = flock_state.positions[boid_index]
    repel_x = 0.0
    repel_y = 0.0
    
    for neighbor_idx in neighbors:
        neighbor_pos = flock_state.positions[neighbor_idx]
        repel_x += boid_pos[0] - neighbor_pos[0]
        repel_y += boid_pos[1] - neighbor_pos[1]
    
    return (repel_x * strength, repel_y * strength)


def compute_alignment_kdtree(
    boid_index: int,
    flock_state: FlockState,
    visual_range: float,
    protected_range: float,
    matching_factor: float
) -> Tuple[float, float]:
    """
    Compute alignment steering using KDTree for neighbor queries.
    
    Excludes boids within protected range.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        visual_range: Distance threshold for visibility
        protected_range: Distance threshold for separation (excluded)
        matching_factor: Multiplier for velocity matching
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment
    """
    # Get neighbors in visual range
    visual_neighbors = set(flock_state.query_neighbors(boid_index, visual_range))
    
    # Get neighbors in protected range (to exclude)
    protected_neighbors = set(flock_state.query_neighbors(boid_index, protected_range))
    
    # Only consider neighbors in visual range but outside protected range
    valid_neighbors = visual_neighbors - protected_neighbors
    
    if not valid_neighbors:
        return (0.0, 0.0)
    
    boid_vel = flock_state.velocities[boid_index]
    
    sum_vx = 0.0
    sum_vy = 0.0
    
    for neighbor_idx in valid_neighbors:
        neighbor_vel = flock_state.velocities[neighbor_idx]
        sum_vx += neighbor_vel[0]
        sum_vy += neighbor_vel[1]
    
    neighbor_count = len(valid_neighbors)
    avg_vx = sum_vx / neighbor_count
    avg_vy = sum_vy / neighbor_count
    
    dvx = (avg_vx - boid_vel[0]) * matching_factor
    dvy = (avg_vy - boid_vel[1]) * matching_factor
    
    return (dvx, dvy)


def compute_cohesion_kdtree(
    boid_index: int,
    flock_state: FlockState,
    visual_range: float,
    protected_range: float,
    centering_factor: float
) -> Tuple[float, float]:
    """
    Compute cohesion steering using KDTree for neighbor queries.
    
    Excludes boids within protected range.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        visual_range: Distance threshold for visibility
        protected_range: Distance threshold for separation (excluded)
        centering_factor: Multiplier for centering force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment
    """
    # Get neighbors in visual range
    visual_neighbors = set(flock_state.query_neighbors(boid_index, visual_range))
    
    # Get neighbors in protected range (to exclude)
    protected_neighbors = set(flock_state.query_neighbors(boid_index, protected_range))
    
    # Only consider neighbors in visual range but outside protected range
    valid_neighbors = visual_neighbors - protected_neighbors
    
    if not valid_neighbors:
        return (0.0, 0.0)
    
    boid_pos = flock_state.positions[boid_index]
    
    sum_x = 0.0
    sum_y = 0.0
    
    for neighbor_idx in valid_neighbors:
        neighbor_pos = flock_state.positions[neighbor_idx]
        sum_x += neighbor_pos[0]
        sum_y += neighbor_pos[1]
    
    neighbor_count = len(valid_neighbors)
    avg_x = sum_x / neighbor_count
    avg_y = sum_y / neighbor_count
    
    dvx = (avg_x - boid_pos[0]) * centering_factor
    dvy = (avg_y - boid_pos[1]) * centering_factor
    
    return (dvx, dvy)


def compute_all_rules_kdtree(
    boid_index: int,
    flock_state: FlockState,
    visual_range: float,
    protected_range: float,
    cohesion_factor: float,
    alignment_factor: float,
    separation_strength: float
) -> Tuple[float, float]:
    """
    Compute all three flocking rules efficiently with shared neighbor queries.
    
    This is more efficient than calling each rule separately because
    we only query the KDTree twice (visual_range and protected_range)
    instead of three times.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        visual_range: Distance threshold for visibility
        protected_range: Distance threshold for separation
        cohesion_factor: Weight for cohesion
        alignment_factor: Weight for alignment
        separation_strength: Weight for separation
        
    Returns:
        Tuple (dvx, dvy) — combined velocity adjustment from all rules
    """
    boid_pos = flock_state.positions[boid_index]
    boid_vel = flock_state.velocities[boid_index]
    
    # Query neighbors once for each range
    visual_neighbors = set(flock_state.query_neighbors(boid_index, visual_range))
    protected_neighbors = set(flock_state.query_neighbors(boid_index, protected_range))
    
    # Neighbors for alignment/cohesion (in visual but outside protected)
    flocking_neighbors = visual_neighbors - protected_neighbors
    
    dvx = 0.0
    dvy = 0.0
    
    # Separation: repel from protected neighbors
    for neighbor_idx in protected_neighbors:
        neighbor_pos = flock_state.positions[neighbor_idx]
        dvx += (boid_pos[0] - neighbor_pos[0]) * separation_strength
        dvy += (boid_pos[1] - neighbor_pos[1]) * separation_strength
    
    # Alignment and Cohesion: only if we have flocking neighbors
    if flocking_neighbors:
        sum_x = 0.0
        sum_y = 0.0
        sum_vx = 0.0
        sum_vy = 0.0
        
        for neighbor_idx in flocking_neighbors:
            neighbor_pos = flock_state.positions[neighbor_idx]
            neighbor_vel = flock_state.velocities[neighbor_idx]
            sum_x += neighbor_pos[0]
            sum_y += neighbor_pos[1]
            sum_vx += neighbor_vel[0]
            sum_vy += neighbor_vel[1]
        
        n = len(flocking_neighbors)
        
        # Cohesion
        avg_x = sum_x / n
        avg_y = sum_y / n
        dvx += (avg_x - boid_pos[0]) * cohesion_factor
        dvy += (avg_y - boid_pos[1]) * cohesion_factor
        
        # Alignment
        avg_vx = sum_vx / n
        avg_vy = sum_vy / n
        dvx += (avg_vx - boid_vel[0]) * alignment_factor
        dvy += (avg_vy - boid_vel[1]) * alignment_factor
    
    return (dvx, dvy)


def compute_predator_avoidance_kdtree(
    boid_index: int,
    flock_state: FlockState,
    predator_x: float,
    predator_y: float,
    detection_range: float,
    avoidance_strength: float
) -> Tuple[float, float]:
    """
    Compute predator avoidance steering using KDTree-indexed positions.
    
    This is the fourth rule, added for Tier 2. Provides same interface
    as compute_predator_avoidance but uses FlockState for consistency
    with other optimized rules.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        predator_x: Predator x position
        predator_y: Predator y position
        detection_range: Distance at which boid detects predator
        avoidance_strength: Base multiplier for avoidance force
        
    Returns:
        Tuple (dvx, dvy) — velocity adjustment
    """
    boid_pos = flock_state.positions[boid_index]
    
    # Compute displacement from predator to boid (flee direction)
    dx = boid_pos[0] - predator_x
    dy = boid_pos[1] - predator_y
    
    # Compute distance
    squared_distance = dx * dx + dy * dy
    detection_range_squared = detection_range * detection_range
    
    # No avoidance if predator outside detection range
    if squared_distance >= detection_range_squared:
        return (0.0, 0.0)
    
    # Handle edge case: predator at exact same position
    if squared_distance < 1e-10:
        angle = np.random.uniform(0, 2 * np.pi)
        return (
            avoidance_strength * 10 * np.cos(angle),
            avoidance_strength * 10 * np.sin(angle)
        )
    
    # Scale avoidance inversely with distance
    distance = squared_distance ** 0.5
    scale = (detection_range - distance) / detection_range
    
    # Normalize direction and apply scaled strength
    dvx = (dx / distance) * avoidance_strength * scale * detection_range
    dvy = (dy / distance) * avoidance_strength * scale * detection_range
    
    return (dvx, dvy)


def compute_all_rules_with_predator_kdtree(
    boid_index: int,
    flock_state: FlockState,
    visual_range: float,
    protected_range: float,
    cohesion_factor: float,
    alignment_factor: float,
    separation_strength: float,
    predator_x: Optional[float],
    predator_y: Optional[float],
    predator_detection_range: float,
    predator_avoidance_strength: float
) -> Tuple[float, float]:
    """
    Compute all four flocking rules (including predator avoidance).
    
    Combines the three standard rules with predator avoidance for
    efficient single-pass computation.
    
    Args:
        boid_index: Index of the current boid
        flock_state: FlockState with spatial index
        visual_range: Distance threshold for visibility
        protected_range: Distance threshold for separation
        cohesion_factor: Weight for cohesion
        alignment_factor: Weight for alignment
        separation_strength: Weight for separation
        predator_x: Predator x position (None if no predator)
        predator_y: Predator y position (None if no predator)
        predator_detection_range: Distance for predator detection
        predator_avoidance_strength: Weight for predator avoidance
        
    Returns:
        Tuple (dvx, dvy) — combined velocity adjustment from all rules
    """
    # Get base flocking rules
    dvx, dvy = compute_all_rules_kdtree(
        boid_index, flock_state,
        visual_range, protected_range,
        cohesion_factor, alignment_factor, separation_strength
    )
    
    # Add predator avoidance if predator exists
    if predator_x is not None and predator_y is not None:
        pred_dvx, pred_dvy = compute_predator_avoidance_kdtree(
            boid_index, flock_state,
            predator_x, predator_y,
            predator_detection_range, predator_avoidance_strength
        )
        dvx += pred_dvx
        dvy += pred_dvy
    
    return (dvx, dvy)