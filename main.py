#!/usr/bin/env python3
"""
Boids Flocking Simulation

A simulation of emergent flocking behavior using Craig Reynolds' Boids model.

Usage:
    python main.py [num_boids]
    
Controls:
    ESC: Quit
    R: Reset simulation

Author: Kimon Anagnostopoulos
Date: January 2026
"""

from visualization import run_simulation
from flock import SimulationParams


def main():
    """Run the Boids simulation with default parameters."""
    
    # Configure simulation parameters (Cornell University reference values)
    params = SimulationParams(
        # Simulation bounds
        width=800,
        height=600,
        
        # Perception ranges
        visual_range=20,
        protected_range=2,
        
        # Speed constraints
        max_speed=3.0,
        min_speed=2.0,
        
        # Rule weights
        cohesion_factor=0.0005,
        alignment_factor=0.05,
        separation_strength=0.05,
        
        # Boundary handling
        margin=50,
        turn_factor=0.2
    )
    
    # Run simulation
    run_simulation(
        num_boids=50,
        params=params,
        fps=60,
        show_fps=True
    )


if __name__ == "__main__":
    main()