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
    
    # Configure simulation parameters
    # Based on Cornell reference, tuned for 800x600 screen
    params = SimulationParams(
        # Simulation bounds
        width=800,
        height=600,
        
        # Perception ranges
        visual_range=50,
        protected_range=12,
        
        # Speed constraints
        max_speed=3.0,
        min_speed=2.0,
        
        # Rule weights
        cohesion_factor=0.002,
        alignment_factor=0.06,
        separation_strength=0.15,
        
        # Boundary handling
        margin=75,
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