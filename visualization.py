"""
Pygame-based visualization for the Boids simulation.

Renders boids as triangles pointing in their velocity direction.
"""

import pygame
import numpy as np
import sys
from flock import Flock, SimulationParams


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BLUE = (100, 149, 237)  # Cornflower blue
DARK_BLUE = (25, 25, 112)      # Midnight blue


def draw_boid(surface: pygame.Surface, x: float, y: float, 
              vx: float, vy: float, size: float = 8, 
              color: tuple = LIGHT_BLUE) -> None:
    """
    Draw a single boid as a triangle pointing in its velocity direction.
    
    Args:
        surface: Pygame surface to draw on
        x, y: Boid position
        vx, vy: Boid velocity (determines orientation)
        size: Triangle size in pixels
        color: RGB tuple for fill color
    """
    # Calculate angle from velocity
    angle = np.arctan2(vy, vx)
    
    # Triangle vertices relative to center
    # Point in direction of velocity, with two rear vertices
    front = (x + size * np.cos(angle), 
             y + size * np.sin(angle))
    
    back_left = (x + size * 0.5 * np.cos(angle + 2.5), 
                 y + size * 0.5 * np.sin(angle + 2.5))
    
    back_right = (x + size * 0.5 * np.cos(angle - 2.5), 
                  y + size * 0.5 * np.sin(angle - 2.5))
    
    # Draw filled triangle
    pygame.draw.polygon(surface, color, [front, back_left, back_right])


def run_simulation(
    num_boids: int = 50,
    params: SimulationParams = None,
    fps: int = 60,
    show_fps: bool = True
) -> None:
    """
    Run the Boids simulation with pygame visualization.
    
    Args:
        num_boids: Number of boids to simulate
        params: Simulation parameters (uses defaults if None)
        fps: Target frames per second
        show_fps: Whether to display FPS counter
    """
    params = params or SimulationParams()
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((int(params.width), int(params.height)))
    pygame.display.set_caption(f"Boids Simulation — {num_boids} boids")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36) if show_fps else None
    
    # Create flock
    flock = Flock(num_boids=num_boids, params=params)
    
    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset simulation
                    flock = Flock(num_boids=num_boids, params=params)
        
        # Update simulation
        flock.update()
        
        # Draw
        screen.fill(DARK_BLUE)
        
        # Draw all boids
        for boid in flock.boids:
            draw_boid(screen, boid.x, boid.y, boid.vx, boid.vy)
        
        # Draw FPS
        if show_fps and font:
            fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
            screen.blit(fps_text, (10, 10))
        
        # Flip display
        pygame.display.flip()
        
        # Cap framerate
        clock.tick(fps)
    
    pygame.quit()


def run_headless(
    num_boids: int = 50,
    params: SimulationParams = None,
    num_steps: int = 1000
) -> Flock:
    """
    Run simulation without visualization (for testing/benchmarking).
    
    Args:
        num_boids: Number of boids
        params: Simulation parameters
        num_steps: Number of simulation steps to run
        
    Returns:
        The final Flock state
    """
    flock = Flock(num_boids=num_boids, params=params)
    
    for _ in range(num_steps):
        flock.update()
    
    return flock


if __name__ == "__main__":
    # Parse command line arguments
    num_boids = 50
    if len(sys.argv) > 1:
        try:
            num_boids = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [num_boids]")
            sys.exit(1)
    
    print(f"Starting Boids simulation with {num_boids} boids...")
    print("Controls:")
    print("  ESC: Quit")
    print("  R: Reset simulation")
    
    run_simulation(num_boids=num_boids)