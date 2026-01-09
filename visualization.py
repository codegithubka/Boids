"""
Pygame-based visualization for the Boids simulation.

Renders boids as triangles pointing in their velocity direction.
"""

import pygame
import numpy as np
import sys
from flock import Flock, SimulationParams
from flock_optimized import FlockOptimized


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BLUE = (100, 149, 237)  # Cornflower blue
DARK_BLUE = (25, 25, 112)      # Midnight blue
RED = (220, 60, 60)            # Predator color (Tier 2)
YELLOW = (255, 215, 0)         # Gold for highlights


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


def draw_predator(surface: pygame.Surface, x: float, y: float,
                  vx: float, vy: float, size: float = 15,
                  color: tuple = RED) -> None:
    """
    Draw the predator as a larger triangle with outline.
    
    The predator is visually distinct from boids: larger, red, with outline.
    
    Args:
        surface: Pygame surface to draw on
        x, y: Predator position
        vx, vy: Predator velocity (determines orientation)
        size: Triangle size in pixels (larger than boids)
        color: RGB tuple for fill color
    """
    # Calculate angle from velocity
    speed = (vx**2 + vy**2) ** 0.5
    if speed < 0.01:
        angle = 0
    else:
        angle = np.arctan2(vy, vx)
    
    # Triangle vertices (same shape as boid, just larger)
    front = (x + size * np.cos(angle), 
             y + size * np.sin(angle))
    
    back_left = (x + size * 0.6 * np.cos(angle + 2.5), 
                 y + size * 0.6 * np.sin(angle + 2.5))
    
    back_right = (x + size * 0.6 * np.cos(angle - 2.5), 
                  y + size * 0.6 * np.sin(angle - 2.5))
    
    # Draw filled triangle
    pygame.draw.polygon(surface, color, [front, back_left, back_right])
    
    # Draw outline for emphasis
    pygame.draw.polygon(surface, YELLOW, [front, back_left, back_right], 2)


def run_simulation(
    num_boids: int = 50,
    params: SimulationParams = None,
    fps: int = 60,
    show_fps: bool = True,
    use_kdtree: bool = True,
    enable_predator: bool = False
) -> None:
    """
    Run the Boids simulation with pygame visualization.
    
    Args:
        num_boids: Number of boids to simulate
        params: Simulation parameters (uses defaults if None)
        fps: Target frames per second
        show_fps: Whether to display FPS counter
        use_kdtree: If True, use KDTree optimization (Tier 1)
        enable_predator: If True, start with predator enabled (Tier 2)
    """
    params = params or SimulationParams()
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((int(params.width), int(params.height)))
    
    # Set window title based on optimization mode
    mode = "KDTree" if use_kdtree else "Naive"
    pygame.display.set_caption(f"Boids Simulation ({mode}) — {num_boids} boids")
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36) if show_fps else None
    small_font = pygame.font.Font(None, 24)
    
    # Create flock using selected implementation
    if use_kdtree:
        flock = FlockOptimized(num_boids=num_boids, params=params, enable_predator=enable_predator)
    else:
        flock = Flock(num_boids=num_boids, params=params, enable_predator=enable_predator)
    
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
                    predator_was_enabled = flock.predator is not None
                    if use_kdtree:
                        flock = FlockOptimized(num_boids=num_boids, params=params, 
                                               enable_predator=predator_was_enabled)
                    else:
                        flock = Flock(num_boids=num_boids, params=params,
                                      enable_predator=predator_was_enabled)
                elif event.key == pygame.K_p:
                    # Toggle predator (Tier 2)
                    predator_enabled = flock.toggle_predator()
                    status = "ON" if predator_enabled else "OFF"
                    print(f"Predator: {status}")
        
        # Update simulation
        flock.update()
        
        # Draw
        screen.fill(DARK_BLUE)
        
        # Draw all boids
        for boid in flock.boids:
            draw_boid(screen, boid.x, boid.y, boid.vx, boid.vy)
        
        # Draw predator if present (Tier 2)
        if flock.predator is not None:
            draw_predator(screen, flock.predator.x, flock.predator.y,
                         flock.predator.vx, flock.predator.vy)
        
        # Draw FPS and mode
        if show_fps and font:
            fps_text = font.render(f"FPS: {int(clock.get_fps())} ({mode})", True, WHITE)
            screen.blit(fps_text, (10, 10))
            
            # Draw predator status
            predator_status = "Predator: ON" if flock.predator else "Predator: OFF (P to toggle)"
            predator_color = RED if flock.predator else WHITE
            pred_text = small_font.render(predator_status, True, predator_color)
            screen.blit(pred_text, (10, 45))
        
        # Flip display
        pygame.display.flip()
        
        # Cap framerate
        clock.tick(fps)
    
    pygame.quit()


def run_headless(
    num_boids: int = 50,
    params: SimulationParams = None,
    num_steps: int = 1000,
    use_kdtree: bool = True,
    enable_predator: bool = False
) -> Flock:
    """
    Run simulation without visualization (for testing/benchmarking).
    
    Args:
        num_boids: Number of boids
        params: Simulation parameters
        num_steps: Number of simulation steps to run
        use_kdtree: If True, use KDTree optimization
        enable_predator: If True, include predator
        
    Returns:
        The final Flock state
    """
    if use_kdtree:
        flock = FlockOptimized(num_boids=num_boids, params=params, enable_predator=enable_predator)
    else:
        flock = Flock(num_boids=num_boids, params=params, enable_predator=enable_predator)
    
    for _ in range(num_steps):
        flock.update()
    
    return flock


if __name__ == "__main__":
    # Parse command line arguments
    num_boids = 50
    use_kdtree = True
    enable_predator = False
    
    if len(sys.argv) > 1:
        try:
            num_boids = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [num_boids] [--naive] [--predator]")
            sys.exit(1)
    
    if "--naive" in sys.argv:
        use_kdtree = False
    
    if "--predator" in sys.argv:
        enable_predator = True
    
    mode = "KDTree optimized" if use_kdtree else "Naive"
    pred_status = "with predator" if enable_predator else "without predator"
    print(f"Starting Boids simulation with {num_boids} boids ({mode}, {pred_status})...")
    print("Controls:")
    print("  ESC: Quit")
    print("  R: Reset simulation")
    print("  P: Toggle predator")
    
    run_simulation(num_boids=num_boids, use_kdtree=use_kdtree, enable_predator=enable_predator)