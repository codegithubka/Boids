# GenAI Usage Documentation — Boids Simulation

**Author:** Kimon Anagnostopoulos  
**Date:** January 2026  
**AI Assistant:** Claude (Anthropic)

---

## Overview

This document records the prompting strategy, code generation process, and iterative refinements used to build a Boids flocking simulation with AI assistance. Each development step is documented with the prompt goal, code produced, evaluation, and any corrections made.

---

## Prompting Strategy

The implementation follows an incremental, test-driven approach:

1. **Decomposition**: Break the simulation into self-contained modules (Boid class, individual rules, visualization)
2. **Test-first mentality**: Each module includes unit tests before integration
3. **Iterative refinement**: Evaluate generated code against the Phase 1 specification, request corrections as needed
4. **Documentation**: Record every interaction for reproducibility

---

## Implementation Log

### Step 1: Boid Class (Data Structure)

**Goal:** Create a minimal `Boid` class that stores position (x, y) and velocity (vx, vy).

**Constraints:**
- Simple data structure, no methods yet
- Should support easy instantiation with random or specified values
- Use standard Python with NumPy for vector operations where beneficial

**Prompt summary:** "Implement a Boid class as a data structure storing position and velocity. Include a factory method for random initialization within given bounds."

**Code produced:** See `boid.py`

**Tests:** See `test_boid.py`

**Evaluation:**
- [x] Boid instantiates with explicit position/velocity
- [x] Boid instantiates with random values within bounds
- [x] Attributes are accessible and modifiable
- [x] Edge case: zero velocity handled

**Test results:** 13/13 tests passed

**Issues found:** None — implementation worked on first attempt.

**Design decisions:**
- Used `@dataclass` for clean, minimal boilerplate
- Added `speed`, `position`, and `velocity` properties for convenience
- Factory method `create_random()` encapsulates random initialization
- Random velocity uses angle-based generation for uniform direction distribution

---

### Step 2: Separation Rule

**Goal:** Implement the separation behavior — boids steer away from neighbors within protected range.

**Constraints:**
- Match the algorithm from Phase 1 specification
- Input: current boid, list of all boids, protected range, strength
- Output: velocity adjustment tuple (dvx, dvy)
- Handle edge case: no neighbors in protected range

**Prompt summary:** "Implement separation, alignment, and cohesion rules in a single module. Each rule should be a pure function returning velocity adjustments. Separation uses protected range only; alignment and cohesion use visual range but exclude boids in protected range (per the Phase 1 pseudocode structure)."

**Code produced:** See `rules.py` — `compute_separation()` function

**Tests:** 8 tests covering:
- No neighbors → zero adjustment
- Neighbor outside protected range → zero adjustment
- Single neighbor in range → correct repulsion direction
- Multiple symmetric neighbors → forces cancel
- Asymmetric neighbors → net force
- Strength factor scaling
- Diagonal neighbor → both x and y components

**Evaluation:**
- [x] Matches Phase 1 algorithm exactly
- [x] Uses squared distance to avoid unnecessary sqrt
- [x] Accumulates displacement vectors from all intruders
- [x] Applies strength factor correctly

**Test results:** 8/8 passed

---

### Step 3: Alignment Rule

**Goal:** Implement velocity matching with visible neighbors.

**Constraints:**
- Boids in protected range are excluded
- Only boids in visual range (but outside protected range) contribute
- Returns velocity adjustment toward average neighbor velocity

**Code produced:** See `rules.py` — `compute_alignment()` function

**Tests:** 7 tests covering:
- No neighbors → zero adjustment
- Neighbor in protected range → excluded
- Neighbor outside visual range → excluded
- Same velocity → zero adjustment
- Different velocity → steer toward neighbor
- Matching factor scaling
- Opposing velocities → average to zero

**Evaluation:**
- [x] Correctly excludes boids in protected range
- [x] Computes average velocity of valid neighbors
- [x] Applies matching factor as (avg - current) * factor

**Test results:** 7/7 passed

---

### Step 4: Cohesion Rule

**Goal:** Implement steering toward center of mass of visible neighbors.

**Constraints:**
- Same visibility rules as alignment (exclude protected range)
- Steer toward center of mass, not individual neighbors

**Code produced:** See `rules.py` — `compute_cohesion()` function

**Tests:** 6 tests covering:
- No neighbors → zero adjustment
- Neighbor in protected range → excluded
- Single neighbor → steer toward neighbor position
- Centering factor scaling
- Multiple neighbors → steer toward center of mass
- Equilateral triangle → boid at centroid has zero net force

**Evaluation:**
- [x] Correctly computes center of mass
- [x] Steering is (center - position) * factor
- [x] Symmetric arrangements produce balanced forces

**Test results:** 6/6 passed

**Additional integration tests:** 2 tests verifying correct interaction between rules (protected range exclusion behavior)

**Total for Steps 2-4:** 23/23 tests passed

---

### Step 5: Combined Update Loop

**Goal:** Integrate all three rules into a single update step per boid.

**Constraints:**
- Apply rules in order: separation, alignment, cohesion
- Add boundary steering
- Enforce speed limits after all adjustments
- Update position last

**Prompt summary:** "Create a Flock class with SimulationParams dataclass. Implement the combined update loop that applies all rules, boundary steering, speed limits, and position updates in the correct order."

**Code produced:** See `flock.py` — `Flock.update_boid()` method

**Design decisions:**
- Created `SimulationParams` dataclass to encapsulate all tunable parameters
- `Flock` manages list of boids and provides update methods
- Sequential update (noted as simplification vs. parallel update)

**Test results:** See Step 8 summary

---

### Step 6: Boundary Handling

**Goal:** Implement edge avoidance with turn factor.

**Constraints:**
- Screen coordinates: (0,0) top-left, y increases downward
- Soft margins: gradual steering, not hard teleportation
- Each edge handled independently

**Code produced:** See `flock.py` — `Flock.apply_boundary_steering()` method

**Tests:** 7 tests covering:
- No steering in center
- Left/right/top/bottom margin steering directions
- Corner steering (both dimensions)
- Turn factor magnitude

**Evaluation:**
- [x] Correct coordinate system (y increases downward)
- [x] Independent handling of each margin
- [x] Steering direction pushes boid toward center

**Test results:** 7/7 passed

---

### Step 7: Speed Limits

**Goal:** Enforce minimum and maximum speed constraints.

**Constraints:**
- Preserve velocity direction when clamping
- Handle edge case: zero speed

**Code produced:** See `flock.py` — `Flock.enforce_speed_limits()` method

**Tests:** 5 tests covering:
- Speed within limits unchanged
- Speed above max clamped
- Speed below min boosted
- Direction preserved when clamped
- Zero speed handled (random direction at min_speed)

**Evaluation:**
- [x] Direction preservation via unit vector scaling
- [x] Zero speed edge case handled with random direction

**Test results:** 5/5 passed

---

### Step 8: Position Update

**Goal:** Update boid positions based on velocity.

**Implementation:** Simple Euler integration: `position += velocity`

**Code produced:** Integrated into `Flock.update_boid()` method

**Tests for full integration (Steps 5-8):**
- 3 tests for flock initialization
- 7 tests for boundary handling
- 5 tests for speed limits  
- 3 tests for single boid update
- 2 tests for full flock update
- 2 tests for helper methods

**Total for Steps 5-8:** 22/22 tests passed

**Cumulative test count:** 58/58 passed

---

### Step 9: Visualization (Pygame)

**Goal:** Render boids and animate the simulation.

**Constraints:**
- Smooth 60fps animation
- Boids rendered as triangles pointing in velocity direction
- Dark background for visibility
- FPS counter for debugging
- Keyboard controls (ESC to quit, R to reset)

**Prompt summary:** "Create pygame visualization with triangular boids oriented by velocity. Include headless mode for testing and benchmarking. Add keyboard controls."

**Code produced:** See `visualization.py` and `main.py`

**Features implemented:**
- `draw_boid()`: Renders boid as oriented triangle
- `run_simulation()`: Main pygame loop with event handling
- `run_headless()`: Non-visual simulation for testing
- Command-line argument for boid count
- FPS display toggle

**Smoke test:** Headless simulation with 20 boids runs 100 steps successfully

**Evaluation:**
- [x] Boids rendered as triangles
- [x] Orientation matches velocity direction
- [x] ESC and R keyboard controls
- [x] FPS counter displayed
- [x] Headless mode for testing

**Note:** Full visual testing requires running on a system with display. Headless mode verified programmatically.

---

### Step 10: Parameter Tuning

**Goal:** Adjust parameters for realistic flocking behavior.

*(to be filled)*

---

## Enhancement Tiers

### Tier 1: KDTree Optimization

*(to be filled)*

---

### Tier 2: Predator Avoidance

*(to be filled)*

---

### Tier 3: Quantitative Analysis

*(to be filled)*

---

## Reflection

*(to be filled after completion)*

---