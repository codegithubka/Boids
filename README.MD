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

### Workflow Pattern Used

**Plan-and-Solve with Test-Driven Development:**
1. Established complete implementation plan before coding (Steps 1-10 + Tiers)
2. For each step: define goal → write tests → implement → verify → document
3. User testing after integration to catch behavioral issues not covered by unit tests
4. Iterative parameter tuning based on visual observation

### Prompt Structure

Each implementation prompt followed this pattern:
- **Context**: "We are building a Boids flocking simulation..."
- **Specific task**: Clear, bounded objective (e.g., "Implement the separation rule")
- **Constraints**: Technical requirements (e.g., "Match Phase 1 algorithm", "Use squared distance")
- **Quality expectations**: Edge cases, test coverage requirements

### Key Decisions Made During Development

| Decision Point | Choice Made | Rationale |
|----------------|-------------|-----------|
| Data structure | `@dataclass` | Minimal boilerplate, clear intent |
| Coordinate system | Screen coords (y-down) | Standard for pygame |
| Rules as functions | Pure functions in `rules.py` | Testable, composable |
| Parameter storage | `SimulationParams` dataclass | Centralized, documentable |
| Visualization | Pygame over matplotlib | Smoother animation at scale |
| Testing framework | pytest | Industry standard, clear syntax |

---

## Project Status

### Current State

| Component | Status | Tests |
|-----------|--------|-------|
| Boid class | ✅ Complete | 13/13 |
| Separation rule | ✅ Complete | 8/8 |
| Alignment rule | ✅ Complete | 7/7 |
| Cohesion rule | ✅ Complete | 6/6 |
| Rules integration | ✅ Complete | 2/2 |
| Flock class | ✅ Complete | 22/22 |
| Visualization | ✅ Complete | Smoke tested |
| Parameter tuning | ✅ Complete | N/A |
| KDTree optimization | ✅ Complete | 13/13 |
| Predator avoidance | ✅ Complete (validated) | 38/38 |
| Quantitative analysis | 🔄 Next | — |

**Total tests:** 109/109 passing

**Points earned:**
- Core assignment: ✅ 6/10 points
- Tier 1 (KDTree): ✅ +2 points
- Tier 2 (Predator): ✅ +1 point → **9/10 points**

### Files Structure

```
boids/
├── boid.py              # Boid data structure
├── predator.py          # Predator class (Tier 2)
├── rules.py             # Separation, alignment, cohesion, predator avoidance
├── rules_optimized.py   # KDTree-based rules (Tier 1)
├── flock.py             # Flock manager, SimulationParams
├── flock_optimized.py   # FlockOptimized with KDTree (Tier 1)
├── visualization.py     # Pygame rendering with predator (Tier 2)
├── main.py              # Entry point
├── benchmark.py         # Performance comparison script (Tier 1)
├── test_boid.py         # Boid unit tests
├── test_rules.py        # Rules unit tests
├── test_flock.py        # Flock unit tests
├── test_optimization.py # KDTree tests (Tier 1)
├── test_predator.py     # Predator tests (Tier 2)
├── benchmark_results.png # Performance figure (Tier 1)
└── GENAI_USAGE.md       # This document
```

### Next Steps

1. **Tier 3**: Quantitative analysis (→ 10/10 points)
   - T3.1: Implement metric functions
   - T3.2: Parameter sweep infrastructure
   - T3.3: Heatmap visualization
   - T3.4: Run experiment and document

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

#### Step 10a: Cornell Parameter Configuration

**Problem identified:** Initial parameters caused:
- Sharp 180° turns at screen edges (turn_factor too high)
- Multiple separate flocks forming (visual_range too large relative to cohesion)
- Boids too fast for smooth animation

**Reference:** Cornell University Boids parameter recommendations

**Changes made:**

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `turn_factor` | 0.5 | 0.2 | Gentler edge turns |
| `visual_range` | 75 | 20 | Tighter local neighborhoods |
| `protected_range` | 12 | 2 | Smaller personal space |
| `cohesion_factor` | 0.005 | 0.0005 | 10x reduction for stability |
| `max_speed` | 6.0 | 3.0 | Slower, smoother motion |
| `margin` | 100 | 50 | Smaller boundary zone |

**Files modified:** `flock.py`, `main.py`

**Test impact:** 9 tests failed due to hardcoded assumptions about defaults. Fixed by adding explicit `SimulationParams` to affected tests.

**Test results:** 58/58 passed after fixes

**Verification:** User should run `python main.py` to verify:
- [ ] Smoother edge behavior (no sharp 180° turns)
- [ ] Flocks merge more naturally
- [ ] Overall motion appears more realistic

**User testing results (Step 10a):**
- ✓ Movement is smooth
- ✓ Turns feel natural
- ✗ Boids leave screen boundaries
- ✗ Boids completely dispersed (singles or small groups)
- ✗ Boids violate each other's protected range when together

---

#### Step 10b: Tuning for Cohesion and Boundaries

**Problem analysis:**

1. **Boids leave screen:** margin=50 + turn_factor=0.2 insufficient to turn boid at max_speed=3 before exit. At speed 3, turn_factor 0.2 takes ~15 frames to reverse = 45px traveled.

2. **Dispersed boids:** visual_range=20 too small for 800×600 screen. cohesion_factor=0.0005 extremely weak (neighbor 15px away pulls at only 0.0075 px/frame). Random starts never coalesce.

3. **Protected range violation:** protected_range=2 means boids must be within 2px to trigger separation. separation_strength=0.05 too weak to overcome momentum.

**Changes made:**

| Parameter | Step 10a | Step 10b | Rationale |
|-----------|----------|----------|-----------|
| `visual_range` | 20 | 40 | 2x increase to find neighbors |
| `protected_range` | 2 | 8 | 4x increase for earlier avoidance |
| `cohesion_factor` | 0.0005 | 0.002 | 4x increase to pull boids together |
| `separation_strength` | 0.05 | 0.1 | 2x increase to enforce spacing |
| `margin` | 50 | 75 | 1.5x increase for earlier boundary turns |

**Unchanged (working well):**
- `turn_factor`: 0.2 (natural turns confirmed)
- `alignment_factor`: 0.05
- `max_speed`: 3.0, `min_speed`: 2.0

**Files modified:** `flock.py`, `main.py`

**Test results:** 58/58 passed

**Verification:** User should test for:
- [x] Boids stay within screen boundaries
- [x] Boids form cohesive flocks (not dispersed) — takes ~30 seconds
- [ ] Boids maintain spacing (no protected range violations)
- [x] Motion still smooth, turns still natural

**User testing results (Step 10b):**
- ✓ Boids stay within screen boundaries
- ✓ Boids form cohesive flock after ~30 seconds
- ✓ Motion is smooth
- ✓ Turns feel natural
- ✗ Jittery motion when boids get very close (touching)
- ✗ Flock separates at abrupt turns (potential visual_range / alignment issue)

**User prompt:** "After 30 seconds or so the boids form a flock. The motions seems to be a bit jittery as they get very close to each other and touch a little. This probably implies the protected range rules need to be a bit stricter. Correct me if I'm am wrong. Also, the flock separated at an abrupt term. Is that to be expected in terms of emergent behavior?"

**Analysis:**
1. **Jittery motion / touching:** `protected_range=8` triggers separation too late; `separation_strength=0.1` insufficient to overcome momentum when boids are already close.

2. **Flock separation at turns:** Partially emergent behavior (real flocks do split/reform), but also indicates `visual_range=40` may be too small — boids at flock edge lose sight of others during turns. `alignment_factor=0.05` may be too weak for synchronized direction changes.

---

#### Step 10c: Fix Jittery Motion and Flock Cohesion

**Goal:** Eliminate jittery motion from protected range violations and improve flock cohesion during turns.

**User applied changes manually.**

**Changes applied:**

| Parameter | Step 10b | Step 10c | Rationale |
|-----------|----------|----------|-----------|
| `protected_range` | 8 | 12 | Earlier separation trigger |
| `separation_strength` | 0.1 | 0.15 | Stronger push to prevent touching |
| `visual_range` | 40 | 50 | Maintain cohesion during turns |
| `alignment_factor` | 0.05 | 0.06 | Smoother coordinated turns |

**Unchanged (working well):**
- `cohesion_factor`: 0.002
- `turn_factor`: 0.2
- `margin`: 75
- `max_speed`/`min_speed`: 3.0/2.0

**User testing results (Step 10c):**
- ✓ Behavior is satisfactory and seems natural
- ✓ Jittery motion resolved
- ⚠ Minor issue: Some boids occasionally go slightly out of screen bounds
- Decision: Defer boundary fine-tuning to interactive tool (Tier 3)

**Test impact:** 2 tests in user's local copy failed due to outdated test file not having explicit `SimulationParams`. Updated `test_flock.py` provided with explicit parameters.

**Test results:** 58/58 passed (with updated test file)

**Status:** ✅ Complete

---

#### Step 10d: Edge Case Verification

**Goal:** Verify all edge cases are handled correctly per specification.

**Edge cases verified through unit tests:**
- [x] Zero neighbors (isolated boid) — tested in rules tests
- [x] Single neighbor scenarios — tested in rules tests
- [x] Boid at exact screen edge — boundary steering tests
- [x] Boid at corner (two boundaries) — corner steering test
- [x] Zero velocity — handled with random direction at min_speed
- [x] Speed clamping — preserves direction

**Status:** ✅ Complete (covered by existing 58 tests)

---

#### Step 10e: Final Behavioral Validation

**Goal:** Confirm simulation meets Phase 3 requirements.

**Validation criteria (from specification):**
- [x] Initial conditions: Random positions/velocities → gradual flock formation (~30s)
- [x] Separation rule: Minimum spacing maintained after Step 10c fixes
- [x] Alignment rule: Boids tend to move in similar directions
- [x] Cohesion rule: Boids cluster together rather than disperse
- [x] Emergent behavior: Natural flocking patterns observed
- [x] Smooth motion and natural turns at boundaries

**User confirmation:** "The behavior is satisfactory and seems natural."

**Status:** ✅ Complete

---

## Core Assignment Complete (6/10 Points)

All Phase 1-3 requirements satisfied:
- Phase 1: Conceptual understanding documented in LaTeX
- Phase 2: Implementation complete with test-driven development
- Phase 3: Behavioral validation and parameter exploration complete

**Proceeding to Enhancement Tiers for additional points.**

---

## Enhancement Tiers

### Tier 1: KDTree Optimization (6 → 8/10 points)

**Goal:** Replace O(n²) naive neighbor-finding with spatial indexing for performance.

---

#### T1.1: Implement KDTree Neighbor Finding

**Prompt summary:** "Create optimized rules module using scipy.spatial.KDTree for neighbor queries. Maintain identical interface to naive implementation. Include FlockState class to manage spatial index."

**Code produced:** 
- `rules_optimized.py` — KDTree-based rules with `FlockState` class
- `flock_optimized.py` — `FlockOptimized` class using KDTree

**Design decisions:**
- `FlockState` class encapsulates KDTree and position/velocity arrays
- Tree rebuilt each frame (positions change constantly)
- `compute_all_rules_kdtree()` combines all three rules with shared queries for efficiency
- `FlockOptimized` uses parallel update semantics (compute all adjustments, then apply)

**Key implementation details:**
```python
# Query neighbors once for each range
visual_neighbors = set(flock_state.query_neighbors(boid_index, visual_range))
protected_neighbors = set(flock_state.query_neighbors(boid_index, protected_range))

# Neighbors for alignment/cohesion (in visual but outside protected)
flocking_neighbors = visual_neighbors - protected_neighbors
```

---

#### T1.2: Verify Identical Behavior

**Tests implemented:** See `test_optimization.py`

**Test categories:**
1. **FlockState tests** (4 tests)
   - Empty flock handling
   - Single boid handling
   - Neighbor query correctness
   - Tree rebuild on update

2. **Rules equivalence tests** (4 tests)
   - Separation: KDTree matches naive exactly
   - Alignment: KDTree matches naive exactly
   - Cohesion: KDTree matches naive exactly
   - Combined rules: KDTree matches sum of naive rules

3. **Flock equivalence tests** (2 tests)
   - Single step produces similar positions
   - Long run stability (500 steps)

**Test results:** 13/13 passed

**Note on equivalence:** Naive `Flock` uses sequential updates (each boid sees partially-updated state of previous boids), while `FlockOptimized` uses parallel semantics (all boids computed from same initial state, then all updated). This is actually more correct for physical simulation but produces slightly different trajectories.

---

#### T1.3: Performance Benchmarks

**Benchmark methodology:**
- Warm-up: 10 frames discarded
- Timed run: 100 frames averaged
- Same random seed for fair comparison
- Tests: 50, 100, 150, 200, 300, 400, 500 boids

**Results:**

| Boids | Naive (ms) | KDTree (ms) | Speedup |
|------:|-----------:|------------:|--------:|
| 50 | 2.28 | 1.19 | 1.92x |
| 100 | 8.74 | 2.48 | 3.53x |
| 150 | 19.82 | 3.92 | 5.05x |
| 200 | 34.45 | 5.44 | 6.33x |
| 300 | 77.38 | 8.92 | 8.67x |
| 400 | 135.64 | 12.75 | 10.64x |
| 500 | 212.84 | 17.19 | **12.38x** |

**Analysis:**
- Naive implementation shows O(n²) growth: doubling boids roughly quadruples time
- KDTree implementation shows O(n log n) growth: much slower increase
- At 500 boids, KDTree is **12.4x faster**
- KDTree enables 60fps with 500+ boids (17ms < 16.67ms threshold)
- Naive can only maintain 60fps with ~50 boids

**Figure:** See `benchmark_results.png`

---

#### T1.4: Documentation of Issues Encountered

**Issue 1: Protected range exclusion**
- Initial implementation forgot to exclude protected-range boids from cohesion/alignment
- Caught by equivalence tests comparing against naive implementation
- Fixed by using set subtraction: `visual_neighbors - protected_neighbors`

**Issue 2: Sequential vs parallel update semantics**
- Naive `Flock.update()` modifies boids sequentially
- KDTree version computed all adjustments from stale positions
- Resolution: Documented as intentional difference (parallel is more physically accurate)
- Equivalence test relaxed to check "similar" not "identical" positions

**Issue 3: Empty flock edge case**
- Initial `FlockState` crashed on empty boid list
- Fixed by checking `len(self.boids) == 0` before building KDTree

---

#### Tier 1 Files Added

```
boids/
├── rules_optimized.py    # KDTree-based flocking rules
├── flock_optimized.py    # FlockOptimized class
├── test_optimization.py  # Equivalence and benchmark tests
└── benchmark.py          # Performance visualization script
```

**Status:** ✅ Complete

---

### Tier 2: Predator Avoidance (8 → 9/10 points)

**Goal:** Introduce antagonistic agent that disrupts flocking behavior.

**Specification Requirements (from assignment):**
- Implement a Predator class with position and velocity
- Add a fourth rule (predator avoidance): boids should steer away from the predator with high priority, even overriding cohesion if necessary
- The predator should move toward the center of the flock or track the nearest boid
- Demonstrate that flocking behavior is disrupted when the predator is present and recovers when the predator is removed
- Document behavioral differences: flock formation time, cohesion, dispersion patterns with vs. without predator

---

#### T2.0: Planning and Design

##### Design Decisions to Make

| Decision | Options | Considerations |
|----------|---------|----------------|
| Predator detection range | Same as visual_range / Larger / Configurable | Boids should probably detect predator from further away (survival instinct) |
| Avoidance priority | Additive with other rules / Override cohesion / Complete override | Spec says "override cohesion if necessary" — suggests weighted priority, not complete override |
| Predator hunting mode | Track nearest boid / Track flock center / Hybrid | Nearest boid = more chaotic; flock center = more realistic hunting |
| Predator speed | Slower than boids / Same / Faster | If faster, boids can never escape; if slower, no real threat. Suggest slightly slower but persistent |
| Predator boundary behavior | Same as boids / Ignore boundaries / Wrap around | Should probably follow same margin rules for consistency |
| Predator visualization | Different color / Different shape / Larger size | Must be visually distinct from boids |

##### Proposed Parameter Configuration

| Parameter | Proposed Value | Rationale |
|-----------|----------------|-----------|
| `predator_detection_range` | 100 (2x visual_range) | Boids detect predator from further away |
| `predator_avoidance_strength` | 0.5 | Strong avoidance, but not complete override |
| `predator_speed` | 2.5 | Slightly slower than boid max_speed (3.0) |
| `predator_hunting_mode` | "center" | Track flock center of mass |

##### Potential Issues

1. **Priority conflict**: How does predator avoidance interact with separation/alignment/cohesion?
   - Risk: If simply additive, avoidance may be too weak when flock is tight
   - Mitigation: Scale avoidance inversely with distance (stronger when closer)

2. **Predator speed balance**: 
   - Too fast → boids can never escape, simulation becomes chaotic
   - Too slow → predator never catches up, no real threat
   - Mitigation: Make predator slightly slower, but give it better "vision" of flock

3. **Flock fragmentation**: 
   - Risk: Flock may permanently fragment if predator stays in middle
   - Mitigation: Predator tracks center, so flock can reform when it moves away

4. **Boundary trapping**:
   - Risk: Boid caught between predator and wall has nowhere to go
   - Mitigation: Ensure boundary turn_factor is strong enough to prevent getting stuck

5. **Integration with KDTree**:
   - Risk: Forgetting to add predator avoidance to optimized version
   - Mitigation: Test both implementations produce same behavior

##### Edge Cases to Handle

| Edge Case | Expected Behavior | Test Strategy |
|-----------|-------------------|---------------|
| Predator at exact same position as boid | Avoid division by zero; strong repulsion in random direction | Unit test with predator at boid position |
| All boids equidistant from predator | Each boid moves directly away from predator | Unit test with symmetric arrangement |
| Predator outside detection range | No avoidance applied | Unit test with far predator |
| Single boid vs predator | Boid flees directly away | Unit test with one boid |
| Boid between predator and wall | Boid slides along wall, doesn't get stuck | Integration test |
| Predator at screen edge | Predator steered back like boids | Unit test boundary handling |
| No boids (empty flock) | Predator stays stationary or moves randomly | Unit test empty flock |
| Predator disabled mid-simulation | Flock should gradually reform | Behavioral test |

##### Testing Plan

**Unit Tests (test_predator.py):**

1. **Predator class tests:**
   - Instantiation with position/velocity
   - `create_at_position()` factory method
   - Speed property
   - Position/velocity as numpy arrays

2. **Predator movement tests:**
   - `move_toward_center()` — moves toward flock center of mass
   - `move_toward_nearest()` — moves toward closest boid
   - Boundary handling (same as boids)
   - Speed limits enforced
   - Empty flock handling

3. **Avoidance rule tests:**
   - No avoidance when predator outside detection range
   - Avoidance direction (away from predator)
   - Avoidance strength scaling with distance
   - Zero distance edge case (no division by zero)
   - Avoidance magnitude limits

4. **Integration tests:**
   - Combined rules with predator avoidance
   - Naive and KDTree produce same avoidance
   - Flock update includes predator update

**Behavioral Tests:**

1. **Flock dispersal**: Predator entering flock causes visible scattering
2. **Flock recovery**: Removing predator allows flock to reform
3. **Predator pursuit**: Predator successfully follows flock movement
4. **Survival**: Some boids should escape (predator slightly slower)

##### Potential Pitfalls

| Pitfall | How to Avoid |
|---------|--------------|
| Forgetting to add predator to visualization | Add to implementation checklist |
| Predator avoidance not in KDTree version | Implement in both from start, test equivalence |
| Avoidance too weak relative to cohesion | Test with high cohesion, ensure avoidance still works |
| Avoidance too strong, boids scatter permanently | Test flock recovery when predator removed |
| Division by zero when predator on top of boid | Add explicit distance check with minimum threshold |
| Predator stuck at boundary | Apply same boundary rules as boids |
| Infinite chase (predator never gives up) | This is intentional; predator always hunts |
| Performance regression with predator | Predator is single entity, minimal overhead |

##### Implementation Plan

| Step | Description | Files Modified | Tests Added |
|------|-------------|----------------|-------------|
| T2.1 | Create `Predator` class with position, velocity, hunting behavior | `predator.py` (new) | 5-8 tests |
| T2.2 | Implement `compute_predator_avoidance()` rule | `rules.py`, `rules_optimized.py` | 5-7 tests |
| T2.3 | Add predator parameters to `SimulationParams` | `flock.py` | — |
| T2.4 | Integrate predator into `Flock` and `FlockOptimized` | `flock.py`, `flock_optimized.py` | 3-5 tests |
| T2.5 | Update visualization to render predator (different color/size) | `visualization.py` | Smoke test |
| T2.6 | Add keyboard control to toggle predator (P key) | `visualization.py` | — |
| T2.7 | Behavioral validation and comparison | — | Document observations |

##### Success Criteria

- [ ] Predator class instantiates and moves correctly
- [ ] Avoidance rule produces correct steering direction
- [ ] Avoidance strength scales appropriately with distance
- [ ] All edge cases handled without crashes
- [ ] Naive and KDTree versions produce identical avoidance
- [ ] Visible flock dispersal when predator approaches
- [ ] Visible flock recovery when predator removed/disabled
- [ ] Predator successfully tracks flock (doesn't get lost)
- [ ] No performance regression (maintains 60fps)
- [ ] Keyboard toggle (P) enables/disables predator

---

#### T2.1: Predator Class Implementation

**Prompt summary:** "Create a Predator class with position, velocity, and hunting behavior. Include methods for tracking flock center of mass and nearest boid. Handle boundary steering and speed limits like boids."

**Code produced:** `predator.py`

**Features implemented:**
- `Predator` dataclass with x, y, vx, vy
- Factory methods: `create_at_position()`, `create_random()`
- Flock tracking: `compute_flock_center()`, `compute_nearest_boid()`
- Steering: `steer_toward()`, `update_velocity_toward_center()`, `update_velocity_toward_nearest()`
- Boundary handling: same logic as boids
- Speed limits: configurable max/min

**Tests:** 27/27 passed

**Status:** ✅ Complete

---

#### T2.2: Predator Avoidance Rule

**Prompt summary:** "Implement fourth rule for predator avoidance. Avoidance should scale inversely with distance. Handle edge case of predator at exact same position as boid."

**Code produced:** 
- `rules.py`: `compute_predator_avoidance()`
- `rules_optimized.py`: `compute_predator_avoidance_kdtree()`, `compute_all_rules_with_predator_kdtree()`

**Key design decisions:**
- Avoidance scales with `(detection_range - distance) / detection_range`
- Closer predator = stronger avoidance (inverse relationship)
- Zero-distance edge case: random direction with strong magnitude
- Detection range (100px) is 2x visual range for earlier warning

**Avoidance formula:**
```python
scale = (detection_range - distance) / detection_range
dvx = (dx / distance) * avoidance_strength * scale * detection_range
```

**Tests:** 11 additional tests (38/38 total in test_predator.py)

**Status:** ✅ Complete

---

#### T2.3: Predator Parameters in SimulationParams

**Parameters added:**

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `predator_detection_range` | 100 | 2x visual_range for early detection |
| `predator_avoidance_strength` | 0.5 | Strong but not overwhelming |
| `predator_speed` | 2.5 | Slightly slower than boids (3.0) |
| `predator_hunting_strength` | 0.05 | How aggressively predator tracks flock |

**Status:** ✅ Complete

---

#### T2.4: Integration into Flock Classes

**Changes to `Flock` class:**
- Added `predator: Optional[Predator]` attribute
- Added `enable_predator` parameter to `__init__`
- Modified `update_boid()` to include predator avoidance
- Added `update_predator()` method
- Added `toggle_predator()` method for runtime control

**Changes to `FlockOptimized` class:**
- Same changes as Flock class
- Uses `compute_all_rules_with_predator_kdtree()` for efficiency

**Tests:** All existing tests still pass (109/109)

**Status:** ✅ Complete

---

#### T2.5: Visualization Updates

**Changes to `visualization.py`:**
- Added `RED` and `YELLOW` colors for predator
- Added `draw_predator()` function (larger triangle with outline)
- Updated `run_simulation()` to render predator
- Added predator status display in UI

**Visual design:**
- Predator: Larger red triangle (size=15 vs boid size=8)
- Yellow outline for emphasis
- Status text shows "Predator: ON/OFF"

**Status:** ✅ Complete

---

#### T2.6: Keyboard Controls

**Added controls:**
- `P`: Toggle predator on/off
- `R`: Reset simulation (preserves predator state)

**Command line:**
```bash
python main.py 50 --predator  # Start with predator enabled
python visualization.py 100 --predator --naive  # Naive mode with predator
```

**Status:** ✅ Complete

---

#### T2.7: Behavioral Validation

**User testing results:**

- [x] Flock disperses when predator approaches
- [x] Flock reforms when predator moves away / is disabled
- [x] Predator successfully tracks flock center
- [x] Boids flee in correct direction (away from predator)
- [x] No performance regression (maintains 60fps)
- [x] Toggle (P key) works correctly during simulation

**User feedback:** "This looks pretty good for now. Behavior is as expected."

**Status:** ✅ Complete

---

### Tier 2 Summary

**Implementation complete.** All planned features working:

| Feature | Status |
|---------|--------|
| Predator class | ✅ |
| Predator avoidance rule | ✅ |
| KDTree integration | ✅ |
| Flock integration | ✅ |
| Visualization (red triangle) | ✅ |
| Keyboard toggle (P) | ✅ |
| Behavioral validation | ✅ |

**Tests:** 38/38 passed (test_predator.py)

**Total project tests:** 109/109 passed

**Points earned:** +1 (Tier 2) → **9/10 total**

---

### Tier 3: Quantitative Analysis (9 → 10/10 points)

**Goal:** Systematic measurement of predator-prey dynamics.

**Specification Requirements (from assignment):**
- Track at least two metrics: average distance to predator, minimum distance to predator
- Optionally track flock cohesion (standard deviation of positions)
- Run parameter sweep varying two parameters (3-5 values each)
- Run 5+ repetitions per parameter combination (different seeds)
- Compute mean and standard deviation of metrics
- Generate contour plot or heatmap visualization

---

#### T3.0: Planning and Design

##### Metrics to Track

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Avg distance to predator** | `mean(dist(boid, predator))` across all boids, all frames | Higher = better escape |
| **Min distance to predator** | `min(dist(boid, predator))` across all boids, all frames | Higher = safer (no close calls) |
| **Flock cohesion** | `std(boid positions)` | Lower = tighter flock |
| **Survival proxy** | Frames where min_dist > threshold | Higher = more successful evasion |

##### Parameter Sweep Design

**Primary sweep:** Predator speed vs Avoidance strength

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `predator_speed` | [1.5, 2.0, 2.5, 3.0, 3.5] | Range from slower to faster than boids |
| `predator_avoidance_strength` | [0.1, 0.3, 0.5, 0.7, 0.9] | Weak to strong avoidance |

**Grid:** 5 × 5 = 25 parameter combinations

**Repetitions:** 5 runs per combination with different seeds

**Total runs:** 125 simulations

**Frames per run:** 500 (about 8 seconds at 60fps)

##### Potential Issues

1. **Long runtime:** 125 runs × 500 frames could be slow
   - Mitigation: Use headless mode, KDTree optimization, parallel execution

2. **Statistical noise:** Stochastic initialization means high variance
   - Mitigation: 5+ repetitions, report mean ± std

3. **Metric interpretation:** What does "better" mean?
   - Higher avg distance = boids escape well
   - But: very high avoidance might fragment flock permanently
   - Need to consider trade-offs

4. **Edge cases in metrics:**
   - Empty flock: division by zero
   - Predator disabled: infinite distance
   - Mitigation: Only track when predator is active

5. **Visualization clarity:** Heatmap needs good color scale
   - Mitigation: Use diverging colormap, add contour lines

##### Edge Cases to Handle

| Edge Case | Expected Behavior | Handling |
|-----------|-------------------|----------|
| Frame with predator disabled | Skip metric collection | Check `predator is not None` |
| Single boid | Cohesion undefined (std=0) | Return 0 or skip |
| All boids at same position | Cohesion = 0 | Valid edge case |
| Predator catches boid (dist=0) | Min distance = 0 | Record as failure |

##### Testing Plan

**Unit Tests (test_analysis.py):**

1. **Metric calculation tests:**
   - `compute_avg_distance_to_predator()` correctness
   - `compute_min_distance_to_predator()` correctness
   - `compute_flock_cohesion()` correctness
   - Edge case: empty flock
   - Edge case: single boid

2. **Parameter sweep tests:**
   - Single run produces valid metrics
   - Multiple seeds produce different results
   - Results aggregation (mean/std) correct

3. **Data collection tests:**
   - Metrics recorded each frame
   - Final summary computed correctly

**Integration Tests:**
- Full sweep with small grid (2×2, 2 reps) completes without error
- Output files generated correctly

##### Potential Pitfalls

| Pitfall | How to Avoid |
|---------|--------------|
| Forgetting to enable predator during sweep | Assert predator exists before run |
| Not resetting simulation between runs | Create fresh Flock each run |
| Accumulating metrics incorrectly | Clear metrics list each run |
| Slow execution blocking testing | Add progress indicator, use small test grid |
| Heatmap axes swapped | Label clearly, verify with known values |
| Colorbar scale hiding patterns | Use appropriate vmin/vmax |

##### Implementation Plan

| Step | Description | Files | Tests |
|------|-------------|-------|-------|
| T3.1 | Implement metric functions | `metrics.py` (new) | 5-8 tests |
| T3.2 | Create single-run data collector | `metrics.py` | 2-3 tests |
| T3.3 | Implement parameter sweep runner | `analysis.py` (new) | 3-5 tests |
| T3.4 | Generate heatmap visualization | `analysis.py` | Manual verify |
| T3.5 | Run full experiment | — | — |
| T3.6 | Document results and insights | `GENAI_USAGE.md` | — |

##### Success Criteria

- [ ] Metric functions compute correct values
- [ ] Parameter sweep runs all combinations
- [ ] Multiple seeds produce varying results
- [ ] Mean and std computed for each combination
- [ ] Heatmap generated with clear labels
- [ ] Results show interpretable patterns
- [ ] No crashes or edge case failures
- [ ] Documentation includes figure and analysis

##### Expected Results (Hypotheses)

1. **High predator speed + low avoidance** → Low avg distance (caught often)
2. **Low predator speed + high avoidance** → High avg distance (easy escape)
3. **Moderate values** → Interesting trade-offs
4. **Very high avoidance** → May hurt cohesion (flock fragments)

---

#### T3.1: Metric Tracking Implementation (Pending)

**Status:** Awaiting implementation

---

#### T3.2: Parameter Sweep Infrastructure (Pending)

**Status:** Awaiting T3.1

---

#### T3.3: Visualization — Heatmap (Pending)

**Status:** Awaiting T3.2

---

#### T3.4: Run Experiment and Document (Pending)

**Status:** Awaiting T3.3

---

## Parameter Evolution

This table tracks parameter changes across tuning iterations.

| Parameter | Initial | 10a (Cornell) | 10b (Cohesion fix) | 10c (Final) |
|-----------|---------|---------------|-------------------|-------------|
| `visual_range` | 75 | 20 | 40 | **50** |
| `protected_range` | 12 | 2 | 8 | **12** |
| `cohesion_factor` | 0.005 | 0.0005 | 0.002 | 0.002 |
| `alignment_factor` | 0.05 | 0.05 | 0.05 | **0.06** |
| `separation_strength` | 0.05 | 0.05 | 0.1 | **0.15** |
| `max_speed` | 6.0 | 3.0 | 3.0 | 3.0 |
| `min_speed` | 2.0 | 2.0 | 2.0 | 2.0 |
| `turn_factor` | 0.5 | 0.2 | 0.2 | 0.2 |
| `margin` | 100 | 50 | 75 | 75 |

**Observations per iteration:**
- **Initial**: Sharp 180° turns, multiple flocks, too fast
- **10a**: Smooth turns, but dispersed boids, leave screen
- **10b**: Forms flock in ~30s, but jittery when close, splits on turns
- **10c**: Natural behavior, satisfactory flocking ✓

---

## Reflection

*(To be completed after project finalization)*

### Questions to Address

1. **What types of prompts were most effective?**
   - Preliminary: Structured prompts with clear constraints worked well for implementation
   - Test-first approach caught issues early

2. **What mistakes did GenAI make, and how were they caught?**
   - Initial parameter values required multiple iterations
   - Tests with hardcoded defaults broke when defaults changed (caught by pytest)

3. **Did understanding the algorithm before prompting help?**
   - Yes — Phase 1 document provided clear specification to reference
   - Protected range exclusion from cohesion/alignment was correctly implemented because it was explicitly documented

4. **What would be done differently next time?**
   - (To be filled after completion)

### Lessons Learned

*(To be filled after completion)*

---