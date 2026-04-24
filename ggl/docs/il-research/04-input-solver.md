# IL Research: RLCarInputSolver — replay state-pair to full controls

## 1. What it is and what it claims

[RLCarInputSolver](https://github.com/ZealanL/RLCarInputSolver) is a C++20 static
library by ZealanL (author of RocketSim) that takes two consecutive `SolverCarState`
structs and a `deltaTime`, and returns the full 8-float controller input
(throttle, steer, pitch, yaw, roll, jump, boost, handbrake) that would have
produced the state transition. The README states it handles:

- Throttle (all surfaces + air), steering (all surfaces), powerslide
- Boost (ground + air)
- Ground jumps (held/short), double jumps, flips (including direction), flip cancels, stalls
- Aerial pitch/yaw/roll with **partial analog input** support

**Claimed accuracy**: "for 99% of gameplay, the accuracy *should* be enough for most
purposes." Known failure modes include ambiguous diagonal flip cancels (only detected
until pitch torque ceases), delayed stall detection (~0.15 s), powerslide not detected
at zero angular velocity, inconsistent jump detection during angled landings, and
collisions in air being confused for flips/double-jumps.

**Status**: the README says "WORK IN PROGRESS" and the commit history (14 commits total,
last batch adding Python bindings) confirms active but early development. No versioned
releases. No test suite is shipped (the `TESTING_MODE` cmake option references
`test/AccuracyTest.cpp` but that file is not included in the repo).

## 2. How it works

### Architecture

The solver is a **physics-informed heuristic** — not a neural net, not a brute-force
forward sim. It uses RocketSim internally but only for specific queries (ground contact
checks, steering friction sampling), not full-tick simulation of candidate inputs.

**Entry point**: `RLCIS::Solve()` in `src/Solver.cpp:71`. It:

1. Calls `CheckOnGround(toState)` (line 77) which creates a thread-local
   `Arena` + `Car`, sets the car state, and calls RocketSim's internal
   `_PreTickUpdate()` / `_PostTickUpdate()` to run wheel traces and get
   `isOnGround` — without stepping physics.
2. Dispatches to `SolveGround()` or `SolveAir()` based on `isOnGround`.
3. Applies configurable deadzones and clamping (lines 92-109).

### Air solver (`src/AirSolver.cpp`)

The core of aerial PYR recovery is `ReverseAirOrientInputs()` (line 10), credited to
**Sam Mish** (`smish.dev/rocket_league/inverse_aerial_control/`). This is the same
analytical inverse of the RL torque equation used by rlgym-tools, with one enhancement:
when `angVelAfter` is near `CAR_MAX_ANG_SPEED`, it scales the target angular velocity up
by 1.25x to avoid outputting partial inputs at the angular velocity cap (line 13-17).

Beyond PYR, the air solver also reconstructs:

- **Boost**: compares velocity delta to expected `BOOST_ACCEL * forcesScale` (lines 72-83)
- **Air throttle**: checks for `THROTTLE_AIR_FORCE`-sized deltas (lines 85-89)
- **Flip detection**: measures flat velocity delta against `FLIP_INITIAL_VEL_SCALE`, then
  decomposes into forward/side components to recover the flip direction as pitch/yaw
  inputs, accounting for forward/backward/side impulse speed scaling (lines 96-135)
- **Double jump**: checks for `JUMP_IMMEDIATE_FORCE`-sized z-velocity delta (line 140)
- **In-flip state**: detects z-velocity damping pattern (`FLIP_Z_DAMP_120`) to identify
  ongoing flips, then checks for stalls (opposing yaw+roll angular velocity) and flip
  cancels (decreasing pitch angular velocity) (lines 155-206)
- **Continued jump hold**: detects `JUMP_ACCEL * deltaTime` z-delta (lines 211-216)

### Ground solver (`src/GroundSolver.cpp`)

- **Steering**: Uses RocketSim's Bullet physics internals directly — calls
  `vehicle.calcFrictionImpulses()` / `applyFrictionImpulses()` for steer=0 and steer=1,
  measures the resulting yaw angular acceleration difference, then linearly interpolates
  to find the steer value that matches the observed angular acceleration (lines 51-107).
  This is a **two-sample forward sim** — far cheaper than full-tick stepping.
- **Throttle/boost**: analytical based on drive torque constants from RocketSim's
  `RLConst` (`THROTTLE_TORQUE_AMOUNT`, `DRIVE_SPEED_TORQUE_FACTOR_CURVE`), with a
  correction for turning-induced under-prediction (lines 113-178).
- **Handbrake**: velocity alignment heuristic — detects alignment decreasing at
  `POWERSLIDE_RISE_RATE` vs increasing at `POWERSLIDE_FALL_RATE` (lines 181-216).
- **Jump**: detects large upward local-z velocity deltas (lines 219-228).

### Python bindings

Pybind11 bindings in `python/src/PYB_*.cpp` expose the module as `rlcis_py` with:

```python
from rlcis_py import Vec, RotMat, CarControls, SolverCarState, SolverConfig, SolveResult
result = rlcis_py.solve(state_before, state_after, delta_time, config)
# result.controls has: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
# result.is_on_ground, result.flip_started, result.is_flipping, result.double_jumping
```

Property names are auto-converted to snake_case by `PYB_MakePythonString()` in `PYB.h:31`.
The `SolverCarState` constructor takes `(pos: Vec, rot_mat: RotMat, vel: Vec, ang_vel: Vec)`.
`RotMat` takes `(forward: Vec, right: Vec, up: Vec)`.

## 3. Comparison: RLCarInputSolver vs rlgym-tools `predict_pyr` vs IDM

| Aspect | RLCarInputSolver | rlgym-tools `predict_pyr` | IDM (replay-pretraining) |
|---|---|---|---|
| **What it recovers** | All 8 controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake) | Only pitch, yaw, roll (other controls taken from replay columns) | All 8 as discrete action probabilities |
| **Aerial PYR method** | Sam Mish analytical inverse + max-angvel scaling | Same Sam Mish inverse + max-angvel scaling + flip-cancel detection from RLCarInputSolver (line 54 cites it) | Learned neural net (inverse dynamics model) |
| **Ground controls** | Physics-informed heuristic using RocketSim Bullet internals (steering via friction sim sampling) | N/A — throttle/steer come from replay bytes (`player.throttle`, `player.steer`) | Learned |
| **Flip/dodge handling** | Velocity-delta heuristic to recover flip direction + cancel detection | Uses `dodge_torque_x/y` columns from replay directly | Learned from (s_t, s_{t+1}) pairs |
| **Jump detection** | Velocity-delta heuristic | Uses `jump_is_active`, `dodge_is_active` columns from replay | Learned |
| **Boost detection** | Velocity-delta heuristic | Uses `boost_is_active` column from replay | Learned |
| **Language** | C++ (with pybind11 Python bindings) | Python (numba-accelerated) | Python + PyTorch |
| **Dependencies** | RocketSim (submodule, collision meshes required) | numpy, numba, scipy | PyTorch, trained checkpoint |
| **Speed** | Fast (no full sim steps, analytical + 2-sample steer) | Very fast (pure math, numba JIT) | Moderate (neural net inference per frame) |
| **Accuracy on PYR** | Identical to rlgym-tools (same algorithm, same constants) | Same | Slightly worse — learned approximation with training error |
| **Accuracy on non-PYR** | Good for 99% of gameplay per README; fails on edge cases (§1) | Exact — reads ground truth from replay | Depends on training data quality |
| **Needs replay metadata** | No — works from raw physics states only | Yes — needs `dodge_torque`, `*_is_active`, `throttle`, `steer` columns | Yes — needs (s_t, s_{t+1}) pairs + trained model |

**Key insight**: rlgym-tools already incorporates the RLCarInputSolver's flip-cancel
detection logic (see `inverse_aerial_controls.py:54`, which explicitly credits
`RLCarInputSolver/src/AirSolver.cpp`). For aerial PYR, the two approaches are
**functionally identical**.

The unique value of RLCarInputSolver is recovering **ground controls** (throttle, steer,
boost, handbrake, jump) from pure physics state pairs — which is unnecessary when
processing replays, because those controls are already present as replay columns
(`player.throttle`, `player.steer`, `boost_is_active`, `handbrake`, `jump_is_active`).

## 4. Integration into our pipeline

### When RLCarInputSolver would be useful

The solver is valuable when you have **state-pair data without embedded controls** — e.g.:

- Recovering controls from RocketSim recordings where only physics states were saved
- Validating that reconstructed controls match replay ground truth
- Building training data from non-replay sources (manual trajectories, procedural demos)

For **standard replay-to-BC conversion** (our current pipeline), it is **not needed**:
rlgym-tools' `convert.py` already reads throttle/steer/boost/handbrake/jump directly
from the replay columns, and the PYR reconstruction uses the identical Sam Mish
algorithm.

### If we wanted to integrate it anyway

1. **Build the Python module**:
   ```bash
   cd /tmp/RLCarInputSolver
   git submodule update --init --recursive
   mkdir build && cd build
   cmake .. -DMAKE_PYBIND=ON
   make -j$(nproc)
   # Output: python/out/rlcis_py.cpython-312-x86_64-linux-gnu.so
   ```
   Requires: RocketSim submodule checkout, pybind11, collision meshes at `./collision_meshes`.

2. **Call from our pipeline** (per frame pair):
   ```python
   import rlcis_py
   from rlcis_py import Vec, RotMat, SolverCarState, SolverConfig

   def states_to_controls(pos0, rot0, vel0, angvel0, pos1, rot1, vel1, angvel1, dt):
       s0 = SolverCarState(Vec(*pos0), RotMat(Vec(*rot0[:3]), Vec(*rot0[3:6]), Vec(*rot0[6:])),
                           Vec(*vel0), Vec(*angvel0))
       s1 = SolverCarState(Vec(*pos1), RotMat(Vec(*rot1[:3]), Vec(*rot1[3:6]), Vec(*rot1[6:])),
                           Vec(*vel1), Vec(*angvel1))
       cfg = SolverConfig()
       cfg.input_deadzone = 0.1
       result = rlcis_py.solve(s0, s1, dt, cfg)
       c = result.controls
       return [c.throttle, c.steer, c.pitch, c.yaw, c.roll,
               float(c.jump), float(c.boost), float(c.handbrake)]
   ```

3. **Gotcha**: the solver auto-initializes RocketSim (`RocketSim::Init("./collision_meshes")`)
   on first call. You need collision mesh files in the working directory, or you need to
   call `RLCIS::Init()` with the correct path before the first solve. This is hardcoded
   in `Solver.cpp:31` and would need a source patch for custom paths.

4. **Threading**: uses `thread_local Arena*` so it is thread-safe, but each thread
   allocates its own Arena (heavy memory — `ArenaMemWeightMode::HEAVY`).

### Does rlgym-tools already use it?

**Partially.** The flip-cancel detection in `rlgym_tools/rocket_league/math/inverse_aerial_controls.py:54`
is explicitly ported from RLCarInputSolver's `AirSolver.cpp`. The rest of RLCarInputSolver
(ground solver, full air solver with boost/jump/flip detection) is **not** used by
rlgym-tools, because rlgym-tools reads those controls from replay columns instead.

## 5. Verdict

**For our current IL pipeline (replay-based BC), RLCarInputSolver is not needed.**

The replay files already contain throttle, steer, boost, handbrake, and jump as
ground-truth columns. The only missing controls (pitch/yaw/roll) are recovered by
rlgym-tools using the identical Sam Mish analytical inverse that RLCarInputSolver uses
internally. rlgym-tools has already incorporated the one enhancement from
RLCarInputSolver (flip-cancel detection).

**When to use RLCarInputSolver:**

- **Validation tool**: run it on replay state pairs and compare its output to the
  replay's embedded controls. This would quantify the solver's accuracy and reveal
  systematic biases — useful if we ever doubt the replay data quality.
- **Non-replay data sources**: if we generate training trajectories from RocketSim
  (e.g., procedural kickoff drills, scripted aerial setups) and only log physics
  states, the solver can recover the controls that produced those trajectories.
- **Replacing the IDM**: if we were using Rolv-Arild's IDM approach (a learned neural
  net), RLCarInputSolver would be a strictly better alternative — deterministic,
  physics-grounded, no training required. But since we are using rlgym-tools'
  analytical approach (which is already equivalent), there is nothing to replace.

**Recommendation**: do not add RLCarInputSolver as a dependency. Bookmark it for future
use if we need state-pair-to-controls conversion outside of replay processing. The
solver's ground-contact steering reconstruction (using Bullet friction sampling) is
clever engineering that has no equivalent elsewhere, but it solves a problem we do not
currently have.
