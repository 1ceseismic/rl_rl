# GGL Port — Rocket League RL training in C++

Port of the Python rlgym-learn training setup to GigaLearnCPP. The Python
files (`train.py`, `rewards.py`, etc.) are unchanged and still work; this
branch adds a parallel C++ path under `ggl/` that reproduces the same
training behavior against a faster inner loop.

## Layout

```
ggl/
  train.cpp                    — main() and LearnerConfig
  rewards/CustomRewards.h      — port of rewards.py custom classes
  mutators/RandomFieldState.h  — port of RandomFieldMutator
  mutators/DrillTimeoutCondition.h
  obs/NormalizedObs.h          — DefaultObs with train.py coefficients
  actions/ExpandedLookupAction.{h,cpp}  — 126-action lookup table
  StepCallback.{h,cpp}         — port of metrics.py:CustomMetricsProvider
CMakeLists.txt                 — integrates ../GigaLearnCPP-Leak/
scripts/build_ggl.sh
scripts/watch_ggl.py
```

## Prerequisites

1. **GigaLearnCPP source** at `../GigaLearnCPP-Leak/` (sibling of `rl_rl/`).
   Override with `-DGGL_ROOT=/other/path` if elsewhere.
2. **LibTorch** — PyTorch C++ API, CUDA or CPU. Download from
   `https://pytorch.org/get-started/locally/` (select "LibTorch" + your CUDA
   version or CPU). Extract to e.g. `/opt/libtorch` and set
   `TORCH_PATH=/opt/libtorch`.
3. **Python development headers** — required by the embedded pybind11.
   On Arch: `pacman -S python`. The project venv at `.venv/` has them.
4. **Collision meshes** — extract from a Rocket League install with
   [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper).
   Put the output folder at `./collision_meshes/` or point to it with
   `GGL_MESH_DIR`.
5. **C++20 compiler** — GCC 11+ or Clang 13+.

## Build

```bash
TORCH_PATH=/opt/libtorch scripts/build_ggl.sh
# Produces build-ggl/rl_rl_ggl
```

## Run

```bash
# Training (headless, default 36 envs)
GGL_MESH_DIR=./collision_meshes ./build-ggl/rl_rl_ggl

# Tuning per machine
GGL_NUM_GAMES=48 GGL_RUN_NAME=nyx-1 ./build-ggl/rl_rl_ggl

# Watch the latest checkpoint in rlviser
scripts/watch_ggl.py
```

### Env vars

| Variable          | Default                 | Meaning                                   |
|-------------------|-------------------------|-------------------------------------------|
| `GGL_MESH_DIR`    | `./collision_meshes`    | Folder with Rocket League collision meshes|
| `GGL_NUM_GAMES`   | `36`                    | Parallel envs (szmchn=36, nyx=48)         |
| `GGL_RUN_NAME`    | `ggl-port`              | wandb run name                            |
| `GGL_RENDER`      | unset                   | Any value enables rlviser render          |

## Porting notes (for future you)

- **Reward parity is not bit-exact.** Event rewards (`ShotReward`,
  `SaveReward`) use GGL's tick-event semantics; Python used cumulative
  `rsim_stats` counts. Over many episodes the aggregated magnitudes match.
- **`GoalRatioReward` → `GoalReward(concedeScale=-0.75)`** with weight 100
  reproduces the Python version's bias=0.25 behavior exactly.
- **`PickupBoostReward` scale differs.** Python was integer-per-pickup; GGL
  uses sqrt-normalized boost delta (~0.58 per 33-boost pad, 1.0 per
  100-boost). Revisit weight 15 if pickup incentive looks weak.
- **`InAirReward` semantics preserved** — deliberately differs from GGL's
  built-in `AirReward` (which rewards any airborne state including bumps).
- **Obs encoding differs from Python** (rotation matrix vs euler), so
  checkpoints are NOT interchangeable between this branch and master.
- **Drill mutators not ported.** Training config uses 60/40 kickoff/random,
  so drill timeouts aren't exercised. `DrillTimeoutCondition` is
  match-timeout-only.
- **`entropyScale` is not the same knob as Python `ent_coef`.** Kept at 0.01
  for intent; GGL's default is 0.018. Tune if learning stalls or explodes.

## Known gaps / TODO

- No inference-only "watch" mode — `watch_ggl.py` runs training with a low
  env count. Cleaner solution would be adding a `--watch` flag to `train.cpp`
  that early-returns after `learner->Load()`.
- No Python wandb receiver customization — GGL ships a default
  `metric_receiver.py` in its `python_scripts/` folder. If you need custom
  entity (`rl_rlbot`) or grouping, extend that script or wrap it.
- No checkpoint load flag — GGL auto-loads the latest in `checkpointFolder`.
  To resume from a specific timestep, rename the target dir to be the newest.

## Build status

Builds and runs end-to-end on Linux. Smoke-tested:
- 4-env, GPU (CUDA) training loop starts cleanly
- PPO model allocates (1.2M params — 636k policy + 572k critic)
- Obs size resolves to 89, action count to 126 (matches our ExpandedLookupAction)
- First iteration completes and writes a checkpoint to `checkpoints_ggl/`

Required to make the leaked GGL source build on Linux — captured as patches
in `ggl/_patches/` that must be applied once to the `GigaLearnCPP-Leak/`
checkout:

1. `0001-timer-use-steady-clock.patch` — `Util/Timer.h` mixed
   `steady_clock::time_point` (field) with `high_resolution_clock::now()`
   (assignment), which GCC/Clang reject.
2. `0002-models-drop-deprecated-iterator.patch` — `Util/Models.h` used the
   C++20-removed `std::iterator` base and malformed `typename Model*`.

Two additional leaked-source bugs are handled via CMake workarounds that
don't touch the out-of-repo tree:

- RLGymCPP `CommonValues.h` uses `#include "../Framework.h"` from the wrong
  directory level. Worked around by adding `...src/RLGymCPP/Gamestates/` as
  an include path so the preprocessor's fallback search resolves it.
- RLGymCPP `EnvSet.h` includes `../OBSBuilders/OBSBuilder.h` with the wrong
  case (actual dir is `ObsBuilders/`). Worked around by a forwarder header
  at `ggl/_shim/OBSBuilders/OBSBuilder.h` plus an anchor include path.

See `ggl/_patches/README.md` for application instructions and `CMakeLists.txt`
comments for the shim details.
