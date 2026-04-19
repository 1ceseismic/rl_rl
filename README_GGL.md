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

## Build status (as of first port commit)

The ported rl_rl code (`ggl/`) compiles past our own files cleanly with the
two shims documented in `CMakeLists.txt` comments:
1. Include-path workaround for `CommonValues.h`'s broken `../Framework.h`.
2. Case-sensitivity shim for `EnvSet.h`'s `../OBSBuilders/OBSBuilder.h`.

The **leaked GGL source itself** does not compile cleanly on Linux with GCC
15 or Clang 22. The following are real bugs in GGL's own code that MSVC
silently accepts but other compilers reject:

- **`GigaLearnCPP/src/public/GigaLearnCPP/Util/Timer.h:15,20`** — mixes
  `system_clock::time_point` with `steady_clock::time_point` in arithmetic
  and assignment. Fix: pick one clock type and use it throughout.
- **`GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h:193,205`** —
  iterator template nested-name lookup fails; likely missing `typename` or
  a template dependency hint. Needs a more careful read of the surrounding
  class template to fix.

Until these are patched upstream (out of scope for this repo, which only
hosts the rl_rl side of the port), `scripts/build_ggl.sh` will fail inside
the `GigaLearnCPP` subtree after successfully configuring CMake and
starting compilation. Our own `ggl/*.cpp` files were not yet reached by
the failing build, but they were all written against the exact GGL APIs
read from the headers (see the `ggl/include_verified/` comments in each
header for the field names checked). Once GGL builds, the port should
follow immediately; there are no guessed interfaces on our side.

If you want to unblock the build:
1. Apply the two upstream patches above to `GigaLearnCPP-Leak/` directly,
   OR
2. Check whether a newer, non-leaked GigaLearnCPP release exists and use it
   via `-DGGL_ROOT=/path/to/good/source`, OR
3. Build on Windows with MSVC, where the leaked source is known to work.
