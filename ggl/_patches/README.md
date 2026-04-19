# GGL source patches

The leaked GigaLearnCPP tree (`/path/to/GigaLearnCPP-Leak/`) has two Linux
build breakages that MSVC silently accepts. We patch them in place because
we now own the fork (no upstream to contribute back to).

## Patches

- `0001-timer-use-steady-clock.patch` — `Util/Timer.h` mixes `steady_clock`
  (field type) with `high_resolution_clock` (assignment/subtraction). On
  GCC/Clang these are distinct types. Fix: use `steady_clock` everywhere
  (correct for monotonic duration measurement).
- `0002-models-drop-deprecated-iterator.patch` — `Util/Models.h` iterator
  derives from the C++17-deprecated / C++20-removed `std::iterator` helper
  and contains ill-formed `typename Model*` uses. Fix: replace the base
  class with explicit iterator trait typedefs and drop the `typename`.
- `0003-rendersender-honor-ui-game-speed.patch` — `Util/RenderSender.cpp`
  now polls the Python receiver's `get_game_speed()` each frame so the
  rlviser UI's speed slider actually applies. Falls back silently to the
  config-time `timeScale` when the Python side doesn't export it.

## Applying

If the `GigaLearnCPP-Leak/` checkout is clean, apply from its root:

```bash
cd /path/to/GigaLearnCPP-Leak
patch -p1 < /home/seis/code/rl_rl/ggl/_patches/0001-timer-use-steady-clock.patch
patch -p1 < /home/seis/code/rl_rl/ggl/_patches/0002-models-drop-deprecated-iterator.patch
```

Alternatively if it's a git checkout:

```bash
cd /path/to/GigaLearnCPP-Leak
git apply /home/seis/code/rl_rl/ggl/_patches/*.patch
```

## Required but not patched here

- RLGymCPP's `CommonValues.h` does `#include "../Framework.h"` which is wrong
  on any platform (goes one dir too high). Worked around via CMake
  `include_directories(BEFORE SYSTEM ...Gamestates)` — see our top-level
  `CMakeLists.txt`. If you'd rather patch GGL directly, change that line
  to `#include "Framework.h"`.
- RLGymCPP's `EnvSet/EnvSet.h` includes `../OBSBuilders/OBSBuilder.h` (wrong
  case). Worked around with a forwarder at `ggl/_shim/OBSBuilders/OBSBuilder.h`.
  If you'd rather patch GGL, fix the case of that include.
