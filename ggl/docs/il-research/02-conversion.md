# IL Research: Replay frames → (obs, action) for GGL

Scope: how to turn parsed replay frames (carball parquet output, or equivalent) into
tensors compatible with GigaLearnCPP's `DefaultObs` (RLGymCPP) and `DefaultAction` (90
discrete actions). The RL trainer is C++, but the IL pipeline is Python — obs vectors
and action indices are just `numpy` arrays and can be shipped to the C++ side via
`.npz` / `.pt` files.

Exact references (absolute paths):

- Obs builder: `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/src/RLGymCPP/ObsBuilders/DefaultObs.cpp` and `DefaultObs.h`
- Action parser: `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/src/RLGymCPP/ActionParsers/DefaultAction.cpp` and `DefaultAction.h`
- Inversion: `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/StateUtil.cpp`
- Game state: `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates/{GameState.h,Player.h}`
- Constants: `/home/seis/code/GigaLearnCPP-Leak/GigaLearnCPP/RLGymCPP/src/RLGymCPP/CommonValues.h`
- Python rlgym lookup table (identical 90 entries): `/home/seis/code/rl_rl/.venv/lib/python3.12/site-packages/rlgym/rocket_league/action_parsers/lookup_table_action.py`
- Canonical IL code (cloned): `/home/seis/code/rl_rl/.claude/worktrees/il-research-scraping/ggl/docs/il-research/replay-pretraining/`

## 1. Tick alignment

### What rates are in play
- RocketSim physics tick: **120 Hz** (`CommonValues::TICK_TIME = 1/120`).
- GGL decision rate: **15 Hz** (tickSkip = 8). The actor picks one action per step;
  RLGymCPP then feeds that action to every one of the 8 ticks in the skip (see
  `DefaultAction::ParseAction` returning a single `Action` and `Player::UpdateFromCar`
  storing `prevAction` for a full step).
- Rocket League replay native rate: **30 Hz** physics network frames, but the delta
  per frame is **variable** (carball stores `game["delta"]` per frame — typical 1/30,
  occasional larger gaps during lag or at kickoff). 60 Hz replays exist for older
  replays but modern ones are 30 Hz; carball normalizes to the same schema regardless.

### Alignment strategy (recommended)
Two workable approaches; community work uses (a).

(a) **Resample replay frames onto a fixed 15 Hz grid in replay time.** Since GGL is
15 Hz and replays are 30 Hz, take every 2nd frame (or, more robustly, the replay frame
whose cumulative `delta` is closest to `k / 15` seconds). Rolv-Arild's
`replay-pretraining` does something equivalent: it decimates at feature-extraction time
(`get_data_df` in `replay-pretraining/replay_pretraining/replays/replays.py:17`) and
runs the IDM at replay rate, then down-selects to training windows. For pure BC we can
skip the IDM and just downsample.

(b) **Re-simulate the replay in RocketSim at 120 Hz** with `ReplaySetter` / state
injection, then read GameState every 8 ticks. This is what `rlgym-tools`'
`ReplaySetter` supports. Higher fidelity (no interpolation error) but ~10x slower and
requires every replay state to be reachable, which it generally isn't (goal cutscenes,
demos, pre-kickoff).

For BC, (a) is standard and what we should do first.

### Action during the 8 skipped ticks
In RL inference: the agent picks one `action_idx`, RLGymCPP expands it via
`DefaultAction::ParseAction` to 8 floats, and the *same* 8 floats are applied to every
one of the 8 ticks (see `CarState::Tick` loop driven by `Arena::Step`). `Player.prevAction`
is updated from that single action, not a per-tick history.

In a replay: the human's controls can change every tick (or, more precisely, every
replay physics update). When we downsample to 15 Hz we lose the sub-step variation.
Two reasonable choices:

- **First-tick of window** (simplest; matches what the agent sees as "prev_action"):
  use the controls at the frame that starts the 8-tick window.
- **Majority / mean of the 8 ticks**: closer to what the human "meant" over that
  decision interval. For BC targets, `replay-pretraining` uses a **ratings-based match**
  (see `get_actions_from_player` in `replays.py:217`) — it computes a plausibility
  rating per lookup-table entry per frame and lets a separate IDM model combine them.

For our first cut, use *first-tick controls*. The discretization error is small
compared to the IDM's noise, and this keeps the pipeline fully deterministic.

## 2. Replay-state → obs conversion

### Canonical output order of `DefaultObs::BuildObs`

Reading `DefaultObs.cpp:19-53` top to bottom, for a given agent player:

| Section | Floats | Source |
|---|---|---|
| ball.pos × posCoef | 3 | `state.ball.pos` (inverted if agent is orange) |
| ball.vel × velCoef | 3 | `state.ball.vel` |
| ball.angVel × angVelCoef | 3 | `state.ball.angVel` |
| prevAction | 8 | `player.prevAction` — throttle, steer, pitch, yaw, roll, jump, boost, handbrake |
| boost pads | 34 | `state.GetBoostPads(inv)` — `bool` → float, inverted order for orange |
| self: player block | 20 | see below |
| teammates | 20 × N_tm | each teammate block |
| opponents | 20 × N_op | each opponent block |

Per-player block (20 floats), from `AddPlayerToObs` (`DefaultObs.cpp:4-17`):

| Field | Floats |
|---|---|
| pos × posCoef | 3 |
| rotMat.forward | 3 |
| rotMat.up | 3 |
| vel × velCoef | 3 |
| angVel × angVelCoef | 3 |
| boost / 100 | 1 |
| isOnGround (0/1) | 1 |
| HasFlipOrJump() (0/1) | 1 |
| isDemoed (0/1) | 1 |
| **total** | **20** |

### Normalization constants
From `CommonValues.h:12-42` and the `DefaultObs` default constructor:
- `posCoef = Vec(1/4096, 1/5120, 1/2044)` — axis-wise (per-axis normalization, not
  isotropic).
- `velCoef = 1/2300`.
- `angVelCoef = 1/5.5`.
- `player.boost` divided by 100.

Note: this differs from the Python `rlgym.rocket_league.obs_builders.DefaultObs` which
uses isotropic `pos_coef=1/2300` and also includes a 9-dim partially-observable jump
state block (`is_holding_jump`, `handbrake`, `has_jumped`, `is_jumping`, `has_flipped`,
`is_flipping`, `has_double_jumped`, `can_flip`, `air_time_since_jump`) — our C++
version does **not**. Any community IL code targeting the Python DefaultObs will need
its obs builder swapped out before the .npz is usable by our trainer.

### Rotation: quaternion → (forward, up)
Replays give quaternions. RLGymCPP's `PhysState::rotMat` is a 3x3 rotation matrix with
columns `{forward, right, up}` (see `StateUtil.cpp:7-14`: inversion scales each column
by `(-1,-1,1)` — confirming columns are the basis vectors). The Python math for
quat→rotmat with the Rocket League convention is in
`/home/seis/code/rl_rl/.venv/lib/python3.12/site-packages/rlgym/rocket_league/math.py:117`
(`quat_to_rot_mtx`) — note the leading sign flip `w=-quat[0]` etc. which matches Rocket
League's quaternion handedness. The numba version in `replay-pretraining` utils
(`inverse_aerial_controls.py:14`) is identical.

```python
def forward_up_from_quat(quat_wxyz):
    R = quat_to_rot_mtx(quat_wxyz)   # 3x3
    return R[:, 0], R[:, 2]          # columns 0 and 2 = forward, up
```

Carball parquet quaternion columns are `quat_w, quat_x, quat_y, quat_z`
(`replays.py:29`, `:105`).

### Inversion for orange team: `InvertPhys`
From `StateUtil.cpp:3-17`:
```
result.pos  *= (-1, -1,  1)
result.vel  *= (-1, -1,  1)
result.angVel *= (-1, -1,  1)
for each column c of rotMat: c *= (-1, -1, 1)
```
i.e. 180° rotation about the Z axis. The orange player now sees the field as if the
blue goal were in the +Y direction — identical to what the blue player sees —
so a single actor network can be trained without a team-indicator input.

### Inverted boost pads
`state.boostPads` is the 34-entry boolean list in the ordering of `BOOST_LOCATIONS` in
`CommonValues.h:45-80`. `state.boostPadsInv` is the same list permuted so that
`pads_inv[i]` corresponds to location `mirror_z180(BOOST_LOCATIONS[i])` — i.e. the pad
that a rotated orange player should "see" at slot `i`. Mapping:

```python
# Build once at import time.
BOOST_LOCATIONS = np.array([...])  # copy the 34 tuples from CommonValues.h
# Rotate 180° around Z: (x,y,z) -> (-x,-y,z).
mirrored = BOOST_LOCATIONS * np.array([-1, -1, 1])
# Find each mirrored location in the original list.
inv_pad_index = np.array([
    np.argmin(np.linalg.norm(BOOST_LOCATIONS - m, axis=1))
    for m in mirrored
])
# Usage: pads_inv = pads[inv_pad_index]
```

GGL's `DefaultObs` emits `(float)pads[i]` i.e. a 0/1 availability flag — **not** the
timer. That means for IL we only need to know "is this pad currently available" at the
frame. Carball gives us frame-level `boost_pickup > 0` events per player, from which
we can reconstruct the 4 s (small pad) / 10 s (big pad) cooldown — see
`replays.py:165-171` for a reference implementation. We then threshold `> 0` to get
the same 0/1 the C++ side sees.

## 3. Continuous controls → discrete action

### Regenerate the 90-entry lookup table in Python
`DefaultAction.cpp:3-59` builds the table deterministically. The easiest way to stay
in sync is to reuse the rlgym Python copy — the same construction, same 90 entries, same
order:

```python
from rlgym.rocket_league.action_parsers import LookupTableAction
LOOKUP = LookupTableAction.make_lookup_table()   # shape (90, 8)
```

Each row is `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]` — same order
as `DefaultAction::actions` and as `RLGC::Action`. I verified the two tables are
identical by reading both files (ground loop 24 entries, aerial loop with the
`jump==1 and yaw!=0` skip giving the remaining 66). **index 0 is not a no-op** —
it is `[-1, -1, 0, -1, 0, 0, 0, 0]` (full reverse, full left). The closest to no-op
is `[0, 0, 0, 0, 0, 0, 0, 0]` which appears at index 10 (throttle=0, steer=0, boost=0,
handbrake=0, rest zero) — but caller shouldn't rely on that, use nearest-neighbour.

Action count breakdown:

- Ground: 24. Generator loops throttle∈{-1,0,1} × steer∈{-1,0,1} × boost∈{0,1} ×
  handbrake∈{0,1} = 36, minus the 12 cases with boost=1 and throttle≠1 (useless
  boosting) → 24.
- Aerial: 66. 162 combos minus duplicates: skips (jump=1 & yaw≠0), skips the
  pitch=roll=jump=0 row as a duplicate with ground.

Total 90. The aerial block also sets `throttle=boost` and `steer=yaw`, so e.g. a
"boost + pitch-up + no yaw" entry exists with throttle=1.

### Matching a replay frame's continuous controls to a discrete action
For a given frame with controls `c = [throttle, steer, pitch, yaw, roll, jump, boost,
handbrake] ∈ R^8`, pick `argmin_i ||LOOKUP[i] - c||_2`. Boolean controls in `c` should
be 0 or 1 (not `True`/`False`).

This is adequate for 80% of frames. The two failure modes:

1. **Flips and dodges**: the instantaneous `pitch/yaw/roll` inferred from angular
   velocity differences does not correspond to what the human pressed during a dodge —
   the human's stick direction sets the dodge torque at the *start* of the dodge, then
   the game applies a fixed torque pulse for ~0.15 s regardless. `replay-pretraining`
   handles this by reading the per-frame `dodge_torque_x/y/z` column, normalizing to
   max-1, and using *that* as the target stick direction while `dodging` is active
   (see `replays.py:234-244`).
2. **Double-jump vs. directional dodge**: both press jump=1 in the air. Disambiguated
   by whether `|pitch|+|yaw|+|roll| < 0.5` (the dodge deadzone) at the jump frame. If
   below deadzone, it's a double-jump (all torque components should be 0); otherwise
   it's a dodge with the stick direction above.

For our first cut we can ignore both and just nearest-neighbour all frames — the BC
loss on dodges will be high but the model still learns ground driving well. A better
cut is to apply the dodge_torque fix only for the handful of frames where
`dodge_is_active` flips true.

### Python rlgym's `LookupTableAction` — what does it do with inputs?
It's pure forward: it holds the 90x8 table, accepts an integer `action_idx` per
agent, and outputs the 8-float control vector. It has no quantizer — the "inverse"
mapping we need is our own responsibility. See
`rlgym/rocket_league/action_parsers/lookup_table_action.py:24-35`:

```python
def parse_actions(self, actions, state, shared_info):
    return {agent: self._lookup_table[a] for agent, a in actions.items()}
```

### What tick's controls represent the 8-tick window?
Given the discussion above (rl applies the same action for all 8 ticks), the **first
tick of the 8-tick window** is the cleanest target: that's the one where the agent's
choice gets committed, and in our simulator the identical action is what
`prevAction` reflects for the next step's obs. For dodges we still want to be looking
at the frame where `dodge_is_active` flips true — but that frame will be within the
window, so we can special-case: if any replay frame in [t, t+8 ticks) has
`dodge_is_active` flipping, use that frame's controls; else use the first frame's
controls.

## 4. Existing conversion code

Cloned: `/home/seis/code/rl_rl/.claude/worktrees/il-research-scraping/ggl/docs/il-research/replay-pretraining/`

Key files and what they give us:

- `replay_pretraining/replays/replays.py`
  - `load_parsed_replay`: reads carball's parquet output — `__ball.parquet`,
    `__game.parquet`, `player_{uid}.parquet`, plus `metadata.json` and
    `analyzer.json`. This is the replay-frame schema we should target: columns
    `pos_x/y/z`, `quat_w/x/y/z`, `vel_x/y/z`, `ang_vel_x/y/z`, `boost_amount`,
    `is_sleeping`, `throttle`, `steer`, `handbrake`, `jump_is_active`,
    `dodge_is_active`, `double_jump_is_active`, `flip_car_is_active`,
    `boost_is_active`, `boost_pickup`, `dodge_torque_x/y/z`, `match_goals`,
    `match_saves`, `match_shots`.
  - `to_rlgym_dfs`: builds a "wide" dataframe similar to a rlgym_sim GameState
    encoding, including per-player inverted variants.
  - `get_actions_from_player` (`replays.py:217`): the *reference* continuous→discrete
    matcher, producing per-frame "ratings" per lookup-table row. Our simpler
    argmin-L2 replaces this; their version is designed to feed an IDM.
  - `get_data_df` (`replays.py:17`): per-agent feature extractor producing a 45-dim
    feature vector per frame. **This is Necto/Nexto's feature layout, NOT our GGL
    DefaultObs layout** — do not reuse directly.
- `replay_pretraining/bcm/bc_dataset.py`: shows the BC-side loader. Useful as a
  reference for shard layout (`.npz` with `x_data`, `y_data`, gzip-compressed) and
  episode-boundary masking. Again, `x_data` width (231) is Necto/Nexto-shaped, not
  ours.
- `replay_pretraining/replays/inverse_aerial_controls.py`: ground-truth
  `aerial_inputs()` function we can reuse verbatim to infer pitch/yaw/roll from
  (omega_start, omega_end, theta_start, dt). `@njit` numba-optimized.
- `replay_pretraining/replays/label_replays.py`: wraps the IDM model to emit final
  action labels. We do not need the IDM for a first cut but can steal the
  `to_rlgym_dfs → get_data_df → argmax over lookup_table` pipeline.

Other community repos skimmed:

- `Rolv-Arild/Necto` (same author, superset). Their obs is a transformer-shaped obs
  (`Nexto/advanced_obs.py`) with per-player tokens — not our layout.
- `SaltieRL/carball` (also cloned at `il-research/carball/`). Gives us the
  `ControlsCreator` path that works on the older tree-pickle output. Only useful if we
  pivot away from the parquet path; modern carball (`decompile_replays.py`) emits
  parquet.
- `rlrml/subtr-actor`, `jjbott/RocketLeagueReplayParser`: lower-level parsers without
  RL-aligned schema. Not useful for IL directly.

**No repo I found matches our exact DefaultObs layout.** We'll need to write
`replay_frame_to_default_obs()` ourselves, reusing the quat/inversion/pad-timer
utilities from `replay-pretraining`.

## 5. Recommendations for our project

### Function sketch

```python
# ggl/python_scripts/il/convert.py
import numpy as np
from typing import TypedDict

class ReplayFrame(TypedDict):
    # All fields at a single replay time t (already downsampled to 15 Hz)
    ball_pos: np.ndarray           # (3,)
    ball_vel: np.ndarray           # (3,)
    ball_ang_vel: np.ndarray       # (3,)
    pad_available: np.ndarray      # (34,) bool
    players: list                  # list of Player dicts (see below)
    agent_index: int               # index into players[]
    prev_action: np.ndarray        # (8,) previous action from the sim convention

class Player(TypedDict):
    team: int                      # 0 blue, 1 orange
    pos: np.ndarray                # (3,)
    quat_wxyz: np.ndarray          # (4,)
    vel: np.ndarray
    ang_vel: np.ndarray
    boost: float                   # [0,100]
    on_ground: bool
    has_flip_or_jump: bool
    is_demoed: bool

POS_COEF = np.array([1/4096, 1/5120, 1/2044], dtype=np.float32)
VEL_COEF = np.float32(1/2300)
ANG_COEF = np.float32(1/5.5)
INV_VEC  = np.array([-1, -1, 1], dtype=np.float32)

def _player_block(p, inv):
    forward, up = _fwd_up(p['quat_wxyz'])
    pos, vel, ang = p['pos'], p['vel'], p['ang_vel']
    if inv:
        pos, vel, ang = pos * INV_VEC, vel * INV_VEC, ang * INV_VEC
        forward = forward * INV_VEC
        up      = up * INV_VEC
    return np.concatenate([
        pos * POS_COEF, forward, up,
        vel * VEL_COEF, ang * ANG_COEF,
        [p['boost']/100, float(p['on_ground']),
         float(p['has_flip_or_jump']), float(p['is_demoed'])],
    ], dtype=np.float32)

def replay_frame_to_rl_sample(frame: ReplayFrame, inv_pad_index, lookup_table):
    agent = frame['players'][frame['agent_index']]
    inv = agent['team'] == 1
    pos, vel, ang = frame['ball_pos'], frame['ball_vel'], frame['ball_ang_vel']
    if inv:
        pos, vel, ang = pos*INV_VEC, vel*INV_VEC, ang*INV_VEC
    pads = frame['pad_available'].astype(np.float32)
    if inv:
        pads = pads[inv_pad_index]
    head = np.concatenate([
        pos*POS_COEF, vel*VEL_COEF, ang*ANG_COEF,
        frame['prev_action'].astype(np.float32), pads
    ])
    blocks = [_player_block(agent, inv)]
    tm, op = [], []
    for i, p in enumerate(frame['players']):
        if i == frame['agent_index']: continue
        (tm if p['team'] == agent['team'] else op).append(_player_block(p, inv))
    obs = np.concatenate([head] + blocks + tm + op, dtype=np.float32)

    action_vec = frame['controls']                 # (8,) already [throttle,...,handbrake]
    action_idx = int(np.argmin(np.linalg.norm(lookup_table - action_vec, axis=1)))
    return obs, action_idx
```

### Expected obs vector length (1v1)
- ball: 3 + 3 + 3 = 9
- prev_action: 8
- boost pads: 34
- self block: 20
- opponent block: 20
- teammates: 0
Total = **91 floats** per sample for 1v1.

Formulae: `9 + 8 + 34 + 20 + 20*(n_players - 1)`. For 2v2 `= 9 + 8 + 34 + 20 + 20*3 = 131`.
For 3v3 = 171.

Sanity check against the `AdvancedObs` equivalent would be good once we're in-tree.
Note `DefaultObsPadded` is a zero-padded variant; if we train from scratch against
mixed team sizes we'll likely want padded obs — but for first-pass 1v1 IL, plain
DefaultObs is fine.

### Storage format
Recommendation: **sharded compressed `.npz`**, ~1 hour of replay (~54k samples at 15 Hz)
per shard, matching `replay-pretraining`'s `shard_size = 30 * 60 * 60` default. One
shard per split (train/val/test) per batch of replays. Each shard holds:

```
x_data: (N, 91)       float32  — concatenated obs
y_data: (N,)          int32    — action_idx ∈ [0, 89]
ep_ends: (M,)         int32    — indices where an episode ends (goal / goalpost)
meta: structured      — source replay id, frame indices, rank of players
```

`.npz` (numpy) over hdf5 because:
- No extra dependency (numpy is always there).
- `replay-pretraining` uses the same format, so their BC dataset code
  (`bcm/bc_dataset.py`) is reusable.
- Compressed (`np.savez_compressed`) gets us ~5x over raw on these obs shapes.

One flat stream (not per-replay) because the C++ trainer we're warm-starting into
already has an experience buffer that's flat — treating BC as a single contiguous
dataset simplifies the dataloader. Episode boundaries are preserved in `ep_ends` so
the BC loop can reset any LSTM state if we later add one. For our current MLP actor,
ep_ends is not strictly necessary.

### Open questions / risks

- **Boost pad availability**: reconstructed from `boost_pickup` events. The replay
  occasionally drops pickup events (network-replicated), so a few pads will look
  "always-available" when they aren't. Low impact for BC (the pad bits are 34 of 91
  features).
- **Demo timing**: replays tell us who demoed whom but the exact respawn timer is
  reconstructed from `is_sleeping`. Fine for `is_demoed` bit; the full timer isn't
  in our obs, so this doesn't matter.
- **Prev-action bootstrapping**: at episode start we have no prior action. Use zeros;
  this matches what GGL's `Player::prevAction` defaults to on `ResetBeforeStep`.
- **Kickoff no-op bias**: `bc_dataset.py:35` explicitly masks out the first frame of
  each episode because replays contain a ~1.5 s countdown where the human literally
  does nothing. Worth replicating — otherwise the BC actor learns to stand still at
  kickoff.
- **Dodge inference correctness**: verified numerically by `replay-pretraining`'s IDM
  paper; for first-cut BC we eat the ~5 % of frames where this is wrong.
