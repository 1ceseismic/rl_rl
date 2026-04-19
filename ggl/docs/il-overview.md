# Imitation Learning for GGL — Overview

Consolidated reference for doing BC (behavior-cloning) warm-start of the GGL
trainer from pro Rocket League replays. Detailed deep-dives in
`ggl/docs/il-research/0{1,2,3}-*.md` — this page is the at-a-glance map:
what to build, in what order, how much work, what the concrete numbers are.

## Pipeline shape

```
   ballchasing.com REST API
              │  (auth; ~1-2 req/s free tier)
              ▼
      .replay binary files              ← 01-scraping-parsing.md §1
              │
              ▼
   rlgym-tools' bundled carball         ← 01-scraping-parsing.md §2
   (Rust, emits parquet directly)
              │
              ▼
      ParsedReplay (parquet)            ← 01-scraping-parsing.md §3
      __game, __ball, player_<uid>
              │
              │  (30 Hz replay → 15 Hz decisions, quat → rotMat,
              │   orange-team InvertPhys, continuous controls →
              │   argmin-L2 into 90-entry lookup table)
              ▼                         ← 02-conversion.md §1-3
   (obs: R^89, action_idx: int) pairs
              │
              ▼
   sharded .npz on disk (x_data, y_data, ep_ends)
              │                         ← 02-conversion.md §5
              ▼
   Python BC training loop              ← 03-bc-training-warmstart.md §1
   (CE over 90 classes, Adam 5e-5)
              │
              ▼
   torch.jit.script(net).save('POLICY.lt')
              │                         ← 03-bc-training-warmstart.md §5
              ▼
   build-ggl/checkpoints_ggl/bc-warm/0/
     POLICY.lt, CRITIC.lt (random), RUNNING_STATS.json
              │
              ▼
   GGL Learner::Load() auto-picks it    ← 03-bc-training-warmstart.md §5
              │
              ▼
   Continue PPO with guidingPolicy      ← 03-bc-training-warmstart.md §3 Option C
   regularizer pointing at a frozen
   copy of the same POLICY.lt
```

## Key concrete numbers (verified)

| Quantity | Value | Source |
|---|---|---|
| Obs dim (1v1) | **89** floats | Inspected live `POLICY.lt` — `Linear(89, 512)` first layer |
| Obs dim (2v2) | 131 | `9 + 8 + 34 + 20 + 20·(n_players−1)` formula |
| Action count | **90** discrete | `DefaultAction.cpp` — 24 ground + 66 aerial |
| Action idx 0 | **NOT no-op** — it's `[-1,-1,0,-1,0,0,0,0]` | `DefaultAction.cpp` |
| RL decision rate | 15 Hz | 120 Hz sim × `tickSkip=8` |
| Replay native rate | ~30 Hz, variable | carball `delta` column |
| Replay size on disk | ~584 KB raw | verified locally |
| Parsed replay size | ~1.7 MB parquet | verified locally |
| 10k-replay corpus | ~6 GB raw, ~17 GB parsed, ~50 GB materialized | projected |
| BC LR | 5e-5 Adam | Rolv-Arild `replay-pretraining` |
| BC batch | 300 | same |
| BC schedule | `LambdaLR(1/(0.25·e + 1))` | same |
| Val top-1 accuracy target | 20-30% | community claim |
| Val top-5 accuracy target | 50-60% | community claim |

## Critical gotchas (don't skip these)

1. **Pitch/yaw/roll are NOT in the replay** — only ang_vel is. Reconstruct via
   `rlgym_tools.../convert.py` with `predict_pyr=True` (uses quaternion
   derivative + dodge torque vectors). *(01-§3, 02-§3)*
2. **Controller inputs are forward-filled.** `throttle`/`steer` get repeated
   across stale physics ticks. Drop stale rows with the physics-diff filter
   from `bc_dataset.py`. *(01-§3, 03-§1)*
3. **Kickoff no-op filter is mandatory.** ~1.5s of "do nothing" at game start;
   if not filtered, the BC bot sits still on kickoff. *(03-§1)*
4. **Boost pad availability comes from `boost_pickup` events**, not
   directly — reconstruct the 4s/10s cooldowns per pad. *(02-§2)*
5. **Orange team inversion**: GGL's `DefaultObs` applies `InvertPhys`
   (180° rotation about Z) to ball + all players when agent is orange, plus
   permutes the 34 boost pads through a precomputed index. Build both
   perspectives per replay to double your dataset size. *(02-§2)*
6. **Dodges need special-case matching.** Continuous→discrete argmin fails on
   dodge frames because the human's stick direction determines the dodge
   torque at *start* of the dodge, then a fixed pulse runs for ~0.15s. Read
   `dodge_torque_x/y/z` and match stick direction during `dodge_is_active`
   frames. Skip this for a first cut (~5% of frames hit the wrong target,
   acceptable noise). *(02-§3)*
7. **Checkpoint archive format risk.** Python `torch.jit.script(seq).save()`
   and C++ `torch::save(seq)` have matching param keys but different
   top-level module class names. The load *should* succeed because libtorch's
   `torch::load(seq&, stream)` walks params by dotted-path name (e.g.
   `0.weight`) — but this is unverified without running C++. Fallback is a
   one-shot C++ helper that copies state_dict into a `GGL::Model` and calls
   its `Save()`. *(03-§3, §5)*

## Decision points

| Question | Recommended first-pass | Alternative |
|---|---|---|
| Where to get replays? | ballchasing.com API @ SSL rank + ranked-doubles | Kaggle "High-Level Rocket League Replay Dataset" (Rolv-Arild) |
| How many replays? | **1k SSL** (usable BC), **10k** (data-bottlenecked) | >10k hits diminishing returns without IDM |
| Parser? | `rlgym-tools` bundled carball (Rust) | avoid legacy Python carball — won't install on 3.12 |
| Tick alignment? | Take every 2nd replay frame (30 Hz → 15 Hz) | RocketSim re-simulation — higher fidelity, ~10× slower, fails on goal cutscenes |
| Action in the 8-tick window? | First tick's controls | Majority-over-8 / IDM (replay-pretraining approach) |
| Dodge handling (first pass)? | Skip — eat the ~5% error | Special-case via `dodge_torque_x/y/z` when `dodge_is_active` |
| Obs perspective augmentation? | Blue + mirrored orange (×2 dataset) | Single-perspective only |
| BC → PPO handoff? | Option A (build in Python, save via `torch.jit.script`) + **Option C** (guidingPolicy regularizer) | Option B (train only final head) |
| Critic at warm-start? | Random — freeze actor (`cfg.ppo.policyLR=0`) for ~1M steps to let critic catch up | BC target from replay discounted-reward — much harder |
| `guidingStrength`? | 0.03 (GGL default), hold flat or manually anneal to 0 over ~50M steps | |

## Effort estimate (fresh-start to BC-warm bot)

| Phase | Wall-clock |
|---|---|
| ballchasing scrape (5k SSL replays, free tier) | 1 day (mostly waiting; ~1-2 downloads/sec) |
| carball parse pass | 1-2 hours unattended |
| Conversion pipeline (obs builder + action matcher + shard writer) | 1-2 days of coding + verification |
| BC training | ~1 hour on GPU |
| GGL checkpoint packaging + load validation | ~half a day |
| First training run + sanity-check in rlviser | the rest |
| **Total** | **~1 week** for first-pass BC warm-started bot, excluding training time itself |

## GGL hooks that already exist for this

- **`cfg.ppo.useGuidingPolicy` + `guidingPolicyPath` + `guidingStrength`**
  (`PPOLearnerConfig.h:55-57`, `PPOLearner.cpp:224-235`) —
  applies L1 prob-diff regularizer every minibatch against a frozen reference
  policy. Use for BC-regularized PPO.
- **`Learner::StartTransferLearn()` + `TransferLearnConfig::useKLDiv`**
  (`TransferLearnConfig.h`) — true KL distillation mode, separate from `Start()`.
  Useful if we ever need to map between old and new obs/action spaces.
- **`cfg.ppo.policyLR = 0`** (`PPOLearner.cpp:161-164`) — cleanly freezes
  actor; enables critic-warm-up phase.
- **Per-run checkpoint layout** — `checkpoints_ggl/<run-name>/<ts>/` (see
  our `scripts/go` picker). BC checkpoints drop in as a synthetic
  `checkpoints_ggl/bc-warm/0/` directory; Learner loads the highest-numbered
  subdir automatically.

## When to actually do this

Probably not yet. The current baseline (GGL defaults) hasn't been trained to
convergence, and IL is a *force multiplier* for runs that already work — it
doesn't fix reward-design issues. Revisit when:

1. The default-reward bot has been trained to convergence (probably
   ~100-300M steps) and you can quantify its skill (ELO graph from
   `Rating/1v1` is the metric).
2. You've decided to keep your obs/action spaces stable for a while. An IL
   corpus is invalidated by any schema change, and re-generating it is a week.
3. You want to skip the "learn basic ball contact" phase rather than
   investigate new reward functions.

Until then: the three deep-dive docs are the reference. Scraping can start
whenever — it's just waiting on the API and costs nothing.

## Canonical reference repos (cloned under `ggl/docs/il-research/`)

- `rlgym-tools` — official rlgym toolkit, provides `ParsedReplay` and the
  bundled Rust carball binary. This is the standard entry point.
- `carball` — legacy Python (avoid), cloned for reference only.
- `boxcars-py` — low-level Rust parser, cloned for schema reference.
- `python-ballchasing` — Rolv-Arild's official ballchasing client.
- `replay-pretraining` — Rolv-Arild's reference IDM + BC pipeline. Closest
  public equivalent to what we want; obs/action layouts differ from our
  DefaultObs so we reuse utilities, not the full pipeline.
