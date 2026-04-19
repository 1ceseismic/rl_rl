# IL Research, Part 1 — Acquiring and Parsing Rocket League Replays

Scope: what we need to feed an imitation-learning / behavioral-cloning warm-start
for the GGL agent. This document stays at the "bytes on disk → parsed
DataFrames" layer. Teammate (B) picks up from `ParsedReplay` and produces
`(obs, action)` pairs; teammate (C) owns the training loop.

Target pipeline, short version:

```
ballchasing.com API ──► .replay files (binary) ──► carball (Rust) ──► parquet ──► ParsedReplay (pandas)
```

Everything below is verified locally unless tagged `[community claim, unverified]`.

---

## 1. Replay sources

### 1a. ballchasing.com REST API

Primary source. Free public uploads searchable and downloadable by anyone with
an API key.

- Base URL: `https://ballchasing.com/api/`
- Auth: `Authorization: <token>` header. No "Bearer" prefix. Keys are issued
  from the Upload tab after logging in.
- Unauthenticated requests return HTTP 401 with
  `{"error": "missing API key", ...}` (verified: `curl -s
  https://ballchasing.com/api/` → 401).

**Rate limits (per ballchasing.com/doc/api):**

| Tier               | calls/sec | list endpoints/hour |
| ------------------ | --------- | ------------------- |
| Free / Bronze-Silver | 2       | 500                 |
| Gold patron          | 2       | 1000                |
| Diamond patron       | 4       | 2000                |
| Champion patron      | 8       | –                   |
| GC patron            | 16      | –                   |

File downloads (`GET /replays/{id}/file`) are stricter: 1–2 calls/sec depending
on tier. Plan for hours, not minutes, to pull a large corpus on free tier.

**Key endpoints for IL:**

- `GET /replays` — list. Filters: `playlist` (e.g. `ranked-duels`,
  `ranked-doubles`, `ranked-standard`), `season`, `match-result`, `min-rank`,
  `max-rank` (values: `bronze-1` … `grand-champion-3`, plus
  `supersonic-legend`), `player-name`, `player-id`, `uploader` (Steam ID or
  `me`), `created-before/after`, `replay-date-before/after` (RFC3339). Pagination:
  `count` (1–200, default 150), `sort-by` (`replay-date`|`upload-date`),
  `sort-dir`. The list payload gives replay IDs and metadata summaries, not
  the binary.
- `GET /replays/{id}` — full metadata including camera settings, per-player
  stats, etc. Optional when you only need the replay file.
- `GET /replays/{id}/file` — the raw `.replay` binary. This is what you feed to
  a parser.

**Python client:** `python-ballchasing` (Rolv-Arild, MIT). Installs via `pip
install python-ballchasing`. It handles pagination automatically (the raw API
caps at 200 per request) and exposes `Rank`, `Playlist`, `Season`, `Map`
enums so you don't hardcode magic strings. Minimal example:

```python
from ballchasing import BallchasingApi, Rank, Playlist
api = BallchasingApi("YOUR_TOKEN")

for rep in api.get_replays(
    min_rank=Rank.SUPERSONIC_LEGEND,
    playlist=[Playlist.RANKED_DUELS],
    count=10_000,                 # library paginates under the hood
    sort_by="replay-date",
    sort_dir="descending",
):
    blob = api.download_replay(rep["id"])        # bytes
    open(f"replays/{rep['id']}.replay", "wb").write(blob)
```

Alternative wrappers: `ballchaser` (pypi), `pychasing`
(github.com/tanrbobanr/pychasing). The Rolv-Arild one is what the RLGym-adjacent
code already uses, so prefer it for consistency.

### 1b. Alternatives / complements

- **Local Steam replay dir:** `~/.steam/steam/steamapps/compatdata/252950/pfx/
  drive_c/users/steamuser/Documents/My Games/Rocket League/TAGame/Demos/` on
  Linux-Proton; Windows equivalent is under Documents. Useful if the user
  records their own SSL-grade gameplay; not a viable corpus source.
- **RLCS / tournament replay packs:** occasionally dumped on Discord and on
  dedicated ballchasing groups (e.g. "RLCS Referee" account,
  Steam ID `76561199225615730`, queryable via `api.get_groups(creator=...)`).
  Top-quality data but small volume. `[community claim, unverified]` for
  quantity.
- **Kaggle — "High-Level Rocket League Replay Dataset"** by `rolvarild` — cited
  by the RLGym Discord as a convenient pre-curated set. Could not auto-verify
  exact count/size from the Kaggle page.

### 1c. Selection heuristics

What the RLGym community actually does for IL (from the replay-pretraining
README and Discord lore):

- **Rank filter:** Supersonic Legend only, or at minimum Grand Champion 3.
  Lower ranks inject noise the model will happily learn to copy.
- **Playlist:** match your training mode. The GGL run here is 2v2-ish? Pull
  `ranked-doubles` + optionally `ranked-duels` if training a 1s specialist.
  Mixing 3v3 with 1v1 hurts: the decision distribution shifts.
- **Modes:** standard Soccar only. Drop Rumble, Hoops, Dropshot, Snow Day —
  carball handles them but the physics/scoring differ enough that obs/reward
  assumptions break.
- **Game length:** keep replays where game duration is long enough and score
  looks sane. Short replays are often forfeits/early exits.

---

## 2. Replay parsers

Three candidates, in order of current community use:

### 2a. `rlgym-tools` bundled `carball` (Rust) — **RECOMMENDED**

`rlgym-tools 2.6.x` (active, v2.6.4 released 2026-04-10, maintained by the
RLGym core team — Rolv-Arild Braaten, Lucas Emery, Matthew Allen) ships a
prebuilt `carball` binary inside the package, for both Linux (ELF) and Windows
(`carball.exe`). This is a **Rust rewrite** of the original SaltieRL carball,
version `carball 0.1.1`, and it emits **parquet files directly**.

Install + use:
```bash
uv pip install --python .venv/bin/python rlgym-tools pyarrow pandas
```

Location after install (or in the cloned repo used for this research):
`rlgym_tools/rocket_league/replays/carball` (ELF, 16 MiB, chmod +x
automatically on first use by `parsed_replay.py`).

CLI verified locally:
```
carball [FLAGS] <data-frame-output-format> -i <input.replay> -o <output-dir>
  <data-frame-output-format>    Csv | Parquet
  --skip-analysis
  --skip-checks
  --skip-data-frames
  --skip-write-data-frames
```

End-to-end test run against
`ggl/docs/il-research/carball/carball/tests/replays/5_GOALS_6_KICKOFFS.replay`:
- Input: 584 KB `.replay`
- Output: 1.7 MB dir (`metadata.json`, `analyzer.json`, `__ball.parquet`,
  `__game.parquet`, `player_<uid>.parquet` per player)
- Wall time: sub-second for 9731 frames
- Emitted one player parquet (1 player in that test replay) with 39 columns.

Python wrapper — `rlgym_tools.rocket_league.replays.parsed_replay.ParsedReplay`:

```python
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay
rep = ParsedReplay.load("path/to/file.replay")  # also accepts a pre-parsed dir
rep.game_df      # pandas DataFrame, per-frame game state
rep.ball_df      # pandas DataFrame, per-frame ball
rep.player_dfs   # dict[unique_id -> DataFrame], per-frame per-player
rep.metadata     # dict — team, players, goals, demos, map
rep.analyzer     # dict — carball's higher-level stats
```

`ParsedReplay.load(path)` transparently runs the bundled carball binary into a
tempdir if given a `.replay` file, or loads existing parquet if given a
already-parsed directory. No extra wiring needed.

### 2b. Legacy `carball` (Python, SaltieRL) — **avoid**

`pypi.org/project/carball` last released 0.7.5 on 2020-10-14. Python 3.6.7–3.8
only. Depends on `boxcars-py==0.1.*` (whose latest wheel is `0.1.14` from
2021-12-02, built only for Python 3.7/3.8). Neither installs on Python 3.12;
building `boxcars-py` from source fails on current PyO3 (verified:
`uv pip install boxcars-py` → maturin build error on cp312).

Output format: JSON, protobuf, or gzipped pandas pickle (`--json`, `--proto`,
`--gzip` flags). Not parquet. If you must touch the old code, use a Python 3.8
env, but there is no reason to — the Rust rewrite above is a drop-in with
parquet and no C-extension hell.

### 2c. `boxcars` / `rrrocket` (Rust) — low-level

[nickbabcock/boxcars](https://github.com/nickbabcock/boxcars) is the Rust
replay parser library that underpins everything above. It exposes a fully
decoded replay with ~976 commits of history and active maintenance, fuzzed and
benchmarked at 100+ replays/sec/core.

[nickbabcock/rrrocket](https://github.com/nickbabcock/rrrocket) is the CLI
sibling: replay → JSON. Latest `v0.10.12` released 2025-03-11; prebuilt Linux
musl static binary available. Verified locally:

```bash
curl -sL https://github.com/nickbabcock/rrrocket/releases/download/v0.10.12/\
rrrocket-0.10.12-x86_64-unknown-linux-musl.tar.gz | tar -xz
rrrocket -n -p --dry-run some.replay     # network-parse + pretty + no write
```

Useful if you want to debug "what does the replay actually contain" at the
actor level, or to build a custom parser. For our IL pipeline it's overkill —
`rlgym-tools` carball already wraps it and gives us tabular frames.

### 2d. `rattletrap` (Haskell)

Maintained on Hackage, reference implementation, roundtrips replays losslessly.
Output JSON is ~2× the size of rrrocket's. No reason to use it unless you hit a
replay that boxcars can't parse. `[community claim, unverified]` that modern
season 5+ replays sometimes parse cleanly on rattletrap when boxcars barfs —
historically both authors cross-test against each other.

### Verdict

Use **`rlgym-tools` + its bundled carball** for everything below. It's what
the RLGym v2 ecosystem standardised on, includes the critical
`replay_to_rlgym` conversion helper in `convert.py` (teammate B's problem), and
the raw parquet output is directly loadable.

---

## 3. Parsed replay schema (from a real run)

Column lists below are from parsing the sample `5_GOALS_6_KICKOFFS.replay`
locally with the bundled carball binary.

### `__game.parquet` — (num_frames, 6)
```
time, delta, seconds_remaining, replicated_game_state_time_remaining,
is_overtime, ball_has_been_hit
```
`delta` is the per-frame dt in seconds. Replay ran at 26.97 fps mean (9731
frames over 371 s of game time). Replays are nominally 30 Hz but variable per
tick — don't assume a constant dt.

### `__ball.parquet` — (num_frames, 15)
```
is_sleeping, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
quat_w, quat_x, quat_y, quat_z, ang_vel_x, ang_vel_y, ang_vel_z, hit_team_num
```

### `player_<unique_id>.parquet` — (num_frames, 39)
```
# Physics (same layout as ball)
is_sleeping, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
quat_w, quat_x, quat_y, quat_z, ang_vel_x, ang_vel_y, ang_vel_z,

# Controller inputs — the IL target
throttle, steer, handbrake,                    # analogue floats (ffilled)
boost_is_active,                               # 0/1 tick flag
jump_is_active, double_jump_is_active,
flip_car_is_active, dodge_is_active,           # 0/1 tick flags
double_jump_torque_x/y/z, dodge_torque_x/y/z,  # dodge direction vectors

# Match stats (cumulative)
match_score, match_goals, match_assists, match_saves, match_shots,
team, ping, boost_amount, boost_pickup,
air_activate_count, dodges_refreshed_counter
```

### `metadata.json`
Top-level keys: `game`, `teams`, `players`, `demos`.
- `game`: `id`, `replay_version`, `num_frames`, `replay_name`, `map_name`,
  `date`, `match_type` ("Offline"/"Online"/"Ranked..."), `team_0_score`,
  `team_1_score`, `goals` (list of `{frame, player_name, is_orange}`).
- `players`: list of `{unique_id, name, online_id, online_id_kind ("Steam"/
  "Epic"/"PsyNet"), is_orange, car_id, match_score, match_goals,
  match_assists, match_saves, match_shots}`. `unique_id` is the key into
  `player_dfs`.
- `demos`: list of demo events with frame/attacker/victim.

### `analyzer.json`
Carball's higher-level analysis output — hits, possessions, boost efficiency,
kickoff frames. The replay-pretraining IDM pipeline uses the goal frames from
`metadata["game"]["goals"]` rather than this file, but it's there if you want
to split into possessions or tag kickoff/post-goal windows.

### Key gotchas

- **`is_sleeping` = NaN means demoed.** Actor is removed from the network
  stream during respawn. Both replay-pretraining and rlgym-tools `convert.py`
  treat NaN as "demoed" (see `replay-pretraining/replay_pretraining/replays/
  replays.py:143` — `is_demoed = player_df["is_sleeping"].isna()`).
- **Not all players appear in every replay directory.** If a player's network
  actor never gets fully resolved, the parquet may be missing. Code using
  `metadata["players"]` must filter by which `player_*.parquet` files actually
  exist.
- **Controller inputs are partially interpolated.** `throttle` and `steer` are
  analogue bytes from the network stream but get forward-filled between
  updates. See `rlgym_tools/.../convert.py:316-318`:
  ```python
  pdf[["throttle", "steer"]] = pdf[["throttle", "steer"]].ffill().fillna(255/2)
  pdf.loc[is_repeat, ["throttle", "steer"]] = np.nan
  pdf[["throttle", "steer"]] = pdf[...].interpolate()...
  ```
  `is_repeat` marks frames where the actor wasn't freshly updated. This is
  why the RLGym replay-pretraining approach wraps an **Inverse Dynamics Model
  (IDM)** around the data rather than training BC directly on the raw
  controller columns.
- **Jumps/dodges are tick-accurate but not directly equal to button presses.**
  `jump_is_active` flips to 1 for the duration the jump action is live, not
  only on the rising edge. `convert.py:344` derives the actual "button
  pressed" moment via `diff() > 0`.
- **Pitch/yaw/roll are not in the replay.** Only angular velocity is. Convert
  via `rlgym_tools.../convert.py` (`predict_pyr=True`, default) which uses the
  quaternion derivative + torque vectors to reconstruct the stick inputs.

### Tick rate vs. our 15 Hz decisions

Replays are ~30 Hz native. Our GGL env runs at 120 Hz sim with `tick_skip=8`,
i.e. **15 Hz decisions**. That's a clean 2:1 downsample from the replay side
with `rlgym_tools.replays.convert.replay_to_rlgym(rep,
interpolation="rocketsim")` — it re-steps RocketSim between replay frames so
you get physically-consistent intermediate states rather than naive linear
interp. This is teammate B's problem to wire up; the parsing side here just
delivers ParsedReplay at 30 Hz.

---

## 4. Practical corpus advice

### How many replays?

- Rolv-Arild's replay-pretraining trained on **~thousands of SSL replays** to
  produce Ripple-like behaviour (exact number not stated in README; trained
  initially on ~tens of millions of frames equivalent). `[community claim,
  unverified for exact number]`
- Rough lower bound from community lore: **1 000 SSL replays** gives a usable
  BC init. **10 000+** is where you stop being bottlenecked by data and start
  being bottlenecked by the IDM quality. `[community claim, unverified]`
- Per replay: ~5 min game time × 30 fps ≈ 9 000 frames. 10 000 replays ≈
  **90 M frames**. At 2 players per replay for 1s, or 4 for 2s, that's
  180 M – 360 M (state, action) rows.

### Storage footprint

Verified on the sample replay above:
- `.replay` binary: **584 KB**
- Parsed parquet dir: **1.7 MB** (~3× expansion)
- Uncompressed in-memory pandas: larger still; plan on ~5× raw replay size.

For 10 000 replays: roughly **6 GB** of raw `.replay` files, **17 GB** of
parsed parquet on disk, maybe **50+ GB** once you materialise the RLGym
tensor-friendly (obs, action) arrays. This fits on a single commodity SSD.

Scaling guidance: **don't store a bigger intermediate than you need.** Keep
the raw `.replay` files (you can re-parse on demand), parse to parquet in a
cached dir, and for the actual BC pipeline let teammate C stream (obs, action)
batches straight from parquet with a Torch `IterableDataset` rather than
pre-materialising numpy files. The replay-pretraining repo goes the opposite
way — pre-materialised numpy — but that was written in the RLGym v1 era and
is memory-hungry.

### On-disk format

**Use parquet.** That's what `rlgym-tools`' carball already emits, and what the
replay-pretraining code reads via `pd.read_parquet`. Columnar, compresses
well, supports filter-pushdown if you use `pyarrow.dataset`. Pickle is
brittle across pandas versions; HDF5 is legacy-heavy; npz loses schema.

For datasets too big for one machine, use `pyarrow.dataset` with a partitioned
layout:
```
corpus/
  rank=SSL/playlist=duels/season=15/<replay_id>/{metadata.json, __game.parquet, ...}
  rank=SSL/playlist=duels/season=15/<replay_id>/...
```
— lets you `filter=` at read time without scanning the whole tree.

### Filtering — drop these

Minimal QA before any replay enters the corpus:

- **Early disconnect / forfeit.** Check
  `metadata["game"]["match_type"]` plus number of frames. Under ~30 s of
  gameplay → drop. `num_frames < 5000` is a reasonable hard floor for
  standard 5-minute games.
- **Missing player parquets.** If `len(player_dfs) < expected_players` (2 for
  duels, 4 for doubles, 6 for standard) → drop. Some players' actors fail to
  resolve in the network stream.
- **Non-standard map.** `metadata["game"]["map_name"]` must be a Soccar
  standard variant (`Stadium_P`, `NeoTokyo_Standard_P`,
  `UtopiaStadium_P`, `Park_P`, `TrainStation_P`, `CS_Day_P`, etc. — the
  "_P" suffix is the Psyonix suffix, and "Standard" or just plain map names
  without "_Hoops/Snow/Basketball" are safe). Community lists exist;
  `rlgym-tools` has this filter baked into `replay_to_rlgym` error paths.
- **Rank mismatch.** Trust the ballchasing filter, but also double-check
  `metadata` per-replay: some replays uploaded to SSL queues include one
  smurf + three Champ players. `[community claim, unverified for the specific
  field ballchasing exposes]` — if in doubt, filter by the uploader's rank at
  upload time via the `/replays/{id}` detail endpoint.
- **AFK / boost-starved 0-0.** Heuristic: if
  `ball_df[["vel_x","vel_y","vel_z"]].abs().sum(1).mean()` is near-zero over a
  long span, one side went AFK. Rare post-rank-filter.
- **Replay parser errors.** `ParsedReplay.load` raises `ValueError` carrying
  the carball stderr. Log these and move on; expect ~1–5 % failure rate on
  a random sample, biased toward old/patched-out seasons.

### Recommended directory layout for our project

```
ggl/data/
  replays/                          # raw .replay, 584 KB avg
    <replay_id>.replay
  parsed/                           # carball parquet, 1.7 MB avg
    <replay_id>/
      metadata.json
      analyzer.json
      __ball.parquet
      __game.parquet
      player_<uid>.parquet
  manifest.parquet                  # our own index: replay_id, rank,
                                    # playlist, num_frames, parse_ok, ...
```

Teammate (B) reads from `parsed/` and produces the `(obs, action)` tensors; we
never re-materialise those to disk unless profiling shows the conversion is
the bottleneck.

---

## 5. Cheatsheet — the minimal IL scraping pipeline

```python
# One-time setup
# uv pip install --python .venv/bin/python rlgym-tools python-ballchasing pyarrow pandas

from pathlib import Path
from ballchasing import BallchasingApi, Rank, Playlist
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay

api = BallchasingApi(open("ballchasing.token").read().strip())

OUT = Path("ggl/data"); (OUT/"replays").mkdir(parents=True, exist_ok=True); (OUT/"parsed").mkdir(exist_ok=True)

for rep in api.get_replays(
    min_rank=Rank.SUPERSONIC_LEGEND,
    playlist=[Playlist.RANKED_DOUBLES],
    count=5_000,
):
    rid = rep["id"]
    raw = OUT/"replays"/f"{rid}.replay"
    if not raw.exists():
        raw.write_bytes(api.download_replay(rid))

    parsed_dir = OUT/"parsed"/rid
    if not parsed_dir.exists():
        try:
            ParsedReplay.load(raw)  # parses into tempdir; call with parsed_dir to persist
        except Exception as e:
            print(f"parse failed {rid}: {e}")
```

(Persisting the parsed dir permanently needs calling the bundled
`process_replay(raw, OUT/"parsed", skip_existing=True)` directly from
`rlgym_tools.rocket_league.replays.parsed_replay` — `ParsedReplay.load` uses a
tempdir when given a raw replay. For corpus building we want the latter;
minor wrapper code, not worth inlining here.)

---

## References

- ballchasing API docs: https://ballchasing.com/doc/api
- ballchasing FAQ: https://ballchasing.com/doc/faq
- `python-ballchasing`: https://github.com/Rolv-Arild/python-ballchasing
- `rlgym-tools` (replay parsing + convert): https://github.com/RLGym/rlgym-tools
  — `rlgym_tools/rocket_league/replays/{parsed_replay,convert,replay_frame}.py`
- `carball` (original Python): https://github.com/SaltieRL/carball
- `boxcars` (Rust core): https://github.com/nickbabcock/boxcars
- `rrrocket` (Rust CLI): https://github.com/nickbabcock/rrrocket
- `rattletrap` (Haskell): https://github.com/tfausak/rattletrap
- `replay-pretraining` (reference IDM+BC pipeline):
  https://github.com/Rolv-Arild/replay-pretraining
- Kaggle SSL corpus:
  https://www.kaggle.com/datasets/rolvarild/high-level-rocket-league-replay-dataset
- VPT paper (OpenAI), the conceptual model for IDM-based IL:
  https://openai.com/blog/vpt/

Local clones of the above sit under
`ggl/docs/il-research/{rlgym-tools,carball,boxcars-py,python-ballchasing,replay-pretraining}/`
for easy grepping.
