# rl_rl

rocket league self-play bot. started in python with rlgym-learn, now using gigalearncpp + libtorch.

## throughput

rtx 4070 super, 24 cores, 1v1 soccar, overall steps/sec. 3 reps each, ~1% noise:

| setup            | default | GGL_INFER_GRAPH=1 |
|------------------|---------|-------------------|
| python, 36 procs | ~30k    | -                 |
| c++, 36 arenas   | 152k    | 221k              |
| c++, 192 arenas  | 270k    | 283k              |

cuda graphs cut policy inference ~60%, so they matter most at low arena counts.
more arenas means a bigger inference batch, but shorter per-env trajectory slices per iteration.

## run

```
scripts/go              train
scripts/go -w           watch in rlviser
scripts/go -h           flags
GGL_NUM_GAMES=192       more arenas
GGL_INFER_GRAPH=1       cuda graph inference
```

## build

needs libtorch and collision meshes in ./collision_meshes (from RLArenaCollisionDumper).

```
scripts/go build
```
