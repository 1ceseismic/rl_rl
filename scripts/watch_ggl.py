#!/usr/bin/env python3
"""Watch the latest GGL checkpoint in rlviser.

GGL writes checkpoints to checkpoints_ggl/<timestep>/. This script runs
rl_rl_ggl with GGL_RENDER=1 and (eventually) a load-from-checkpoint flag.

The leaked GGL source does not currently expose a "load-and-render-only" mode
from the CLI — Learner::Start() always begins a training iteration after load.
If you just want to watch, run with GGL_RENDER=1 and a very small numGames
count to minimize training while the checkpoint plays through rlviser.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "build-ggl" / "rl_rl_ggl"
CKPT_DIR = REPO / "checkpoints_ggl"


def latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: int(p.name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", default=os.environ.get("GGL_MESH_DIR", str(REPO / "collision_meshes")))
    parser.add_argument("--num-games", type=int, default=2, help="Low count keeps rlviser readable.")
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"[watch_ggl] binary missing: {BINARY}. Run scripts/build_ggl.sh first.", file=sys.stderr)
        return 1

    ckpt = latest_checkpoint(CKPT_DIR)
    if ckpt:
        print(f"[watch_ggl] latest checkpoint: {ckpt.name}")
    else:
        print(f"[watch_ggl] no checkpoints found in {CKPT_DIR} — will start from scratch.")

    env = os.environ.copy()
    env["GGL_RENDER"] = "1"
    env["GGL_NUM_GAMES"] = str(args.num_games)
    env["GGL_MESH_DIR"] = args.mesh_dir
    env["GGL_RUN_NAME"] = env.get("GGL_RUN_NAME", "watch")

    return subprocess.call([str(BINARY)], env=env, cwd=str(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
