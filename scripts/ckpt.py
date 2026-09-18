#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Run:
    name: str
    path: Path
    latest_ts: int
    latest_path: Path


def discover_runs(root: Path) -> list[Run]:
    if not root.is_dir():
        return []
    out: list[Run] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        ts_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not ts_dirs:
            continue
        latest = max(ts_dirs, key=lambda d: int(d.name))
        out.append(Run(
            name=run_dir.name,
            path=run_dir,
            latest_ts=int(latest.name),
            latest_path=latest,
        ))
    out.sort(key=lambda r: r.latest_ts)
    return out


def prompt(msg: str, default: str = "") -> str:
    print(msg, end="", file=sys.stderr, flush=True)
    try:
        return input().strip() or default
    except (EOFError, KeyboardInterrupt):
        return default


def pick(runs: list[Run], *, allow_fresh: bool) -> Run | None | str:
    hostname = socket.gethostname()
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"  GGL Checkpoint Selection ({hostname})", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    if allow_fresh:
        print("  [0] Fresh start (new run)", file=sys.stderr)
    default = runs[-1] if runs else None
    for i, r in enumerate(runs, 1):
        marker = " << latest" if r is default else ""
        print(f"  [{i}] {r.name}  ({r.latest_ts:,} steps){marker}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    if default:
        print(f"  Enter = latest ({default.name})", file=sys.stderr)
    elif not allow_fresh:
        print("  no runs found", file=sys.stderr)
        return "quit"

    choice = prompt("  > ")
    if choice == "" and default:
        return default
    if allow_fresh and (choice == "0" or (choice == "" and not default)):
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    print("  invalid, using latest", file=sys.stderr)
    return default


def train_mode(root: Path) -> str:
    runs = discover_runs(root)
    picked = pick(runs, allow_fresh=True)
    hostname = socket.gethostname()

    if picked is None:
        default_name = f"{hostname}-{datetime.now():%m%d-%H%M}"
        name = prompt(f"  Run name? (Enter for '{default_name}')\n  > ", default_name)
        print(f"  -> fresh run: {name}\n", file=sys.stderr)
        return name

    default_name = picked.name
    name = prompt(f"  Run name? (Enter to continue '{default_name}')\n  > ", default_name)
    if name == default_name:
        print(f"  -> continuing {name} ({picked.latest_ts:,} steps)\n", file=sys.stderr)
        return name

    new_run = root / name
    if new_run.exists():
        print(f"  '{name}' exists, continuing it", file=sys.stderr)
        return name
    dst = new_run / picked.latest_path.name
    print(f"  -> branching: copying {picked.name}/{picked.latest_path.name} -> {name}/", file=sys.stderr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(picked.latest_path, dst)
    print(f"  done.\n", file=sys.stderr)
    return name


def watch_mode(root: Path) -> str | None:
    runs = discover_runs(root)
    picked = pick(runs, allow_fresh=False)
    if picked is None or picked == "quit":
        return None
    print(f"  -> watching {picked.name} ({picked.latest_ts:,} steps)\n", file=sys.stderr)
    return picked.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "build-ggl" / "checkpoints_ggl")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--train", action="store_true")
    grp.add_argument("--watch", action="store_true")
    args = ap.parse_args()

    if args.train:
        name = train_mode(args.root)
    else:
        name = watch_mode(args.root)

    if not name:
        return 1
    print(f"RUN_NAME={name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
