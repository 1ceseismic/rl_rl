#!/usr/bin/env python3
"""Sync checkpoints between machines.

Usage:
    python sync.py pull nyx          # pull latest checkpoint from nyx
    python sync.py push nyx          # push latest checkpoint to nyx
    python sync.py pull nyx --all    # pull entire run (all checkpoints)
"""
import os
import sys
import json
import subprocess

LOCAL_BASE = "agent_controllers_checkpoints/PPO1"
REMOTE_PROJECT = "~/Documents/code/rl_rl"


def list_runs(base_dir):
    #list runs with their latest checkpoint and timestep count
    runs = []
    for name in sorted(os.listdir(base_dir)):
        run_path = os.path.join(base_dir, name)
        if not os.path.isdir(run_path):
            continue
        checkpoints = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
        if not checkpoints:
            continue
        latest = str(max(checkpoints, key=lambda d: int(d)))
        latest_path = os.path.join(run_path, latest)
        agent_json = os.path.join(latest_path, "ppo_agent.json")
        ts = "?"
        if os.path.exists(agent_json):
            with open(agent_json) as f:
                ts = json.load(f).get("cumulative_timesteps", "?")
        runs.append((name, latest, latest_path, ts))
    return runs


def list_remote_runs(host):
    """List runs on a remote machine via SSH."""
    remote_base = f"{REMOTE_PROJECT}/{LOCAL_BASE}"
    script = f"""import os, json
base = os.path.expanduser("{remote_base}")
if not os.path.isdir(base): exit()
for name in sorted(os.listdir(base)):
    run_path = os.path.join(base, name)
    if not os.path.isdir(run_path): continue
    cks = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    if not cks: continue
    latest = str(max(cks, key=lambda d: int(d)))
    agent_json = os.path.join(run_path, latest, "ppo_agent.json")
    ts = "?"
    if os.path.exists(agent_json):
        with open(agent_json) as f: ts = json.load(f).get("cumulative_timesteps", "?")
    print(str(name) + "|" + str(latest) + "|" + str(ts))
"""
    result = subprocess.run(
        ["ssh", host, "python3"],
        input=script, capture_output=True, text=True
    )
    runs = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        name, latest, ts = parts
        ts = int(ts) if ts != "?" else "?"
        latest_path = f"{remote_base}/{name}/{latest}"
        runs.append((name, latest, latest_path, ts))
    return runs


def pick_run(runs, source_label):
    """Interactive run picker."""
    if not runs:
        print(f"  No runs found on {source_label}")
        return None

    latest = max(runs, key=lambda x: int(x[1]))
    print(f"\n{'='*60}")
    print(f"  Checkpoints on {source_label}")
    print(f"{'='*60}")
    for i, (name, ckpt, path, ts) in enumerate(runs, 1):
        marker = " << latest" if name == latest[0] else ""
        ts_str = f"{ts:,}" if isinstance(ts, int) else ts
        print(f"  [{i}] {name} ({ts_str} steps){marker}")
    print(f"{'='*60}")
    print(f"  Press Enter for latest ({latest[0]})")

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if choice == "":
        return latest
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass
    print("  Invalid choice")
    return None


def rsync(src, dst, label=""):
    """Run rsync with progress."""
    print(f"\n  Syncing {label}...")
    subprocess.run(["rsync", "-avz", "--progress", src, dst], check=True)
    print(f"  Done!\n")


def pull(host, sync_all=False):
    """Pull a checkpoint from remote to local."""
    runs = list_remote_runs(host)
    picked = pick_run(runs, host)
    if not picked:
        return

    name, latest, remote_path, ts = picked
    remote_base = f"{REMOTE_PROJECT}/{LOCAL_BASE}"

    if sync_all:  # pull entire run directory
        src = f"{host}:{remote_base}/{name}/"
        dst = f"{LOCAL_BASE}/{name}/"
        rsync(src, dst, f"{name} (all checkpoints)")

    else: # only latest checkpoint
        src = f"{host}:{remote_base}/{name}/{latest}/"
        dst = f"{LOCAL_BASE}/{name}/{latest}/"
        os.makedirs(dst, exist_ok=True)
        rsync(src, dst, f"{name}/{latest}")

    ts_str = f"{ts:,}" if isinstance(ts, int) else ts
    print(f"  Pulled: {name} ({ts_str} steps)")
    print(f"  Path:   {LOCAL_BASE}/{name}/{latest}")


def push(host, sync_all=False):  #push checkpoint from local to remote
    if not os.path.isdir(LOCAL_BASE):
        print("  No local checkpoints found")
        return

    runs = list_runs(LOCAL_BASE)
    picked = pick_run(runs, "local")
    if not picked:
        return

    name, latest, local_path, ts = picked
    remote_base = f"{REMOTE_PROJECT}/{LOCAL_BASE}"

    if sync_all:
        src = f"{LOCAL_BASE}/{name}/"
        dst = f"{host}:{remote_base}/{name}/"
        rsync(src, dst, f"{name} (all checkpoints)")
    else:
        src = f"{local_path}/"
        dst = f"{host}:{remote_base}/{name}/{latest}/"
        rsync(src, dst, f"{name}/{latest}")

    ts_str = f"{ts:,}" if isinstance(ts, int) else ts
    print(f"  Pushed: {name} ({ts_str} steps)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sync.py <pull|push> <host> [--all]")
        print("  pull nyx       Pull latest checkpoint from nyx")
        print("  push nyx       Push latest checkpoint to nyx")
        print("  --all          Sync all checkpoints in the run, not just latest")
        sys.exit(1)

    action = sys.argv[1]
    host = sys.argv[2]
    sync_all = "--all" in sys.argv

    if action == "pull":
        pull(host, sync_all)
    elif action == "push":
        push(host, sync_all)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
