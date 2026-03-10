import os
import json
import shutil
import socket


def pick_checkpoint(base_dir="agent_controllers_checkpoints/PPO1", allow_fresh=True):
    """Interactive checkpoint picker. Returns (checkpoint_path, parent_name) or (None, None)."""
    hostname = socket.gethostname()

    # One entry per run, using the latest checkpoint
    runs = []
    if os.path.isdir(base_dir):
        for run in sorted(os.listdir(base_dir)):
            run_path = os.path.join(base_dir, run)
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
            runs.append((run, latest, latest_path, ts))

    # Sort by timesteps
    runs.sort(key=lambda x: x[3] if isinstance(x[3], int) else 0)
    default = runs[-1] if runs else None

    print(f"\n{'='*60}")
    print(f"  Checkpoint Selection ({hostname})")
    print(f"{'='*60}")
    if allow_fresh:
        print(f"  [0] Fresh start (no checkpoint)")
    for i, (run, ckpt, path, ts) in enumerate(runs, 1):
        marker = " << latest" if default and path == default[2] else ""
        ts_str = f"{ts:,}" if isinstance(ts, int) else ts
        print(f"  [{i}] {run} ({ts_str} steps){marker}")
    print(f"{'='*60}")
    if default:
        ts_str = f"{default[3]:,}" if isinstance(default[3], int) else default[3]
        print(f"  Press Enter for latest ({default[0]})")
    else:
        print(f"  No checkpoints found")

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if (choice == "0" and allow_fresh) or (choice == "" and not default):
        print(f"  -> Fresh start")
        return None, None
    elif choice == "" and default:
        ts_str = f"{default[3]:,}" if isinstance(default[3], int) else default[3]
        print(f"  -> {default[0]} ({ts_str} steps)")
        return default[2], default[0]
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs):
                picked = runs[idx]
                ts_str = f"{picked[3]:,}" if isinstance(picked[3], int) else picked[3]
                print(f"  -> {picked[0]} ({ts_str} steps)")
                return picked[2], picked[0]
        except ValueError:
            pass
        if default:
            print(f"  -> Invalid choice, using latest")
            return default[2], default[0]
        return None, None


def select_checkpoint(base_dir="agent_controllers_checkpoints/PPO1"):
    """Full selection for training: pick checkpoint + name the run. Returns (checkpoint_path, run_name, parent)."""
    hostname = socket.gethostname()
    checkpoint_path, parent = pick_checkpoint(base_dir)

    suggestion = parent if parent else "fresh-run"
    print(f"\n  Run name? (Enter for '{suggestion}')")
    try:
        name = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    if not name:
        name = suggestion

    run_name = f"{name}-{hostname}"
    print(f"  -> Run: {run_name}\n")

    # If branching from a checkpoint with a new name, copy it into a new directory
    if checkpoint_path and parent and name != parent:
        new_run_dir = os.path.join(base_dir, run_name)
        # Copy latest checkpoint into new run directory
        ckpt_name = os.path.basename(checkpoint_path)
        new_ckpt_path = os.path.join(new_run_dir, ckpt_name)
        print(f"  Copying checkpoint to {new_run_dir}/...")
        shutil.copytree(checkpoint_path, new_ckpt_path)

        # Clear wandb run ID so it creates a fresh wandb run
        wandb_json = os.path.join(new_ckpt_path, "metrics_logger", "wandb_metrics_logger.json")
        if os.path.exists(wandb_json):
            with open(wandb_json, "w") as f:
                json.dump({}, f)
        print(f"  (New run directory created, wandb ID cleared)\n")
        checkpoint_path = new_ckpt_path

    return checkpoint_path, run_name, parent
