import os
import json
import socket


def pick_checkpoint(base_dir="agent_controllers_checkpoints/PPO1", allow_fresh=True):
    """Interactive checkpoint picker. Returns (checkpoint_path, parent_name) or (None, None)."""
    hostname = socket.gethostname()

    run_info = []
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
            run_info.append((run, latest, latest_path, ts))

    default = max(run_info, key=lambda x: int(x[1])) if run_info else None

    print(f"\n{'='*60}")
    print(f"  Checkpoint Selection ({hostname})")
    print(f"{'='*60}")
    if allow_fresh:
        print(f"  [0] Fresh start (no checkpoint)")
    for i, (run, ckpt, path, ts) in enumerate(run_info, 1):
        marker = " << latest" if default and run == default[0] else ""
        ts_str = f"{ts:,}" if isinstance(ts, int) else ts
        print(f"  [{i}] {run} ({ts_str} steps){marker}")
    print(f"{'='*60}")
    if default:
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
        print(f"  -> {default[0]}")
        return default[2], default[0]
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(run_info):
                picked = run_info[idx]
                print(f"  -> {picked[0]}")
                return picked[2], picked[0]
        except ValueError:
            pass
        if default:
            print(f"  -> Invalid choice, using {default[0]}")
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

    return checkpoint_path, run_name, parent
