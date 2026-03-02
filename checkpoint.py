import os
import json
import socket


def select_checkpoint(base_dir="agent_controllers_checkpoints/PPO1"):
    """Interactive checkpoint selection. Returns (checkpoint_path, run_name, parent_checkpoint)."""
    hostname = socket.gethostname()

    # Collect all runs and their latest checkpoints with timestep counts
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
    print(f"  [0] Fresh start (no checkpoint)")
    for i, (run, ckpt, path, ts) in enumerate(run_info, 1):
        marker = " << latest" if default and run == default[0] else ""
        ts_str = f"{ts:,}" if isinstance(ts, int) else ts
        print(f"  [{i}] {run} ({ts_str} steps){marker}")
    print(f"{'='*60}")
    if default:
        print(f"  Press Enter for auto-resume ({default[0]})")
    else:
        print(f"  No checkpoints found, press Enter for fresh start")

    try:
        choice = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    # Determine checkpoint
    checkpoint_path = None
    parent = None
    if choice == "0" or (choice == "" and not default):
        pass  # fresh start
    elif choice == "" and default:
        checkpoint_path = default[2]
        parent = default[0]
        print(f"  -> Auto-resuming from {parent}")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(run_info):
                picked = run_info[idx]
                checkpoint_path = picked[2]
                parent = picked[0]
                print(f"  -> Resuming from {parent}")
            else:
                raise ValueError
        except ValueError:
            if default:
                checkpoint_path = default[2]
                parent = default[0]
                print(f"  -> Invalid choice, auto-resuming from {parent}")
            else:
                pass

    # Ask for run name
    if checkpoint_path:
        suggestion = parent
    else:
        suggestion = "fresh-run"
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
