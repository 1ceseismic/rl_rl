#!/usr/bin/env python3
# watch trained policy play in rlviser (not native bin) without training
import os
import time
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "1"

from rlgym_learn_algos.ppo import DiscreteFF
from checkpoint import pick_checkpoint
from train import build_env


def watch():
    checkpoint_path, _ = pick_checkpoint(allow_fresh=False)
    if not checkpoint_path:
        print("No checkpoint selected, nothing to watch.")
        return

    actor_path = os.path.join(checkpoint_path, "ppo_learner", "actor.pt")
    if not os.path.exists(actor_path):
        print(f"No actor.pt found at {actor_path}")
        return

    env = build_env()

    obs = env.reset()
    agents = list(obs.keys())
    obs_size = len(obs[agents[0]])
    action_size = 90  # LookupTableAction

    actor = DiscreteFF(obs_size, action_size, (256, 256, 256), "cpu")
    actor.load_state_dict(torch.load(actor_path, map_location="cpu", weights_only=True))
    actor.eval()

    print(f"\n  Watching: {os.path.dirname(os.path.dirname(checkpoint_path)).split('/')[-1]}")
    print(f"  Actor loaded from: {actor_path}")

    # game loop
    while True:
        try:
            obs_list = [obs[a] for a in agents]
            with torch.no_grad():
                actions, _ = actor.get_action(agents, obs_list)

            action_dict = {a: actions[i] for i, a in enumerate(agents)}
            obs, rewards, terminated, truncated = env.step(action_dict)
            env.render()

            # Reset if episode ended
            if any(terminated.values()) or any(truncated.values()):
                obs = env.reset()
                agents = list(obs.keys())

            time.sleep(8 / 120)  # match real-time (8 ticks at 120hz)

        except KeyboardInterrupt:
            print("\n  Stopped.")
            break


if __name__ == "__main__":
    watch()
