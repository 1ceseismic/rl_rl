"""wandb metric receiver for GGL.

Called by GGL's MetricSender (C++) via pybind11. The init() signature is
fixed by GGL: (py_exec_path, project, group, name, id). We extend behavior
by reading WANDB_ENTITY from the environment so the project's wandb account
(rl_rlbot) is used without needing to patch C++.

Replaces GGL's default receiver — copied into build-ggl/python_scripts/ at
build time. See scripts/build_ggl.sh.
"""
from __future__ import annotations

import os
import sys

import wandb

wandb_run = None


def init(py_exec_path, project, group, name, id=None):
    """Start or resume a wandb run. Returns the wandb run ID."""
    global wandb_run

    # GGL's fix for its own interpreter shadowing — copied from upstream.
    # Without this, wandb can recurse back into the training binary.
    sys.executable = py_exec_path

    entity = os.environ.get("WANDB_ENTITY") or None  # wandb treats "" as invalid
    init_kwargs = dict(project=project, group=group, name=name)
    if entity:
        init_kwargs["entity"] = entity
    if id:
        init_kwargs["id"] = id
        init_kwargs["resume"] = "allow"

    print(f"metric_receiver: wandb.init({init_kwargs})")
    wandb_run = wandb.init(**init_kwargs)
    return wandb_run.id


def add_metrics(metrics):
    global wandb_run
    if wandb_run is None:
        return
    try:
        wandb_run.log(metrics)
    except BaseException:
        # Never crash training on a metrics hiccup (network blip, stray
        # SIGINT from the terminal turned into KeyboardInterrupt by the
        # embedded Python, etc.). wandb will catch up on the next log().
        import traceback
        print("metric_receiver: log failed, continuing:")
        traceback.print_exc()
