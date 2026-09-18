from __future__ import annotations

import os
import sys

import wandb

wandb_run = None


def init(py_exec_path, project, group, name, id=None):
    global wandb_run

    sys.executable = py_exec_path  # avoid re-exec of binary

    entity = os.environ.get("WANDB_ENTITY") or None
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
    except BaseException:  # stray KeyboardInterrupt
        import traceback
        print("metric_receiver: log failed, continuing:")
        traceback.print_exc()
