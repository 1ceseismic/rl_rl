#!/bin/bash
rsync -avz \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='agent_controllers_checkpoints' \
  --exclude='.git' \
  --exclude='.claude' \
  --exclude='config.json' \
  --exclude='shmem_flinks' \
  --exclude='rlviser' \
  --exclude='settings.txt' \
  --include='*.py' \
  --include='*.sh' \
  --include='.gitignore' \
  --exclude='*' \
  /home/seis/code/rl_rl/ nyx:~/Documents/code/rl_rl/
