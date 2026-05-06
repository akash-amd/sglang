#! /bin/bash

# make log file directory if not exists
mkdir -p logs

# log file name with timestamp
LOG_FILE="logs/sglang_flash-base_$(date +%Y%m%d_%H%M%S).log"

PYTHONPATH=/_alias:${PYTHONPATH:-} \
  PORT=30010 \
  MODEL=/hf/DeepSeek-V4-Flash-Base \
  bash /sgl-pr/launch_dsv4.sh stacked-best 2>&1 | tee $LOG_FILE