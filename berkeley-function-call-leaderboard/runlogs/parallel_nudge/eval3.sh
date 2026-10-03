#!/usr/bin/env bash
# Probe C: neutral system line, no instruction about how many calls to make.
# Same slot as probe B (before "# Tools"), so only the wording differs.
# Usage: eval3.sh <model-key> <port>
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
PY="$B/.venv/bin/python"
MODEL="$1"
export LOCAL_SERVER_PORT="$2" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=40960
export BFCL_SYSTEM_PREFIX="You are a helpful assistant."
CATS=parallel,parallel_multiple
echo "[$(date '+%F %T')] generating: $MODEL"
"$PY" -m bfcl_eval generate --model "$MODEL" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 128 --include-input-log
rc=$?
echo "[$(date '+%F %T')] generation done (rc=$rc)"
flock runlogs/evaluate.lock "$PY" -m bfcl_eval evaluate --model "$MODEL" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$rc)"
