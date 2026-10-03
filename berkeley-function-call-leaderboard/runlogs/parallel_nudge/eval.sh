#!/usr/bin/env bash
# Wait for the server, generate the two parallel categories with the nudge, score.
# Usage: eval.sh <model-key> <port>
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
PY="$B/.venv/bin/python"
MODEL="$1"
export LOCAL_SERVER_PORT="$2" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=40960
# The whole point of the probe. Read by _system_suffix() in custom_model_config.py.
export BFCL_SYSTEM_SUFFIX='If the request requires several independent function calls, return all of them in the same message, each in its own <tool_call> block.'
CATS=parallel,parallel_multiple

echo "[$(date '+%F %T')] waiting for vLLM on :$2"
until curl -sf "http://localhost:$2/v1/models" >/dev/null; do sleep 10; done
echo "[$(date '+%F %T')] generating: $MODEL"
"$PY" -m bfcl_eval generate --model "$MODEL" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 128 --include-input-log
rc=$?
echo "[$(date '+%F %T')] generation done (rc=$rc)"
flock runlogs/evaluate.lock "$PY" -m bfcl_eval evaluate --model "$MODEL" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$rc)"
