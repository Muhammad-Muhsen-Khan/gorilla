#!/usr/bin/env bash
# One T=1.0 draw over live + non-live. Usage: eval.sh <model-key> <port>
# No BFCL_SYSTEM_* set, so the prompt is byte-identical to the scored
# ckpt-1074 row; only --temperature differs.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
PY="$B/.venv/bin/python"
MODEL="$1"
export LOCAL_SERVER_PORT="$2" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=40960
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance

echo "[$(date '+%F %T')] waiting for vLLM on :$2"
until curl -sf "http://localhost:$2/v1/models" >/dev/null; do sleep 10; done
echo "[$(date '+%F %T')] generating: $MODEL (T=1.0)"
"$PY" -m bfcl_eval generate --model "$MODEL" --test-category "$CATS" --temperature 1.0 \
  --backend vllm --skip-server-setup --num-threads 128
rc=$?
echo "[$(date '+%F %T')] generation done (rc=$rc)"
flock runlogs/evaluate.lock "$PY" -m bfcl_eval evaluate --model "$MODEL" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$rc)"
