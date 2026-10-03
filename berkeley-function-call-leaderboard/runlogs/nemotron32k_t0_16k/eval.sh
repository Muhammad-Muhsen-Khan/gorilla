#!/usr/bin/env bash
# Wait for the server on <port>, generate, then evaluate. Usage: eval.sh <model-key> <port>
# Temperature is left at BFCL's default 0.001, as in the original run. Only
# BFCL_MAX_OUTPUT_TOKENS changes; BFCL_MAX_CONTEXT_LENGTH restates the 40960 the
# handler would read from config.json anyway, so the two servers cannot drift.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
PY="$B/.venv/bin/python"
MODEL="$1"
export LOCAL_SERVER_PORT="$2" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=40960
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum

echo "[$(date '+%F %T')] waiting for vLLM on :$2"
until curl -sf "http://localhost:$2/v1/models" >/dev/null; do sleep 10; done
echo "[$(date '+%F %T')] generating: $MODEL"
"$PY" -m bfcl_eval generate --model "$MODEL" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 128 --include-input-log
rc=$?
echo "[$(date '+%F %T')] generation done (rc=$rc)"
# bfcl evaluate rewrites score/data_*.csv; serialise against any other eval.
flock runlogs/evaluate.lock "$PY" -m bfcl_eval evaluate --model "$MODEL" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$rc)"
