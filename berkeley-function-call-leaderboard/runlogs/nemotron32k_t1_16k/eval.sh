#!/usr/bin/env bash
# Wait for the server on <port>, generate, then evaluate. Usage: eval.sh <model-key> <port>
# Temperature 1.0; BFCL_MAX_OUTPUT_TOKENS / BFCL_MAX_CONTEXT_LENGTH lift the handler's
# hardcoded 4096 output cap and its config.json-derived 40960 context (base_oss_handler.py).
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
PY="$B/.venv/bin/python"
MODEL="$1"
export LOCAL_SERVER_PORT="$2" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=65536
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum

echo "[$(date '+%F %T')] waiting for vLLM on :$2"
until curl -sf "http://localhost:$2/v1/models" >/dev/null; do sleep 10; done
echo "[$(date '+%F %T')] generating: $MODEL"
"$PY" -m bfcl_eval generate --model "$MODEL" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 128 --include-input-log --temperature 1.0
rc=$?
echo "[$(date '+%F %T')] generation done (rc=$rc)"
# bfcl evaluate rewrites score/data_*.csv; serialise so parallel evals don't clobber each other.
flock runlogs/nemotron32k_t1_16k/evaluate.lock "$PY" -m bfcl_eval evaluate --model "$MODEL" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$rc)"
