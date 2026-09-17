#!/usr/bin/env bash
# Wait for the server on <port>, generate, then evaluate. Usage: eval.sh <model-key> <port>
set -u
cd /local/muhsen/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate
MODEL="$1"; export LOCAL_SERVER_PORT="$2" HF_HOME=/local/muhsen/hf-cache TMPDIR=/local/muhsen/tmp
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum

echo "[$(date '+%F %T')] waiting for vLLM on :$2"
until curl -sf "http://localhost:$2/v1/models" >/dev/null; do sleep 10; done
echo "[$(date '+%F %T')] generating: $MODEL"
bfcl generate --model "$MODEL" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 64 --include-input-log
echo "[$(date '+%F %T')] generation done (rc=$?)"
# bfcl evaluate rewrites score/data_*.csv; serialise so parallel evals don't clobber each other.
flock runlogs/nemotron32k/evaluate.lock bfcl evaluate --model "$MODEL" --test-category "$CATS"
echo "[$(date '+%F %T')] EVALUATION COMPLETE (rc=$?)"
