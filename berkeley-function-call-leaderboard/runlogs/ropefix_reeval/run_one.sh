#!/usr/bin/env bash
# serve -> long-context probe -> gate -> generate -> evaluate -> tear the server down.
# Usage: run_one.sh <gpu-list> <ckpt-path> <port> <model-key> <json|xml> <label>
# The probe gate aborts only on a SHORT-context failure (1 or 20 tools), which means
# the server is wrong. Degeneration that appears only at long context is a real
# property of the model, so the eval still runs and the probe log records it.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
D=$B/runlogs/ropefix_reeval
cd "$B"
GPUS=$1; CKPT=$2; PORT=$3; KEY=$4; FMT=$5; LABEL=$6
PY="$B/.venv/bin/python"
export HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp

say(){ echo "[$(date '+%F %T')] [$LABEL] $*"; }
bash "$D/serve.sh" "$GPUS" "$CKPT" "$PORT" "$(awk -F, '{print NF}' <<<"$GPUS")" > "$D/$LABEL.server.log" 2>&1 &
SRV=$!
cleanup(){ say "tearing down server"; kill $SRV 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$GPUS" && kill -9 $p 2>/dev/null
  done; }
trap cleanup EXIT

say "waiting for vLLM on :$PORT (gpus $GPUS)"
for i in $(seq 1 180); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null && break
  kill -0 $SRV 2>/dev/null || { say "SERVER DIED -- see $LABEL.server.log"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { say "SERVER NEVER CAME UP"; exit 1; }

say "long-context probe"
"$PY" "$B/runlogs/sweep_longctx.py" "$PORT" "$CKPT" "$FMT" "$LABEL" 2>&1 | tee "$D/$LABEL.probe.log"
grep -qE '^ *60 ' "$D/$LABEL.probe.log" || { say "PROBE DID NOT RUN (no 60-tool row) -- aborting"; exit 2; }
if grep -qE '^ *(1|20) +[0-9]+ +[0-9]+ +(DEGENERATE|ERROR)' "$D/$LABEL.probe.log"; then
  say "PROBE FAILED AT SHORT CONTEXT -- aborting eval"; exit 2
fi
say "probe ok at short context; starting generation"

CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
export LOCAL_SERVER_PORT="$PORT" BFCL_MAX_OUTPUT_TOKENS=16384 BFCL_MAX_CONTEXT_LENGTH=40960
"$PY" -m bfcl_eval generate --model "$KEY" --test-category "$CATS" \
  --backend vllm --skip-server-setup --num-threads 128 --include-input-log
say "generation done (rc=$?)"
flock "$B/runlogs/evaluate.lock" "$PY" -m bfcl_eval evaluate --model "$KEY" --test-category "$CATS"
say "EVALUATION COMPLETE (rc=$?)"
