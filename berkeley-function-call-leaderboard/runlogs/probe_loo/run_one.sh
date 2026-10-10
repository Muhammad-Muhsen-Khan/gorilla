#!/usr/bin/env bash
# serve -> long-context probe -> gate -> generate -> evaluate -> tear the server down.
# Usage: run_one.sh <gpu-list> <ckpt-path> <port> <model-key> <json|xml> <label>
# The probe gate aborts only on a SHORT-context failure (1 or 20 tools), which means
# the server is wrong. Degeneration that appears only at long context is a real
# property of the model, so the eval still runs and the probe log records it.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
D=$B/runlogs/probe_loo
cd "$B"
GPUS=$1; CKPT=$2; PORT=$3; KEY=$4; FMT=$5; LABEL=$6
PY="$B/.venv/bin/python"
export HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp

say(){ echo "[$(date '+%F %T')] [$LABEL] $*"; }
bash "$B/runlogs/ropefix_reeval/serve.sh" "$GPUS" "$CKPT" "$PORT" "$(awk -F, '{print NF}' <<<"$GPUS")" > "$D/$LABEL.server.log" 2>&1 &
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
rc=$?
say "generation done (rc=$rc)"
# Self-document the Memory column: the memory_* PREREQ conversations run ~41,120-41,145
# prompt tokens against the 40,960 ceiling, so vLLM 400s them. Those are the turns that
# populate the memory store, so the scored cases read an incomplete store and Memory comes
# out understated. Count them per arm so a later reader cannot mistake a low Memory number
# for a property of the held-out corpus.
CTXREJ=$(tr "\r" "\n" < "$D/$LABEL.run.log" 2>/dev/null | grep -ac "maximum context length is 40960")
# "requested N" includes the completion allowance; "N in the messages" is the prompt itself.
RANGE=$(tr "\r" "\n" < "$D/$LABEL.run.log" 2>/dev/null | grep -ao "([0-9]* in the messages" | grep -ao "[0-9]*" | sort -n | sed -n "1p;\$p" | paste -sd-)
{ echo "arm=$LABEL"
  echo "ctx_ceiling=40960"
  echo "memory_prereq_requests_rejected_http400=$CTXREJ"
  echo "rejected_prompt_token_range=${RANGE:-none}  (prompt tokens in messages, ceiling 40960)"
  echo "effect=memory_* prereq turns dropped => memory store incomplete => Memory understated (floor, not measurement)"
  echo "applies_equally_to_all_arms=yes (same ceiling, same data) => LOO deltas and arm-vs-base unaffected"
  echo "also_affects=multi_turn_* (multi_turn_long_context is the LARGEST rejection bucket), not memory alone"
  echo "no_ceiling_fixes_this=prompts reach ~86650 tok = 2.6x the 32k training length; 64k would still reject the longest"
  echo "COMPOUNDING_WARNING=~4% of TRAINING tokens were truncated at the 32k max-length, concentrated in"
  echo "  swe-zero-openhands and openresearcher long trajectories. For the four long-horizon sources"
  echo "  (swe-zero-openhands, openresearcher, terminal-corpus, openseeker) the data is clipped at TRAIN time"
  echo "  and then measured in the column most blunted at EVAL time. A BFCL null for any of those four means"
  echo "  'neither the training nor the measurement could see this', NOT 'this source does not help'."
  echo "  Do not drop a source on a BFCL null alone."
  echo "base_reference=ckpt-737 was scored under the SAME ceiling, so arm-vs-base is common-mode too"
} > "$D/$LABEL.ctxnote.txt"
say "ctx-rejected memory prereq requests: $CTXREJ (prompt tokens ${RANGE:-n/a} vs ceiling 40960)"
# A category can land 1-2 entries short when a case errors out; a strict evaluate
# then aborts with ValueError and writes no scores at all. Retry with --partial-eval.
flock "$B/runlogs/evaluate.lock" "$PY" -m bfcl_eval evaluate --model "$KEY" --test-category "$CATS"
rc=$?
if [ $rc -ne 0 ]; then
  say "strict evaluate failed (rc=$rc); retrying with --partial-eval"
  flock "$B/runlogs/evaluate.lock" "$PY" -m bfcl_eval evaluate --model "$KEY" --test-category "$CATS" --partial-eval
  rc=$?
  say "EVALUATION COMPLETE partial (rc=$rc)"
else
  say "EVALUATION COMPLETE (rc=$rc)"
fi
