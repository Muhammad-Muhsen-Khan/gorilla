#!/usr/bin/env bash
# Phase 2: the two slow models, then the evaluation of everything.
#
# qwen3-4b-base and qwen3-4b-instruct never emit a stop token in this format
# (median output = 2048 tokens, the cap), so at one-GPU-per-model they projected
# to 20 h and 5 h. Here each model is sharded BY CATEGORY across all 8 GPUs --
# distinct categories write distinct result files, so the shards never collide --
# turning those into roughly 2.5 h and 40 min.
set -uo pipefail
cd /home/ubuntu/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate
SCHED_PID="$1"

GROUPS=(
  "live_multiple"
  "live_irrelevance"
  "simple_python,simple_java,simple_javascript"
  "multiple,parallel,parallel_multiple"
  "irrelevance,live_simple,live_parallel,live_parallel_multiple,live_relevance"
  "multi_turn_base,multi_turn_miss_func"
  "multi_turn_miss_param,multi_turn_long_context"
  "memory_kv,memory_vector,memory_rec_sum"
)

say(){ echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for generation scheduler (pid $SCHED_PID)"
while kill -0 "$SCHED_PID" 2>/dev/null; do sleep 30; done
say "scheduler exited; starting sharded runs"

for MODEL in qwen3-4b-base-FC qwen3-4b-instruct-FC; do
  say "=== $MODEL : 8 shards ==="
  pids=()
  for i in "${!GROUPS[@]}"; do
    CUDA_VISIBLE_DEVICES=$i LOCAL_SERVER_PORT=$((1053+i)) \
      bfcl generate --model "$MODEL" --test-category "${GROUPS[$i]}" \
        --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 \
        --num-threads 32 > "runlogs/gen_${MODEL}_shard${i}.log" 2>&1 &
    pids+=($!)
    say "  GPU$i shard$i -> ${GROUPS[$i]}"
  done
  for p in "${pids[@]}"; do wait "$p"; done
  say "=== $MODEL shards complete ==="
done

say "all generation finished; evaluating"
MODELS=$(python - <<'PY'
import sys; sys.path.insert(0,'runlogs')
import schedule
print(",".join(m for m in schedule.queue() if schedule.rows(m) > 0))
PY
)
say "evaluating: $MODELS"
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
bfcl evaluate --model "$MODELS" --test-category "$CATS"
say "EVALUATION COMPLETE"
