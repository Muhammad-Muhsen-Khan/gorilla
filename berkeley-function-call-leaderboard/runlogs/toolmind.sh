#!/usr/bin/env bash
# ToolMind graphsyn SFT suite: the three epoch checkpoints, -FC variant only.
#
# Same methodology as the first 22-checkpoint suite: upstream QwenFCHandler
# (so reasoning is kept only within the current turn's chain and stripped from
# prior turns), same 20 categories, web_search excluded, --backend vllm.
#
# Each model is sharded BY CATEGORY across all 8 GPUs, as in phase2.sh --
# distinct categories write distinct result files, so shards never collide.
set -uo pipefail
cd /root/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate

MODELS=(qwen3-4b-sft-toolmind-916-FC qwen3-4b-sft-toolmind-1832-FC qwen3-4b-sft-toolmind-2748-FC)

SHARDS=(
  "live_multiple"
  "live_irrelevance"
  "simple_python,simple_java,simple_javascript"
  "multiple,parallel,parallel_multiple"
  "irrelevance,live_simple,live_parallel,live_parallel_multiple,live_relevance"
  "multi_turn_base,multi_turn_miss_func"
  "multi_turn_miss_param,multi_turn_long_context"
  "memory_kv,memory_vector,memory_rec_sum"
)
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
EXPECTED=5017

say(){ echo "[$(date +%H:%M:%S)] $*"; }
rows(){ local d="result/$1"; [ -d "$d" ] && find "$d" -name '*_result.json' -exec cat {} + 2>/dev/null | wc -l || echo 0; }

wait_for_gpus(){                       # a killed vLLM server holds ~36GiB for ~1min
  for _ in $(seq 1 60); do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${u:-0}" -lt 2000 ] && return 0
    sleep 10
  done
  say "WARNING: GPUs still busy, proceeding anyway"
}

for MODEL in "${MODELS[@]}"; do
  say "=== $MODEL : 8 category shards ==="
  wait_for_gpus
  pids=()
  for i in "${!SHARDS[@]}"; do
    CUDA_VISIBLE_DEVICES=$i LOCAL_SERVER_PORT=$((1053+i)) \
      bfcl generate --model "$MODEL" --test-category "${SHARDS[$i]}" \
        --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 \
        --num-threads 32 > "runlogs/gen_${MODEL}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  say "=== $MODEL shards done: $(rows "$MODEL")/$EXPECTED rows ==="

  # Top-up pass: bfcl skips entries that already exist, so this is fast and
  # fills any shard that died. `bfcl evaluate` hard-fails on a short model.
  if [ "$(rows "$MODEL")" -lt "$EXPECTED" ]; then
    say "$MODEL short -> top-up pass on all categories"
    wait_for_gpus
    CUDA_VISIBLE_DEVICES=0 LOCAL_SERVER_PORT=1053 \
      bfcl generate --model "$MODEL" --test-category "$CATS" \
        --backend vllm --num-gpus 1 --gpu-memory-utilization 0.9 \
        --num-threads 32 > "runlogs/gen_${MODEL}_topup.log" 2>&1
    say "$MODEL after top-up: $(rows "$MODEL")/$EXPECTED rows"
  fi
done

say "generation complete; evaluating"
JOINED=$(IFS=,; echo "${MODELS[*]}")
wait_for_gpus
bfcl evaluate --model "$JOINED" --test-category "$CATS" 2>&1 | tee runlogs/evaluate_toolmind.log
say "EVALUATION COMPLETE"
