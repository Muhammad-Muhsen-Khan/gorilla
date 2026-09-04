#!/usr/bin/env bash
# Finishes the toolmind suite: runs ckpt-2748 with a REBALANCED 8-way split,
# then dedupes and evaluates all three.
#
# The original split (from phase2.sh) pairs two multi-turn categories per shard,
# so the critical path is ~58 min while 5 GPUs idle after ~7 min. Multi-turn
# entries cost 10-20 sequential generations each; single-turn cost one. Here each
# multi-turn category gets its own GPU and every single-turn category is packed
# into one shard, halving the critical path.
#
# NOTE: 'GROUPS' is a bash special read-only array (the user's gids) -- naming a
# shard array GROUPS silently yields "0". phase2.sh has that bug. Hence SHARDS.
set -uo pipefail
cd /root/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate

MODEL=qwen3-4b-sft-toolmind-2748-FC
ALL=(qwen3-4b-sft-toolmind-916-FC qwen3-4b-sft-toolmind-1832-FC qwen3-4b-sft-toolmind-2748-FC)

SHARDS=(
  "multi_turn_base"                  # 200 entries, slowest class
  "multi_turn_miss_func"             # 200
  "multi_turn_miss_param"            # 200
  "multi_turn_long_context"          # 200
  "memory_kv"
  "memory_vector"
  "memory_rec_sum"
  "simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance"
)
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
EXPECTED=5017

say(){ echo "[$(date +%H:%M:%S)] $*"; }
rows(){ local d="result/$1"; [ -d "$d" ] && find "$d" -name '*_result.json' -exec cat {} + 2>/dev/null | wc -l || echo 0; }

say "waiting for the 1832 shards still in flight"
while pgrep -f "bfcl generate --model qwen3-4b-sft-toolmind-1832-FC" >/dev/null; do sleep 30; done
say "1832 done: $(rows qwen3-4b-sft-toolmind-1832-FC)/$EXPECTED rows"

say "waiting for GPU memory to be released"
for _ in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
  [ "${u:-0}" -lt 2000 ] && break
  sleep 10
done

say "=== $MODEL : 8 rebalanced shards ==="
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

if [ "$(rows "$MODEL")" -lt "$EXPECTED" ]; then
  say "short -> top-up pass"
  CUDA_VISIBLE_DEVICES=0 LOCAL_SERVER_PORT=1053 \
    bfcl generate --model "$MODEL" --test-category "$CATS" --backend vllm \
      --num-gpus 1 --gpu-memory-utilization 0.9 --num-threads 32 \
      > "runlogs/gen_${MODEL}_topup.log" 2>&1
  say "after top-up: $(rows "$MODEL")/$EXPECTED"
fi

# Same guarantee driver.dedupe_results() gives: one row per id, last wins.
say "dedupe pass"
python - <<'PY'
import json, pathlib
root = pathlib.Path("result"); total = 0
for f in root.rglob("*_result.json"):
    seen = {}
    for line in open(f):
        line = line.strip()
        if not line: continue
        try: seen[json.loads(line)["id"]] = line
        except Exception: continue
    n_in = sum(1 for l in open(f) if l.strip())
    if len(seen) != n_in:
        f.write_text("\n".join(seen.values()) + "\n"); total += n_in - len(seen)
        print(f"  deduped {f.parent.name}/{f.name}: -{n_in-len(seen)}")
print(f"dedupe removed {total} duplicate rows")
PY

for m in "${ALL[@]}"; do say "$m: $(rows "$m")/$EXPECTED"; done
say "evaluating all three"
JOINED=$(IFS=,; echo "${ALL[*]}")
bfcl evaluate --model "$JOINED" --test-category "$CATS" 2>&1 | tee runlogs/evaluate_toolmind.log
say "EVALUATION COMPLETE"
