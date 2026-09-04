#!/usr/bin/env bash
# Recovery + completion for the toolmind suite.
#
# Phase A: 1832's 246 missing entries (3 GPUs) run CONCURRENTLY with 2748's four
# multi-turn categories (5 GPUs). Different models write different result trees,
# so they never touch the same file. This keeps all 8 GPUs busy, unlike the
# one-model-at-a-time split inherited from phase2.sh which left 5 idle.
# Phase B: 2748's fast single-turn bundle once GPUs free.
# Then dedupe (one row per id, last wins) and evaluate all three.
#
# 'GROUPS' is a bash read-only special array (user gids) -- never use that name.
set -uo pipefail
cd /root/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate

M1832=qwen3-4b-sft-toolmind-1832-FC
M2748=qwen3-4b-sft-toolmind-2748-FC
ALL=(qwen3-4b-sft-toolmind-916-FC "$M1832" "$M2748")
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum
SINGLE=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance
EXPECTED=5017

say(){ echo "[$(date +%H:%M:%S)] $*"; }
rows(){ local d="result/$1"; [ -d "$d" ] && find "$d" -name '*_result.json' -exec cat {} + 2>/dev/null | wc -l || echo 0; }
run(){ # gpu, model, categories, tag
  CUDA_VISIBLE_DEVICES=$1 LOCAL_SERVER_PORT=$((1053+$1)) \
    bfcl generate --model "$2" --test-category "$3" --backend vllm --num-gpus 1 \
      --gpu-memory-utilization 0.9 --num-threads 32 > "runlogs/fin_$4.log" 2>&1
}

say "=== phase A: 1832 remainder (gpu0-2) + 2748 multi-turn (gpu3-7) ==="
pids=()
run 0 "$M1832" "multi_turn_miss_func"                  "1832_missfunc"  & pids+=($!)
run 1 "$M1832" "multi_turn_miss_param"                 "1832_missparam" & pids+=($!)
run 2 "$M1832" "multi_turn_long_context,memory_vector" "1832_tail"      & pids+=($!)
run 3 "$M2748" "multi_turn_base"                       "2748_base"      & pids+=($!)
run 4 "$M2748" "multi_turn_miss_func"                  "2748_missfunc"  & pids+=($!)
run 5 "$M2748" "multi_turn_miss_param"                 "2748_missparam" & pids+=($!)
run 6 "$M2748" "multi_turn_long_context"               "2748_longctx"   & pids+=($!)
run 7 "$M2748" "memory_kv,memory_vector,memory_rec_sum" "2748_memory"   & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
say "phase A done: 1832=$(rows $M1832)  2748=$(rows $M2748)"

say "=== phase B: 2748 single-turn bundle ==="
run 0 "$M2748" "$SINGLE" "2748_single"
say "phase B done: 2748=$(rows $M2748)"

for m in "${ALL[@]}"; do
  if [ "$(rows "$m")" -lt "$EXPECTED" ]; then
    say "$m short ($(rows $m)/$EXPECTED) -> top-up"
    run 0 "$m" "$CATS" "topup_$m"
    say "$m after top-up: $(rows $m)/$EXPECTED"
  fi
done

say "dedupe pass"
python - <<'PY'
import json, pathlib
total = 0
for f in pathlib.Path("result").rglob("*_result.json"):
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
say "=== evaluating all three ==="
JOINED=$(IFS=,; echo "${ALL[*]}")
bfcl evaluate --model "$JOINED" --test-category "$CATS" 2>&1 | tee runlogs/evaluate_toolmind.log
say "EVALUATION COMPLETE"
