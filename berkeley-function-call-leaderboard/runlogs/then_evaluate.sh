#!/usr/bin/env bash
# Waits for the generation scheduler to exit, then evaluates every model in one
# pass so all rows land in a single score/data_overall.csv.
set -uo pipefail
cd /home/ubuntu/gorilla/berkeley-function-call-leaderboard
source .venv/bin/activate

SCHED_PID="$1"
while kill -0 "$SCHED_PID" 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] generation scheduler exited; starting evaluation"

MODELS=$(python - <<'PY'
import sys; sys.path.insert(0,'runlogs')
import schedule
print(",".join(m for m in schedule.queue() if schedule.rows(m) > 0))
PY
)
echo "[$(date +%H:%M:%S)] evaluating: $MODELS"

CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum

bfcl evaluate --model "$MODELS" --test-category "$CATS"
echo "[$(date +%H:%M:%S)] EVALUATION COMPLETE"
