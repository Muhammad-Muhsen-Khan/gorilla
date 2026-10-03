#!/usr/bin/env bash
# Launch all three length-bias epoch checkpoints concurrently on avey-bm-01.
# GPU split requested by the user: 3 / 3 / 2.
set -u
R=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/lengthbias
#          gpus     port  dp  step
RUNS=("0,1,2 8301 3 375" "3,4,5 8302 3 750" "6,7 8303 2 1020")
for r in "${RUNS[@]}"; do
  set -- $r; GPUS=$1; PORT=$2; DP=$3; STEP=$4
  KEY="abdelrahman-qwen-sft-lengthbias-${STEP}-out16k-FC-keepreason"
  P="/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-lengthbias-32k-bs8-3ep-32gpu/Qwen3-4B-Base-sft-${STEP}"
  nohup "$R/serve.sh" "$GPUS" "$P" "$PORT" "$DP" > "$R/server_${STEP}.log" 2>&1 &
  echo "server step=$STEP gpus=$GPUS port=$PORT dp=$DP pid=$!"
  nohup "$R/eval.sh" "$KEY" "$PORT" > "$R/eval_${STEP}.log" 2>&1 &
  echo "eval   step=$STEP pid=$!"
done
