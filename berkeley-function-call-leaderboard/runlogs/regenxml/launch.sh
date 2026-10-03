#!/usr/bin/env bash
# regen-xml checkpoints (first + last) on avey-bm-04, 4 GPUs each.
# Handler: QwenXMLHandler -- Qwen3.5 XML tool calls, reasoning preserved.
set -u
R=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/regenxml
M=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-regen-xml-8src-32k-bs8-1ep-24gpu
RUNS=("0,1,2,3 8401 4 125" "4,5,6,7 8402 4 261")
for r in "${RUNS[@]}"; do
  set -- $r; GPUS=$1; PORT=$2; DP=$3; STEP=$4
  KEY="abdelrahman-qwen-sft-regenxml-${STEP}-out16k-XML"
  nohup "$R/serve.sh" "$GPUS" "$M/Qwen3-4B-Base-sft-${STEP}" "$PORT" "$DP" > "$R/server_${STEP}.log" 2>&1 &
  echo "server step=$STEP gpus=$GPUS port=$PORT dp=$DP pid=$!"
  if [ "${SERVE_ONLY:-0}" != "1" ]; then
    nohup "$R/eval.sh" "$KEY" "$PORT" > "$R/eval_${STEP}.log" 2>&1 &
    echo "eval   step=$STEP pid=$!"
  fi
done
