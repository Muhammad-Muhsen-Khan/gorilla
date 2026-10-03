#!/usr/bin/env bash
# Serve one checkpoint on <gpus>, run the full suite, then stop that server so
# its GPUs are free for the next round. Usage: run_one.sh <gpus> <step> <port> <dp>
set -u
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/rl_olive_grpo
GPUS=$1; STEP=$2; PORT=$3; DP=$4
CKPT=/mnt/data01/muhsen/tooling/models-sft/grpo-olive-rl300k-olivesft1074/global_step_$STEP
MODEL=grpo-olive-rl300k-sft1074-step${STEP}-out16k-FC-keepreason

bash "$D/serve.sh" "$GPUS" "$CKPT" "$PORT" "$DP" > "$D/server_$STEP.log" 2>&1 &
SRV=$!
echo "[$(date '+%F %T')] step $STEP: server pid $SRV, gpus $GPUS, dp $DP, port $PORT"
until curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  kill -0 $SRV 2>/dev/null || { echo "[$(date '+%F %T')] step $STEP: SERVER DIED before ready"; exit 1; }
  sleep 10
done
echo "[$(date '+%F %T')] step $STEP: server ready, starting eval"
bash "$D/eval.sh" "$MODEL" "$PORT" 2>&1 | tee "$D/eval_$STEP.log"
# free the GPUs: the checkpoint path is unique to this server, so this matches nothing else
kill $SRV 2>/dev/null
sleep 5
pkill -u "$(id -un)" -f "global_step_$STEP" 2>/dev/null
sleep 5
echo "[$(date '+%F %T')] step $STEP: done, server stopped"
