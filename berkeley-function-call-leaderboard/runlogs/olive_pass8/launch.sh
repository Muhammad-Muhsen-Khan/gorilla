#!/usr/bin/env bash
# Run ON avey-bm-02: seven independent T=1.0 draws, one per GPU, in tmux `bfcl-p8`.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/olive_pass8
M=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-32k-bs8-3ep-30gpu/Qwen3-4B-Base-sft-1074
S=bfcl-p8
tmux new-session -d -s "$S" -n idle "sleep infinity"
for i in 1 2 3 4 5 6 7; do
  gpu=$((i-1)); port=$((1080+i))
  tmux new-window -t "$S" -n "srv$i"  "bash $D/serve.sh $gpu $M $port 2>&1 | tee $D/server_s$i.log"
  tmux new-window -t "$S" -n "eval$i" "bash $D/eval.sh abdelrahman-qwen-sft-olive32k-1074-t1-s$i-FC-keepreason $port 2>&1 | tee $D/eval_s$i.log"
done
echo "started tmux session $S on $(hostname)"
