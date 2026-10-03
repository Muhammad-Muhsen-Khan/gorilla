#!/usr/bin/env bash
# Run ON avey-bm-03 (hgx19). GPU 6 is excluded: it idles at ~85 W against ~77 W
# for every other card on this host. Seven usable cards, two checkpoints, so
# three GPUs each (DP-3) and GPU 7 left spare for the next RL step.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/rl_olive_grpo
M=/mnt/data01/muhsen/tooling/models-sft/grpo-olive-rl300k-olivesft1074/global_step_
S=bfcl-rl-olive
tmux new-session -d -s "$S" -n idle "sleep infinity"
i=0
for step in 100 150; do
  case $i in 0) gpus=0,1,2 ;; 1) gpus=3,4,5 ;; esac
  port=$((1091+i))
  tmux new-window -t "$S" -n "srv$step"  "bash $D/serve.sh $gpus ${M}${step} $port 3 2>&1 | tee $D/server_$step.log"
  tmux new-window -t "$S" -n "eval$step" "bash $D/eval.sh grpo-olive-rl300k-sft1074-step${step}-out16k-FC-keepreason $port 2>&1 | tee $D/eval_$step.log"
  i=$((i+1))
done
echo "started tmux session $S on $(hostname)"
