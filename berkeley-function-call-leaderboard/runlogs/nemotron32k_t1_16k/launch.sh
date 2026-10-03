#!/usr/bin/env bash
# Start all three servers + evals in tmux session `bfcl-t1-16k`: ckpt i on GPUs 2i,2i+1 (DP 2), port 1053+i.
# Servers start 20s apart: each picks its DP coordination port with get_open_port().
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/nemotron32k_t1_16k
M=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-
S=bfcl-t1-16k
tmux new-session -d -s "$S" -n progress "bash $D/progress.sh"
i=0
for pair in 500:500 625:625 737:final; do
  step=${pair%%:*}; dir=${pair##*:}; port=$((1053+i))
  gpus="$((2*i)),$((2*i+1))"
  tmux new-window -t "$S" -n "srv$step" "bash $D/serve.sh $gpus ${M}${dir} $port 2>&1 | tee $D/server_$step.log"
  tmux new-window -t "$S" -n "eval$step" "bash $D/eval.sh abdelrahman-qwen-sft-nemotron32k-$step-t1-out16k-ctx64k-FC-keepreason $port 2>&1 | tee $D/eval_$step.log"
  i=$((i+1))
  sleep 20
done
echo "started tmux session $S (tmux attach -t $S)"
