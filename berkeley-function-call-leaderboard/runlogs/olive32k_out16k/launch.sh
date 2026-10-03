#!/usr/bin/env bash
# Run ON avey-bm-04: three single-GPU servers + evals in tmux session `bfcl-olive`.
# ckpt i on GPU i, port 1061+i.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/olive32k_out16k
M=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-32k-bs8-3ep-30gpu/Qwen3-4B-Base-sft-
S=bfcl-olive
tmux new-session -d -s "$S" -n progress "bash $D/progress.sh"
i=0
for step in 875 1000 1074; do
  port=$((1061+i))
  tmux new-window -t "$S" -n "srv$step" "bash $D/serve.sh $i ${M}${step} $port 2>&1 | tee $D/server_$step.log"
  tmux new-window -t "$S" -n "eval$step" "bash $D/eval.sh abdelrahman-qwen-sft-olive32k-$step-out16k-FC-keepreason $port 2>&1 | tee $D/eval_$step.log"
  i=$((i+1))
done
echo "started tmux session $S on $(hostname) (tmux attach -t $S)"
