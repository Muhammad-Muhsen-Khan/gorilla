#!/usr/bin/env bash
# Run ON avey-bm-02: one DP-8 server on all GPUs + the eval, in tmux session `bfcl-rl450`.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/rl_v4_step450_out16k
M=/mnt/data01/muhsen/tooling/llm-pretrainer/models/qwen3-4b-rl-v4-sft737-step450
K=qwen3-4b-rl-v4-sft737-step450-out16k-FC-keepreason
S=bfcl-rl450
PORT=1060
tmux new-session -d -s "$S" -n progress "bash $D/progress.sh"
tmux new-window -t "$S" -n srv "bash $D/serve.sh $M $PORT 2>&1 | tee $D/server.log"
tmux new-window -t "$S" -n eval "bash $D/eval.sh $K $PORT 2>&1 | tee $D/eval.log"
echo "started tmux session $S on $(hostname) (tmux attach -t $S)"
