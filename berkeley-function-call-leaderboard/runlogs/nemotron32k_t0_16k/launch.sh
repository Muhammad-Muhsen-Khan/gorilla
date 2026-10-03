#!/usr/bin/env bash
# One DP-8 server on all GPUs + the eval, in tmux session `bfcl-t0-16k`.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/nemotron32k_t0_16k
M=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-final
K=abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k-FC-keepreason
S=bfcl-t0-16k
PORT=1060
tmux new-session -d -s "$S" -n progress "bash $D/progress.sh"
tmux new-window -t "$S" -n srv737 "bash $D/serve.sh $M $PORT 2>&1 | tee $D/server.log"
tmux new-window -t "$S" -n eval737 "bash $D/eval.sh $K $PORT 2>&1 | tee $D/eval.log"
echo "started tmux session $S (tmux attach -t $S)"
