#!/usr/bin/env bash
# Start both 32B servers on avey-bm-06 and stop. GPUs 3 and 6 are excluded
# (throttled), leaving 0,1,2 for ckpt-954 and 4,5,7 for ckpt-477.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/olive32b
M=/mnt/data01/muhsen/tooling/models-sft/olive-32b-olive-tooling-32k-bs3-3ep-30gpu/checkpoint-
S=bfcl-32b
tmux new-session -d -s "$S" -n idle "sleep infinity"
tmux new-window -t "$S" -n srv954 "bash $D/serve.sh 0,1,2 ${M}954 1101 3 2>&1 | tee $D/server_954.log"
tmux new-window -t "$S" -n srv477 "bash $D/serve.sh 4,5,7 ${M}477 1102 3 2>&1 | tee $D/server_477.log"
echo "servers starting on $(hostname)"
