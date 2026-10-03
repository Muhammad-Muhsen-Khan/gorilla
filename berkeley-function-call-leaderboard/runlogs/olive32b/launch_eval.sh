#!/usr/bin/env bash
# Run ON avey-bm-06. Servers are already up (serve_only.sh): ckpt-954 on GPUs
# 0,1,2 port 1101, ckpt-477 on GPUs 4,5,7 port 1102. GPUs 3 and 6 are throttled
# and stay unused. Scored with the lenient-json handler.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/olive32b
S=bfcl-32b
tmux new-window -t "$S" -n ev954 "bash $D/eval.sh olive32b-olive-tooling-954-out16k-FC-keepreason-lenient 1101 2>&1 | tee $D/eval_954.log"
tmux new-window -t "$S" -n ev477 "bash $D/eval.sh olive32b-olive-tooling-477-out16k-FC-keepreason-lenient 1102 2>&1 | tee $D/eval_477.log"
echo "evals launched on $(hostname)"
