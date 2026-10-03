#!/usr/bin/env bash
# Run ON avey-bm-02 (hgx03): two single-GPU servers in tmux session `bfcl-nudge`.
# olive-1074 on GPU 0 / port 1071, nemotron-737 on GPU 1 / port 1072.
set -eu
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/parallel_nudge
S=bfcl-nudge
OLIVE=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-olive-32k-bs8-3ep-30gpu/Qwen3-4B-Base-sft-1074
NEMO=/mnt/data01/muhsen/tooling/models-sft/abdelrahman-qwen-nemotron-32k-bs8/Qwen3-4B-Base-sft-final
tmux new-session -d -s "$S" -n idle "sleep infinity"
tmux new-window -t "$S" -n srv-olive "bash $D/serve.sh 0 $OLIVE 1071 2>&1 | tee $D/server_olive.log"
tmux new-window -t "$S" -n ev-olive  "bash $D/eval.sh abdelrahman-qwen-sft-olive32k-1074-out16k-nudge-FC-keepreason 1071 2>&1 | tee $D/eval_olive.log"
tmux new-window -t "$S" -n srv-nemo  "bash $D/serve.sh 1 $NEMO 1072 2>&1 | tee $D/server_nemo.log"
tmux new-window -t "$S" -n ev-nemo   "bash $D/eval.sh abdelrahman-qwen-sft-nemotron32k-737-out16k-nudge-FC-keepreason 1072 2>&1 | tee $D/eval_nemo.log"
echo "started tmux session $S on $(hostname)"
