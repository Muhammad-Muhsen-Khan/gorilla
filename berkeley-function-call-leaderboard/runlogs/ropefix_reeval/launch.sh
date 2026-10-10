#!/usr/bin/env bash
# Launch every job in jobs.tsv whose node matches $1, one tmux window each.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
D=$B/runlogs/ropefix_reeval
M=/mnt/data01/muhsen/tooling/models-sft
cd "$B"
NODE=$1
tmux has-session -t reeval 2>/dev/null || tmux new-session -d -s reeval -n idle "sleep infinity"
while IFS=$'\t' read -r node gpus port key fmt label ckpt flag; do
  [ "$node" = "$NODE" ] || continue
  if [ "$flag" = "32b" ]; then
    G='{"max_new_tokens": 16384, "temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0}'
  else
    G='{"max_new_tokens": 16384}'
  fi
  tmux new-window -t reeval -n "$label" \
    "GENCFG='$G' bash $D/run_one.sh $gpus $M/$ckpt $port $key $fmt $label 2>&1 | tee $D/$label.run.log"
  echo "launched $label  gpus=$gpus port=$port"
  sleep 2
done < "$D/jobs.tsv"
tmux list-windows -t reeval | sed 's/^/  /'
