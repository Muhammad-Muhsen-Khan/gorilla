#!/usr/bin/env bash
# Live progress for the nemotron-32k evals. Usage: progress.sh [refresh_seconds]
B=/local/muhsen/gorilla/berkeley-function-call-leaderboard
R=${1:-30}
STEPS="500 625 737"
while true; do
  clear
  echo "BFCL v4 -- abdelrahman-qwen nemotron-32k -- FC keepreason -- $(date '+%F %T')"
  echo "5017 entries/model | 20 categories (no web_search) | 1 GPU each, ctx 40960, max_new_tokens 4096"
  echo "=================================================================================="
  i=0
  for s in $STEPS; do
    k=abdelrahman-qwen-sft-nemotron32k-$s-FC-keepreason
    n=$(cat "$B"/result/$k/*/*_result.json "$B"/result/$k/*_result.json 2>/dev/null | grep -ac '"id"'); n=${n:-0}
    st=RUNNING; grep -q "EVALUATION COMPLETE" "$B/runlogs/nemotron32k/eval_$s.log" 2>/dev/null && st=EVALUATED
    printf "ckpt-%-4s gpu%d :%d  %5d/5017 (%3d%%)  %s\n" "$s" "$i" $((1053+i)) "$n" $((n*100/5017)) "$st"
    i=$((i+1))
  done
  echo "=================================================================================="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -3 \
    | awk -F', *' '{printf "gpu%s %3s%% %4dG   ", $1, $2, $3/1024}'; echo
  echo; echo "refresh ${R}s | switch windows: Ctrl-b <n> | detach: Ctrl-b d"
  sleep "$R"
done
