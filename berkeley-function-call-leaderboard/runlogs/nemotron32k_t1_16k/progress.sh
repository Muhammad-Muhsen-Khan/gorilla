#!/usr/bin/env bash
# Live progress for the T=1.0 / out 16k / ctx 64k re-evals. Usage: progress.sh [refresh_seconds]
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
R=${1:-30}
STEPS="500 625 737"
while true; do
  clear
  echo "BFCL v4 -- abdelrahman-qwen nemotron-32k -- FC keepreason -- T=1.0 out16k ctx64k DP2 -- hgx11 -- $(date '+%F %T')"
  echo "=================================================================================="
  i=0
  for s in $STEPS; do
    k=abdelrahman-qwen-sft-nemotron32k-$s-t1-out16k-ctx64k-FC-keepreason
    n=$(cat "$B"/result/$k/*/*_result.json "$B"/result/$k/*_result.json 2>/dev/null | grep -ac '"id"'); n=${n:-0}
    st=RUNNING; grep -q "EVALUATION COMPLETE" "$B/runlogs/nemotron32k_t1_16k/eval_$s.log" 2>/dev/null && st=EVALUATED
    printf "ckpt-%-4s gpu%d,%d :%d  %5d/5017 (%3d%%)  %s\n" "$s" $((2*i)) $((2*i+1)) $((1053+i)) "$n" $((n*100/5017)) "$st"
    i=$((i+1))
  done
  echo "=================================================================================="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -6 \
    | awk -F', *' '{printf "gpu%s %3s%% %4dG   ", $1, $2, $3/1024}'; echo
  echo; echo "refresh ${R}s | switch windows: Ctrl-b <n> | detach: Ctrl-b d"
  sleep "$R"
done
