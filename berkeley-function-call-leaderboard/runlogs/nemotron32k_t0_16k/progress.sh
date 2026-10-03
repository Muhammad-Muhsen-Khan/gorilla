#!/usr/bin/env bash
# Live progress for the T=0.001 / out 16k / ctx 41k re-eval. Usage: progress.sh [refresh_seconds]
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
R=${1:-30}
K=abdelrahman-qwen-sft-nemotron32k-737-t0-out16k-ctx41k-FC-keepreason
while true; do
  clear
  echo "BFCL v4 -- nemotron-32k ckpt-737 -- FC keepreason -- T=0.001 out16k ctx41k DP8 -- $(date '+%F %T')"
  echo "=================================================================================="
  n=$(cat "$B"/result/$K/*/*_result.json "$B"/result/$K/*/*/*/*_result.json 2>/dev/null | grep -ac '"id"'); n=${n:-0}
  st=RUNNING; grep -q "EVALUATION COMPLETE" "$B/runlogs/nemotron32k_t0_16k/eval.log" 2>/dev/null && st=EVALUATED
  printf "ckpt-737  gpus 0-7  :1060   %5d/5017 (%3d%%)  %s\n" "$n" $((n*100/5017)) "$st"
  echo "=================================================================================="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '{printf "gpu%s %3s%% %4dG  ", $1, $2, $3/1024}'; echo
  echo; echo "refresh ${R}s | switch windows: Ctrl-b <n> | detach: Ctrl-b d"
  sleep "$R"
done
