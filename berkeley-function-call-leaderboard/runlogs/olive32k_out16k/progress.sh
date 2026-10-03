#!/usr/bin/env bash
# Live progress for the olive-32k evals (T=0.001 / out 16k / ctx 41k). Usage: progress.sh [refresh_seconds]
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
R=${1:-30}
while true; do
  clear
  echo "BFCL v4 -- abdelrahman-qwen olive-32k -- FC keepreason -- T=0.001 out16k ctx41k -- $(hostname) -- $(date '+%F %T')"
  echo "=================================================================================="
  i=0
  for s in 875 1000 1074; do
    k=abdelrahman-qwen-sft-olive32k-$s-out16k-FC-keepreason
    n=$(cat "$B"/result/$k/*/*_result.json "$B"/result/$k/*/*/*/*_result.json 2>/dev/null | grep -ac '"id"'); n=${n:-0}
    st=RUNNING; grep -q "EVALUATION COMPLETE" "$B/runlogs/olive32k_out16k/eval_$s.log" 2>/dev/null && st=EVALUATED
    printf "ckpt-%-5s gpu%d :%d  %5d/5017 (%3d%%)  %s\n" "$s" $i $((1061+i)) "$n" $((n*100/5017)) "$st"
    i=$((i+1))
  done
  echo "=================================================================================="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -3 \
    | awk -F', *' '{printf "gpu%s %3s%% %4dG   ", $1, $2, $3/1024}'; echo
  echo; echo "refresh ${R}s | switch windows: Ctrl-b <n> | detach: Ctrl-b d"
  sleep "$R"
done
