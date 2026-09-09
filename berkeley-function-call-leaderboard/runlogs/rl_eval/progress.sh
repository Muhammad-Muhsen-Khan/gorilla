#!/usr/bin/env bash
# Live BFCL eval progress. Usage: progress.sh [refresh_seconds]
B=/home/ubuntu/gorilla/berkeley-function-call-leaderboard
R=${1:-20}
declare -A PORT=( [v2step50]=8160 [v2step100]=8161 )
declare -A KEY=( [v2step50]=qwen3-4b-rl-v2-50-FC-keepreason \
                 [v2step100]=qwen3-4b-rl-v2-100-FC-keepreason )
bar() { local p=$1 w=42; local f=$((p*w/100)); printf '['; printf '#%.0s' $(seq 1 $f 2>/dev/null); printf ' %.0s' $(seq 1 $((w-f)) 2>/dev/null); printf ']'; }

while true; do
  clear
  echo "BFCL v4 -- FC keepreason -- $(date '+%Y-%m-%d %H:%M:%S')"
  echo "5017 entries/model | 20 categories (no web_search) | vLLM DP4, max_new_tokens=4096"
  echo "=================================================================================="
  for m in v2step50 v2step100; do
    done_n=$(find "$B/result/${KEY[$m]}" -name '*_result.json' -exec cat {} + 2>/dev/null \
             | grep -ac '"id"' 2>/dev/null); done_n=${done_n:-0}
    pct=$((done_n*100/5017))
    eta=$(tr '\r' '\n' < "$B/runlogs/rl_eval/$m.log" 2>/dev/null | grep -aoE '<[0-9]+:[0-9]+:?[0-9]*, *[0-9.]+s?/it' | tail -1)
    alive=$(pgrep -fc "bfcl generate --model ${KEY[$m]}" 2>/dev/null || echo 0)
    [ "$alive" -gt 0 ] 2>/dev/null && st="RUNNING" || st="DONE/STOPPED"
    grep -aq "GENERATION DONE" "$B/runlogs/rl_eval/$m.log" 2>/dev/null && st="GENERATION DONE"
    printf "%-10s %s %5s%%  %5d/5017  %-16s %s\n" "$m" "$(bar $pct)" "$pct" "$done_n" "$st" "$eta"
  done
  echo "=================================================================================="
  printf "GPU  "; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | awk -F', *' '{printf "g%s:%s%% %dG  ", $1, $2, $3/1024}'; echo
  printf "srv  "; for p in 8160 8161; do
    curl -s --max-time 2 "http://localhost:$p/v1/models" >/dev/null 2>&1 && printf "%s:UP  " "$p" || printf "%s:DOWN  " "$p"; done; echo
  echo
  echo "refresh ${R}s | Ctrl-b d to detach | Ctrl-c to stop watching (evals keep running)"
  sleep "$R"
done
