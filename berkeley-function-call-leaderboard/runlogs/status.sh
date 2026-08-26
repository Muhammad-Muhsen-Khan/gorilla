#!/usr/bin/env bash
cd /home/ubuntu/gorilla/berkeley-function-call-leaderboard
printf "%-34s %8s %10s %12s\n" MODEL DONE RATE ETA
for f in runlogs/gen_*.log; do
  m=$(basename "$f" .log | sed 's/gen_//')
  line=$(tr '\r' '\n' < "$f" | grep -a "Generating results for $m:" | tail -1)
  n=$(echo "$line" | grep -oE '[0-9]+/5017' | cut -d/ -f1)
  rate=$(echo "$line" | grep -oE '[0-9.]+(it/s|s/it)' | tail -1)
  eta=$(echo "$line" | grep -oE '<[0-9:]+' | tr -d '<')
  printf "%-34s %8s %10s %12s\n" "$m" "${n:-0}" "${rate:-—}" "${eta:-—}"
done
