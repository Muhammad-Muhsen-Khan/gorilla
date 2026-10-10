#!/usr/bin/env bash
# Entries written / 4441 expected, per job.
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
D=$B/runlogs/ropefix_reeval
printf "%-17s %-7s %6s %5s  %s\n" LABEL NODE DONE PCT STATE
tdone=0
while IFS=$'\t' read -r node gpus port key fmt label ckpt flag; do
  n=$(find "$B/result/$key" -name "*.json" -exec cat {} + 2>/dev/null | wc -l)
  tdone=$((tdone+n))
  r="$D/$label.run.log"
  if   grep -q "EVALUATION COMPLETE" "$r" 2>/dev/null; then s="DONE"
  elif grep -q "generation done" "$r" 2>/dev/null;      then s="scoring"
  elif grep -q "starting generation" "$r" 2>/dev/null;   then s="generating"
  elif grep -q "PROBE FAILED\|PROBE DID NOT\|SERVER DIED\|NEVER CAME UP" "$r" 2>/dev/null; then s="FAILED"
  else s="probing/loading"; fi
  printf "%-17s %-7s %6d %4d%%  %s\n" "$label" "$node" "$n" "$((n*100/4441))" "$s"
done < "$D/jobs.tsv"
n=$(find "$B/result/abdelrahman-qwen-sft-jsonfull-448-out16k-FC-keepreason" -name "*.json" -exec cat {} + 2>/dev/null | wc -l)
printf "%-17s %-7s %6d %4d%%  %s\n" "jsonfull-448" "hgx11" "$n" "$((n*100/4441))" "generating"
echo "---- total $((tdone+n)) / $((18*4441)) entries"
