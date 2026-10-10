#!/usr/bin/env bash
# Scrape BFCL's own tqdm progress bar (authoritative: it knows the real denominator).
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/probe_loo
printf "%-18s %-7s %-22s %s\n" LABEL NODE "PROGRESS" STATE
while IFS=$'\t' read -r node gpus port key fmt label ckpt flag; do
  r="$D/$label.run.log"
  bar=$(tr '\r' '\n' < "$r" 2>/dev/null | grep -ao '[0-9]\+%|*[^|]*| *[0-9]\+/[0-9]\+ \[[^]]*\]' | tail -1 \
        | sed 's/|[^|]*| */ /' | awk '{print $1, $2, $3}')
  if   grep -q "EVALUATION COMPLETE" "$r" 2>/dev/null; then s="DONE"
  elif grep -q "generation done" "$r" 2>/dev/null;      then s="scoring"
  elif grep -q "starting generation" "$r" 2>/dev/null;   then s="generating"
  elif grep -q "PROBE FAILED\|PROBE DID NOT\|SERVER DIED\|NEVER CAME UP" "$r" 2>/dev/null; then s="FAILED"
  else s="probe/load"; fi
  printf "%-18s %-7s %-22s %s\n" "$label" "$node" "${bar:-—}" "$s"
done < "$D/jobs.tsv"
