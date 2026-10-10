#!/usr/bin/env bash
# One line per job: probe verdicts (compacted) + where the run has got to.
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/probe_loo
printf "%-17s %-7s %-44s %s\n" LABEL NODE "PROBE 1/20/28/32/39/60" STATE
while IFS=$'\t' read -r node gpus port key fmt label ckpt flag; do
  p="$D/$label.probe.log"; r="$D/$label.run.log"
  if [ -s "$p" ]; then
    v=$(awk 'NR>2 && NF>=4{t=$4; for(i=5;i<=NF;i++)t=t" "$i; gsub(/ \(noise\)|\(prose\)/,"",t); printf "%s|", (t~/OK/?"OK":(t~/DEGEN/?"DEG":(t~/NO CALL/?"noc":"ERR")))}' "$p")
  else v="(no probe yet)"; fi
  if   grep -q "EVALUATION COMPLETE" "$r" 2>/dev/null; then s="DONE"
  elif grep -q "generation done" "$r" 2>/dev/null;      then s="scoring"
  elif grep -q "starting generation" "$r" 2>/dev/null;   then s="generating"
  elif grep -q "ABORT\|PROBE FAILED\|PROBE DID NOT\|SERVER DIED\|NEVER CAME UP" "$r" 2>/dev/null; then s="FAILED"
  elif grep -q "long-context probe" "$r" 2>/dev/null;    then s="probing"
  else s="loading"; fi
  printf "%-17s %-7s %-44s %s\n" "$label" "$node" "$v" "$s"
done < "$D/jobs.tsv"
