#!/usr/bin/env bash
# Run ON avey-bm-03 (hgx19). Seven usable GPUs (6 is faulty: ~85 W idle vs ~77 W).
# Four checkpoints in two rounds, each round using all seven cards:
#   round 1: step 200 on 0,1,2 (DP-3)   step 250 on 3,4,5,7 (DP-4)
#   round 2: step 300 on 0,1,2 (DP-3)   step 350 on 3,4,5,7 (DP-4)
# A 2/2/2/1 split instead would leave six cards idle while the DP-1 straggler
# ran ~3x longer than the others.
set -u
D=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard/runlogs/rl_olive_grpo
{
  echo "[$(date '+%F %T')] ROUND 1 starting"
  bash "$D/run_one.sh" 0,1,2   200 1091 3 > "$D/round_200.log" 2>&1 &
  A=$!
  bash "$D/run_one.sh" 3,4,5,7 250 1092 4 > "$D/round_250.log" 2>&1 &
  B=$!
  wait $A; wait $B
  echo "[$(date '+%F %T')] ROUND 1 complete"
  sleep 20
  echo "[$(date '+%F %T')] ROUND 2 starting"
  bash "$D/run_one.sh" 0,1,2   300 1091 3 > "$D/round_300.log" 2>&1 &
  A=$!
  bash "$D/run_one.sh" 3,4,5,7 350 1092 4 > "$D/round_350.log" 2>&1 &
  B=$!
  wait $A; wait $B
  echo "[$(date '+%F %T')] ALL ROUNDS COMPLETE"
} > "$D/driver.log" 2>&1
