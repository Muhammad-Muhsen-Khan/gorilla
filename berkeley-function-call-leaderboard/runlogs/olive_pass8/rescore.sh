#!/usr/bin/env bash
# Re-score any draw whose score files are truncated. Only needed for runs that
# were scored before the errors="backslashreplace" fix in bfcl_eval/utils.py.
# Usage: rescore.sh <model-key>
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
CATS=simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance
flock runlogs/evaluate.lock "$B/.venv/bin/python" -m bfcl_eval evaluate --model "$1" --test-category "$CATS"
rc=$?
echo "[$(date '+%F %T')] RESCORE COMPLETE $1 (rc=$rc)"
