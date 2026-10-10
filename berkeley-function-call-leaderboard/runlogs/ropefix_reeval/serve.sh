#!/usr/bin/env bash
# Serve a checkpoint whose config.json was patched in place with a flat rope_theta
# (see the .pre-ropefix backups). No --hf-overrides for rope_theta: the point of
# this run is to confirm the on-disk conversion is what vLLM 0.8.5 actually reads.
# Usage: serve.sh <gpu-list> <ckpt-path> <port> <dp>
# GENCFG may override --override-generation-config (the 32B carries sampling defaults).
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export VLLM_CACHE_ROOT=/mnt/data01/muhsen/tmp/vllm-cache
GENCFG="${GENCFG:-{\"max_new_tokens\": 16384\}}"
exec "$B/.venv/bin/python" -m vllm.entrypoints.cli.main serve "$2" \
  --served-model-name "$2" \
  --port "$3" \
  --data-parallel-size "$4" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --hf-overrides '{"max_position_embeddings": 40960}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config "$GENCFG"
