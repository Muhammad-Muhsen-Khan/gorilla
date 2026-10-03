#!/usr/bin/env bash
# One vLLM server over several GPUs (data parallel). Usage: serve.sh <gpu-list> <ckpt-path> <port> <dp>
# Flags match runlogs/olive32k_out16k so these rows stay comparable to the olive
# SFT rows: 40960 context, 16384-token output cap.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export VLLM_CACHE_ROOT=/mnt/data01/muhsen/tmp/vllm-cache
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
  --override-generation-config '{"max_new_tokens": 16384}'
