#!/usr/bin/env bash
# One vLLM server on ONE GPU, modest gpu-memory-utilization so it coexists with
# other tenants on the node. Usage: serve1gpu.sh <gpu> <ckpt-path> <port>
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
export VLLM_CACHE_ROOT=/mnt/data01/muhsen/tmp/vllm-cache
exec "$B/.venv/bin/python" -m vllm.entrypoints.cli.main serve "$2" \
  --served-model-name "$2" \
  --port "$3" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 40960 \
  --hf-overrides '{"max_position_embeddings": 40960}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384}'
