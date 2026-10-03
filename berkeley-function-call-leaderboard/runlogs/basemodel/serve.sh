#!/usr/bin/env bash
# One vLLM server over N GPUs (data parallel). Usage: serve.sh <gpu-list> <ckpt-path> <port> <dp>
# Flags are copied from runlogs/olive32k_out16k so these rows stay directly
# comparable to the olive-32k SFT rows:
#   --max-model-len 40960 + max_position_embeddings 40960: the converted config
#     carries 32768; keepreason multi-turn prompts need more.
#   --override-generation-config: the converted generation_config.json caps
#     max_new_tokens at 2048, which vLLM would otherwise apply server-wide.
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
