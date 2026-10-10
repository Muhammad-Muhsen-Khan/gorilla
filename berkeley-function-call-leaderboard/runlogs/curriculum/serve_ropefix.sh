#!/usr/bin/env bash
# Serve a transformers-5.2.0-converted checkpoint with RoPE restored.
#
# 5.2.0 writes RoPE as a nested block:   "rope_parameters": {"rope_theta": 1000000, ...}
# vLLM 0.8.5 (pinned by BFCL) never reads that key -- the string appears nowhere
# in the package -- so it falls back to getattr(config, "rope_theta", ...) and the
# trained RoPE configuration is silently discarded. Models served this way emit
# coherent text at short context and token noise past ~4k prompt tokens.
# Injecting the flat rope_theta via --hf-overrides restores it without editing
# the checkpoint, and leaves eos/tokenizer/max_position_embeddings as trained.
# Usage: serve_ropefix.sh <gpu-list> <ckpt-path> <port> <dp>
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
  --hf-overrides '{"max_position_embeddings": 40960, "rope_theta": 1000000}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384}'
