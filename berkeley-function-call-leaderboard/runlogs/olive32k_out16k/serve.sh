#!/usr/bin/env bash
# One vLLM server on ONE GPU for one checkpoint. Usage: serve.sh <gpu> <ckpt-path> <port>
#  --max-model-len 40960 + max_position_embeddings 40960: the converted config
#    carries 32768 (qwen-template); keepreason multi-turn prompts need more.
#  --override-generation-config: qwen-template's generation_config.json caps
#    max_new_tokens at 2048, which vLLM applies server-wide.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES="$1" HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
# Some hosts have a root-owned ~/.cache/vllm; keep vLLM's compile cache on shared scratch.
export VLLM_CACHE_ROOT=/mnt/data01/muhsen/tmp/vllm-cache
exec "$B/.venv/bin/python" -m vllm.entrypoints.cli.main serve "$2" \
  --served-model-name "$2" \
  --port "$3" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --hf-overrides '{"max_position_embeddings": 40960}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384}'
