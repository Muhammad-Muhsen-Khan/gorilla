#!/usr/bin/env bash
# One vLLM server, data-parallel across all 8 GPUs. Usage: serve.sh <ckpt-path> <port>
#  --max-model-len 40960 + max_position_embeddings 40960: the ORIGINAL context, so
#    this run differs from the 34.09 baseline only in the output cap.
#  --override-generation-config: the template's generation_config.json caps
#    max_new_tokens at 2048, which vLLM applies server-wide.
set -u
B=/mnt/data01/muhsen/tooling/gorilla/berkeley-function-call-leaderboard
cd "$B"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/mnt/data01/muhsen/hf-cache TMPDIR=/mnt/data01/muhsen/tmp
# avey-bm-02's ~/.cache/vllm/torch_compile_cache is root-owned (left by a docker run),
# so vLLM cannot write its compile cache there. Keep it on the shared scratch instead.
export VLLM_CACHE_ROOT=/mnt/data01/muhsen/tmp/vllm-cache
exec "$B/.venv/bin/python" -m vllm.entrypoints.cli.main serve "$1" \
  --served-model-name "$1" \
  --port "$2" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 40960 \
  --hf-overrides '{"max_position_embeddings": 40960}' \
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384}'
