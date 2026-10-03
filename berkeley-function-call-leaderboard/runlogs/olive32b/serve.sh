#!/usr/bin/env bash
# One vLLM server for a 32B checkpoint. Usage: serve.sh <gpu-list> <ckpt-path> <port> <dp>
# 32B bf16 is ~65.5 GB and an H200 has 143.7 GB, so each GPU holds a full replica:
# data parallel, not tensor parallel, which matches how the 4B rows were served.
#
# --override-generation-config cancels this checkpoint's sampling defaults
# (do_sample/temperature 0.7/top_k 20/top_p 0.8/repetition_penalty 1.05). The 4B
# rows ran with none of those active, and vLLM would otherwise apply them to every
# request, so leaving them in place would make the comparison meaningless.
# max_position_embeddings is already 131072 here, so --max-model-len alone matches
# the 4B's 40960 context; no --hf-overrides needed.
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
  --dtype bfloat16 \
  --trust-remote-code \
  --override-generation-config '{"max_new_tokens": 16384, "temperature": 1.0, "top_k": -1, "top_p": 1.0, "repetition_penalty": 1.0}'
